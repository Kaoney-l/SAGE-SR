// SPDX-License-Identifier: MIT
// source/mcts/mcts.cpp
#include "sage_sr/mcts/mcts.hpp"
#include "sage_sr/eval/timing.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace sage_sr {

namespace {

bool is_valid_rollout_path(const ExpTree& state, std::span<uint8_t const> path) {
    return state.can_add_path(path);
}

std::string path_key(const std::vector<uint8_t>& path) {
    return std::string(path.begin(), path.end());
}

std::string origin_id(const std::vector<uint8_t>& path) {
    constexpr char hex[] = "0123456789abcdef";
    std::string id = "structure:";
    for (uint8_t op : path) { id += hex[op >> 4]; id += hex[op & 15]; }
    return id;
}

std::vector<uint8_t> full_path(const ExpTree& prefix, std::span<const uint8_t> suffix) {
    auto path = prefix.get_op_list();
    path.insert(path.end(), suffix.begin(), suffix.end());
    return path;
}

} // namespace

MCTS::MCTS(const PrimitiveSet& pset, Evaluator& evaluator,
           GPManager& gp_manager, MCTSConfig cfg)
    : pset_(&pset), evaluator_(&evaluator), gp_manager_(&gp_manager), cfg_(cfg)
    , root_(std::make_unique<MCTSNode>(nullptr, 255, cfg.K))
{}

MCTSNode* MCTS::expand_node(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    if (node->unexpanded_moves.empty())
        node->unexpanded_moves = state.available_ops();

    if (policy_prior_ && !node->priors_computed) {
        std::vector<uint8_t> prefix;
        for (MCTSNode* cur = node; cur->parent != nullptr; cur = cur->parent)
            prefix.push_back(cur->move);
        std::reverse(prefix.begin(), prefix.end());
        node->priors = policy_prior_(prefix);
        node->priors_computed = true;
        ++prior_calls_;
    }

    int idx;
    if (node->priors_computed) {
        // Restrict the fused action distribution to unexpanded legal moves.
        std::vector<float> w(node->unexpanded_moves.size());
        float total = 0.0f;
        for (std::size_t k = 0; k < w.size(); ++k) {
            const uint8_t mv = node->unexpanded_moves[k];
            float p = (mv < node->priors.size()) ? node->priors[mv] : 0.0f;
            if (!std::isfinite(p) || p < 0.0f) p = 0.0f;
            w[k] = p;
            total += w[k];
        }
        if (total <= 0.0f) { std::fill(w.begin(), w.end(), 1.0f); total = static_cast<float>(w.size()); }
        const float r = static_cast<float>(rng() % 1000000u) / 1000000.0f * total;
        float acc = 0.0f;
        idx = static_cast<int>(w.size()) - 1;
        for (std::size_t k = 0; k < w.size(); ++k) {
            acc += w[k];
            if (r < acc) { idx = static_cast<int>(k); break; }
        }
    } else {
        idx = static_cast<int>(rng() % node->unexpanded_moves.size());
    }
    uint8_t mv = node->unexpanded_moves[idx];
    node->unexpanded_moves.erase(node->unexpanded_moves.begin() + idx);
    node->children.push_back(std::make_unique<MCTSNode>(node, mv, cfg_.K));
    node->children.back()->prior =
        (node->priors_computed && mv < node->priors.size()) ? node->priors[mv] : 1.0f;
    ++total_nodes_;
    return node->children.back().get();
}

float MCTS::search(ExpTree& tree, RandomGenerator& rng) {
    ScopedTimer timer(TimingSection::MCTSSearch);
    if (count_num_ >= cfg_.max_evals) return best_reward_;
    MCTSNode* node = root_.get();
    ++node->visits;

    ExpTree state = tree;

    // --- Selection ---
    while (!node->is_leaf()) {
        float u1 = static_cast<float>(rng() % 10000) / 10000.0f;
        if (u1 < cfg_.gp_rate && count_num_ < cfg_.max_evals) {
            float u2 = static_cast<float>(rng() % 10000) / 10000.0f;
            float gp_reward;
            if (u2 < cfg_.mutation_rate)
                gp_reward = perform_mutation(node, state, rng);
            else
                gp_reward = perform_crossover(node, state, rng);
            best_reward_ = std::max(best_reward_, gp_reward);
        }
        if (count_num_ >= cfg_.max_evals) return best_reward_;

        float u3 = static_cast<float>(rng() % 10000) / 10000.0f;
        MCTSNode* next;
        if (u3 < cfg_.exploration_rate)
            next = node->random_child(rng);
        else
            next = node->choose(cfg_.c, cfg_.gamma, cfg_.puct_c);

        if (!next) break;
        node = next;
        ++node->visits;
        state.add_op(node->move);
    }

    // --- Expansion ---
    if (!state.is_terminal()) {
        node = expand_node(node, state, rng);
        ++node->visits;
        state.add_op(node->move);
    }

    // --- Simulation & Backpropagation ---
    std::vector<uint8_t> path;
    float sim_reward = rollout_once(state, rng, path);
    best_reward_ = std::max(best_reward_, sim_reward);

    if (path.empty())
        node->is_terminal_flag = true;
    node->backpropagate(path, sim_reward);

    update_terminal_status(node);

    if (node->parent) {
        for (const auto& entry : node->parent->path_queue.entries())
            node->parent->propagate(entry.path, entry.reward);
    }

    return best_reward_;
}

float MCTS::rollout_once(
    ExpTree& state,
    RandomGenerator& rng,
    std::vector<uint8_t>& out_path)
{
    ScopedTimer timer(TimingSection::MCTSRollout);
    ExpTree cloned = state;
    out_path.clear();
    if (!policy_prior_) {
        cloned.random_fill(rng, out_path);
    } else {
        while (!cloned.is_terminal()) {
            const auto& legal = cloned.available_ops();
            if (legal.empty()) throw std::runtime_error("Grammar has no legal completion");
            const auto priors = policy_prior_(cloned.get_op_list());
            ++prior_calls_;
            std::vector<float> weights;
            float total = 0.0f;
            for (uint8_t op : legal) {
                float value = op < priors.size() ? priors[op] : 0.0f;
                if (!std::isfinite(value) || value < 0.0f) value = 0.0f;
                weights.push_back(value);
                total += weights.back();
            }
            if (total <= 0.0f) { std::fill(weights.begin(), weights.end(), 1.0f); total = static_cast<float>(weights.size()); }
            const float draw = static_cast<float>(rng() % 1000000u) / 1000000.0f * total;
            float cumulative = 0.0f;
            std::size_t selected = legal.size() - 1;
            for (std::size_t i = 0; i < legal.size(); ++i) {
                cumulative += weights[i];
                if (draw < cumulative) { selected = i; break; }
            }
            const auto op = legal[selected];
            out_path.push_back(op);
            cloned.add_available_op(op);
        }
    }
    const float reward = evaluator_->evaluate(cloned.get_op_list(), rng);
    ++count_num_;
    record_evaluation("rollout", "completion", cloned.get_op_list(), reward);
    return reward;
}

bool MCTS::rollout_once(
    ExpTree& state,
    std::span<uint8_t const> given_path,
    RandomGenerator& rng,
    std::vector<uint8_t>& out_path,
    float& reward,
    bool path_is_valid)
{
    ScopedTimer timer(TimingSection::MCTSRollout);
    if (count_num_ >= cfg_.max_evals) return false;
    if (!path_is_valid && !state.can_add_path(given_path)) {
        out_path.clear();
        reward = 0.0f;
        return false;
    }

    ExpTree cloned = state;
    for (uint8_t op : given_path) {
        cloned.add_available_op(op);
    }
    out_path.assign(given_path.begin(), given_path.end());
    reward = evaluator_->evaluate(cloned.get_op_list(), rng);
    return true;
}

float MCTS::perform_mutation(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    ScopedTimer timer(TimingSection::MCTSMutation);
    if (node->path_queue.is_empty()) return 0.0f;
    const auto& old_entry = node->path_queue.random_sample(rng);
    const auto parent = full_path(state, old_entry.path.span());
    const float parent_reward = old_entry.reward;
    int selected_operator = 2;
    auto new_path = gp_manager_->mutate(state, old_entry.path.span(), rng,
                                        cfg_.mutation_weights, &selected_operator);
    std::vector<uint8_t> out_path;
    float reward = 0.0f;
    if (!rollout_once(state, std::span<const uint8_t>(new_path), rng, out_path, reward)) {
        return 0.0f;
    }
    ++count_num_;
    static constexpr const char* names[] = {"node_replace", "shrink", "subtree_replace", "insert"};
    record_evaluation("mutation", names[selected_operator], full_path(state, out_path), reward,
                      {parent}, {parent_reward});
    node->backpropagate(out_path, reward);
    node->propagate(std::span<const uint8_t>(out_path), reward);
    return reward;
}

float MCTS::perform_crossover(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    ScopedTimer timer(TimingSection::MCTSCrossover);
    if (node->path_queue.is_empty()) return 0.0f;
    const auto& e1 = node->path_queue.random_sample(rng);
    const auto& e2 = node->path_queue.random_sample(rng);
    const std::vector<std::vector<uint8_t>> parents = {full_path(state, e1.path.span()), full_path(state, e2.path.span())};
    const std::vector<float> parent_rewards = {e1.reward, e2.reward};

    constexpr int kMaxValidityResamples = 4;
    std::vector<uint8_t> np1, np2;
    bool valid1 = false;
    bool valid2 = false;

    for (int attempt = 0; attempt < kMaxValidityResamples; ++attempt) {
        std::tie(np1, np2) = gp_manager_->crossover(e1.path.span(), e2.path.span(), rng);
        valid1 = is_valid_rollout_path(state, np1);
        valid2 = is_valid_rollout_path(state, np2);
        if (valid1 || valid2) {
            break;
        }
    }

    float best = 0.0f;
    for (const auto& [path, is_valid] : {
             std::pair<std::vector<uint8_t> const&, bool>{np1, valid1},
             std::pair<std::vector<uint8_t> const&, bool>{np2, valid2},
         }) {
        if (!is_valid) {
            continue;
        }
        std::vector<uint8_t> out_path;
        float reward = 0.0f;
        if (!rollout_once(state, std::span<const uint8_t>(path), rng, out_path, reward, true)) {
            continue;
        }
        ++count_num_;
        record_evaluation("crossover", "subtree_exchange", full_path(state, out_path), reward,
                          parents, parent_rewards);
        node->backpropagate(out_path, reward);
        node->propagate(std::span<const uint8_t>(out_path), reward);
        best = std::max(best, reward);
    }
    return best;
}

std::vector<std::pair<std::vector<uint8_t>, float>>
MCTS::top_paths(int k) const {
    std::vector<std::pair<std::vector<uint8_t>, float>> out;
    const auto& entries = root_->path_queue.entries();
    std::vector<const QueueEntry*> sorted;
    sorted.reserve(entries.size());
    for (const auto& e : entries) sorted.push_back(&e);
    std::sort(sorted.begin(), sorted.end(),
              [](const QueueEntry* a, const QueueEntry* b) {
                  return a->reward > b->reward;
              });
    for (int i = 0; i < k && i < static_cast<int>(sorted.size()); ++i) {
        out.emplace_back(sorted[i]->path.to_vector(), sorted[i]->reward);
    }
    return out;
}

void MCTS::inject_elites(ExpTree& tree,
                         const std::vector<std::vector<uint8_t>>& paths,
                         const std::vector<float>& rewards,
                         const std::vector<std::vector<std::string>>& root_ids) {
    std::vector<uint8_t> root_used;
    for (std::size_t i = 0; i < paths.size() && i < rewards.size(); ++i) {
        const auto& path = paths[i];
        const float r = rewards[i];
        if (path.empty() || !tree.can_add_path(path)) continue;
        roots_[path_key(path)] = i < root_ids.size() && !root_ids[i].empty()
            ? root_ids[i] : std::vector<std::string>{origin_id(path)};

        MCTSNode* cur = root_.get();
        ExpTree state = tree;
        for (std::size_t k = 0; k < path.size(); ++k) {
            const uint8_t op = path[k];
            if (k == 0) root_used.push_back(op);
            // chain nodes must keep their other branches expandable
            if (cur->unexpanded_moves.empty() && !cur->children.empty()) {
                auto avail = state.available_ops();
                for (const auto& ch : cur->children)
                    avail.erase(std::remove(avail.begin(), avail.end(), ch->move),
                                avail.end());
                cur->unexpanded_moves = std::move(avail);
            }
            MCTSNode* next = nullptr;
            for (auto& ch : cur->children)
                if (ch->move == op) { next = ch.get(); break; }
            if (!next) {
                cur->children.push_back(std::make_unique<MCTSNode>(cur, op, cfg_.K));
                next = cur->children.back().get();
                ++total_nodes_;
            }
            cur = next;
            state.add_op(op);
        }
        root_->path_queue.append(std::span<const uint8_t>(path), r);
        root_->propagate(std::span<const uint8_t>(path), r);
        best_reward_ = std::max(best_reward_, r);
    }
    // root must still branch to moves outside the injected first-moves
    if (!root_used.empty() && root_->unexpanded_moves.empty()) {
        auto avail = tree.available_ops();
        for (uint8_t m : root_used)
            avail.erase(std::remove(avail.begin(), avail.end(), m), avail.end());
        root_->unexpanded_moves = std::move(avail);
    }
}

void MCTS::record_evaluation(const std::string& kind, const std::string& operation,
                              const std::vector<uint8_t>& path, float reward,
                              const std::vector<std::vector<uint8_t>>& parents,
                              const std::vector<float>& parent_rewards) {
    auto roots = roots_[path_key(path)];
    for (const auto& parent : parents) {
        const auto it = roots_.find(path_key(parent));
        if (it == roots_.end()) roots.push_back(origin_id(parent));
        else roots.insert(roots.end(), it->second.begin(), it->second.end());
    }
    if (roots.empty()) roots.push_back(origin_id(path));
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    roots_[path_key(path)] = roots;
    records_.push_back(SearchRecord{kind, operation, path, parents, parent_rewards,
                                    std::move(roots), reward, count_num_});
    best_reward_ = std::max(best_reward_, reward);
    reward_curve_.push_back(best_reward_);
}

void MCTS::update_terminal_status(MCTSNode* node) {
    MCTSNode* cur = node;
    while (cur && cur->parent) {
        MCTSNode* par = cur->parent;
        bool all_terminal = !par->children.empty() && par->unexpanded_moves.empty();
        if (all_terminal) {
            for (auto& ch : par->children)
                if (!ch->is_terminal_flag) { all_terminal = false; break; }
        }
        if (all_terminal && !par->is_terminal_flag) {
            par->is_terminal_flag = true;
            cur = par;
        } else { break; }
    }
}

} // namespace sage_sr
