// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/mcts/mcts.hpp
#pragma once
#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include "sage_sr/core/types.hpp"
#include "sage_sr/mcts/node.hpp"
#include "sage_sr/evaluator/evaluator.hpp"
#include "sage_sr/gp/gp_manager.hpp"

namespace sage_sr {

struct MCTSConfig {
    int   K                = 500;
    float c                = 4.0f;
    float gamma            = 0.5f;
    float gp_rate          = 0.2f;
    float mutation_rate    = 0.2f;
    float exploration_rate = 0.2f;
    float succ_error_tol   = 1e-6f;
    int max_evals = 100000;
    float puct_c           = 0.0f;  // weight of the policy-prior PUCT term (0 = off)
    // Order: {node_replace, shrink, uniform, insert}. All equal = original
    // uniform choice; rescaled by the semantic controller's structure-family
    // guidance (e.g. rational-family prior upweights insert/node_replace).
    float mutation_weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

struct SearchRecord {
    std::string kind;
    std::string operation;
    std::vector<uint8_t> path;
    std::vector<std::vector<uint8_t>> parents;
    std::vector<float> parent_rewards;
    std::vector<std::string> root_ids;
    float reward = 0.0f;
    int evaluation = 0;
};

class MCTS {
public:
    // Returns a probability vector over op indices given the move prefix
    // from the root (used as PUCT prior + prior-weighted expansion).
    using PolicyPriorFn =
        std::function<std::vector<float>(const std::vector<uint8_t>& prefix)>;

    MCTS(const PrimitiveSet& pset,
         Evaluator& evaluator,
         GPManager& gp_manager,
         MCTSConfig cfg = {});

    float search(ExpTree& tree, RandomGenerator& rng);

    float best_reward()  const { return best_reward_; }
    int   count()        const { return count_num_; }
    int   total_nodes()  const { return total_nodes_; }
    int   prior_calls()  const { return prior_calls_; }
    const std::vector<float>& reward_curve() const { return reward_curve_; }
    const std::vector<SearchRecord>& search_records() const { return records_; }

    std::vector<uint8_t> best_path() const {
        return root_->path_queue.best().path.to_vector();
    }

    std::vector<std::pair<std::vector<uint8_t>, float>>
    top_paths(int k) const;

    void set_policy_prior(PolicyPriorFn fn) { policy_prior_ = std::move(fn); }

    // Creates the child chain along each valid path, fills every node's
    // queue, and seeds best_reward_ (no cross-round regression).
    void inject_elites(ExpTree& tree,
                       const std::vector<std::vector<uint8_t>>& paths,
                       const std::vector<float>& rewards,
                       const std::vector<std::vector<std::string>>& root_ids = {});

private:
    float rollout_once(ExpTree& state,
                       RandomGenerator& rng,
                       std::vector<uint8_t>& out_path);
    bool rollout_once(ExpTree& state,
                      std::span<uint8_t const> given_path,
                      RandomGenerator& rng,
                      std::vector<uint8_t>& out_path,
                      float& reward,
                      bool path_is_valid = false);

    float perform_mutation(MCTSNode* node, ExpTree& state, RandomGenerator& rng);
    float perform_crossover(MCTSNode* node, ExpTree& state, RandomGenerator& rng);
    void  update_terminal_status(MCTSNode* node);

    MCTSNode* expand_node(MCTSNode* node, ExpTree& state, RandomGenerator& rng);
    void record_evaluation(const std::string& kind, const std::string& operation,
                           const std::vector<uint8_t>& path, float reward,
                           const std::vector<std::vector<uint8_t>>& parents = {},
                           const std::vector<float>& parent_rewards = {});

    const PrimitiveSet* pset_;
    Evaluator*          evaluator_;
    GPManager*          gp_manager_;
    MCTSConfig          cfg_;

    std::unique_ptr<MCTSNode> root_;
    PolicyPriorFn policy_prior_;
    // convergence curve; one entry per search() call)
    std::vector<float> reward_curve_;
    std::vector<SearchRecord> records_;
    std::unordered_map<std::string, std::vector<std::string>> roots_;
    int   count_num_   = 0;
    int   total_nodes_ = 1;
    int   prior_calls_ = 0;
    float best_reward_ = -std::numeric_limits<float>::infinity();
};

} // namespace sage_sr
