// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/mcts/node.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <cmath>
#include <span>
#include "sage_sr/core/types.hpp"
#include "sage_sr/core/exp_queue.hpp"
#include "sage_sr/core/exp_tree.hpp"
#include "sage_sr/eval/timing.hpp"

namespace sage_sr {

struct MCTSNode {
    MCTSNode*  parent;
    uint8_t    move;           // operator index that led to this node (255 for root)
    int        visits = 0;
    bool       is_terminal_flag = false;
    float      prior = 1.0f;   // policy prior P(move|prefix) for PUCT; 1.0 = neutral
    bool       priors_computed = false;   // lazily fetched from the policy callback
    std::vector<float> priors;            // prob per op index (pset-sized)
    ExpQueue   path_queue;
    std::vector<uint8_t>                    unexpanded_moves;
    std::vector<std::unique_ptr<MCTSNode>>  children;

    MCTSNode(MCTSNode* parent, uint8_t move, int K)
        : parent(parent), move(move), path_queue(K) {}

    float ucb(float c, float gamma, float puct_c) const {
        float q = path_queue.best().reward;
        if (parent == nullptr || visits == 0) return q;
        float explore = std::pow(
            c * std::log(static_cast<float>(parent->visits))
              / static_cast<float>(visits),
            gamma);
        // PUCT prior term: puct_c * P(a|s) * sqrt(ln N_parent) / (1 + n).
        float puct = puct_c > 0.0f
            ? puct_c * prior * std::sqrt(std::log(static_cast<float>(parent->visits)))
                / (1.0f + static_cast<float>(visits))
            : 0.0f;
        return q + explore + puct;
    }

    bool is_leaf() const {
        return children.empty() || !unexpanded_moves.empty();
    }

    // Select child with highest UCB (skip terminal children)
    MCTSNode* choose(float c, float gamma, float puct_c) {
        MCTSNode* best = nullptr;
        float best_ucb = -1e30f;
        for (auto& ch : children) {
            if (ch->is_terminal_flag && ch->visits > 0) continue;
            float u = ch->ucb(c, gamma, puct_c);
            if (u > best_ucb) { best_ucb = u; best = ch.get(); }
        }
        if (best == nullptr && !children.empty())
            best = children[0].get();
        return best;
    }

    // Random non-terminal child
    MCTSNode* random_child(RandomGenerator& rng) {
        int valid_count = 0;
        for (const auto& ch : children) {
            if (!ch->is_terminal_flag) {
                ++valid_count;
            }
        }
        if (valid_count == 0) return children.empty() ? nullptr : children[0].get();

        int selected = static_cast<int>(rng() % static_cast<std::size_t>(valid_count));
        for (auto& ch : children) {
            if (ch->is_terminal_flag) {
                continue;
            }
            if (selected-- == 0) {
                return ch.get();
            }
        }
        return children.empty() ? nullptr : children[0].get();
    }

    // Propagate (reward, path) upward through the tree
    void backpropagate(std::span<const uint8_t> path, float reward) {
        ScopedTimer timer(TimingSection::MCTSBackpropagate);

        std::size_t prefix_len = 0;
        for (MCTSNode* cur = this; cur != nullptr && cur->parent != nullptr; cur = cur->parent) {
            ++prefix_len;
        }

        std::size_t accepted_nodes = 0;
        for (MCTSNode* cur = this; cur != nullptr; cur = cur->parent) {
            if (!cur->path_queue.may_accept_reward(reward)) {
                break;
            }
            ++accepted_nodes;
        }
        if (accepted_nodes == 0) {
            return;
        }

        auto full_path = std::make_shared<std::vector<uint8_t>>(prefix_len + path.size());
        std::copy(path.begin(), path.end(), full_path->begin() + static_cast<std::ptrdiff_t>(prefix_len));

        MCTSNode* cur = this;
        std::size_t index = prefix_len;
        while (cur != nullptr && cur->parent != nullptr) {
            (*full_path)[--index] = cur->move;
            cur = cur->parent;
        }

        cur = this;
        std::size_t start = prefix_len;
        for (std::size_t appended = 0; appended < accepted_nodes && cur != nullptr; ++appended) {
            if (!cur->path_queue.append_shared(full_path, start, reward)) {
                break;
            }
            if (cur->parent != nullptr) {
                --start;
            }
            cur = cur->parent;
        }
    }

    // Propagate downward: update children along a known good path
    void propagate(SharedPath path, float reward) {
        MCTSNode* cur = this;
        auto remaining = std::move(path);
        while (!remaining.empty()) {
            MCTSNode* found = nullptr;
            for (auto& ch : cur->children) {
                if (ch->move == remaining[0]) { found = ch.get(); break; }
            }
            if (!found) break;
            remaining = remaining.suffix(1);
            found->path_queue.append(remaining, reward);
            cur = found;
        }
    }

    void propagate(std::span<const uint8_t> path, float reward) {
        propagate(SharedPath(path), reward);
    }
};

} // namespace sage_sr
