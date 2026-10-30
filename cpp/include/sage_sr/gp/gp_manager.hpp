// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/gp/gp_manager.hpp
#pragma once
#include <vector>
#include <utility>
#include <span>
#include "sage_sr/core/types.hpp"
#include "sage_sr/core/symbol.hpp"
#include "sage_sr/core/exp_tree.hpp"

namespace sage_sr {

class GPManager {
public:
    explicit GPManager(const PrimitiveSet& pset);

    std::vector<uint8_t> node_replace(const ExpTree& state,
        std::span<uint8_t const> path, RandomGenerator& rng,
        int num_replacements = 1) const;

    std::vector<uint8_t> shrink_mutate(const ExpTree& state,
        std::span<uint8_t const> path, RandomGenerator& rng) const;

    std::vector<uint8_t> uniform_mutate(const ExpTree& state,
        std::span<uint8_t const> path, RandomGenerator& rng) const;

    std::vector<uint8_t> insert_mutate(const ExpTree& state,
        std::span<uint8_t const> path, RandomGenerator& rng) const;

    // nullptr / all-equal preserves the original uniform choice exactly.
    std::vector<uint8_t> mutate(const ExpTree& state,
        std::span<uint8_t const> path, RandomGenerator& rng,
        const float* mutation_weights = nullptr,
        int* selected_operator = nullptr) const;

    std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
    crossover(std::span<uint8_t const> p1,
              std::span<uint8_t const> p2,
              RandomGenerator& rng) const;

    int subtree_size(std::span<uint8_t const> path, int index) const;
    int depth_at(std::span<uint8_t const> path, int target) const;

private:
    const PrimitiveSet* pset_;
};

} // namespace sage_sr
