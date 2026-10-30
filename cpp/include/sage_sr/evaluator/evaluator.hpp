// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/evaluator/evaluator.hpp
#pragma once
#include <vector>
#include <span>
#include <string>
#include <unordered_map>
#include "sage_sr/core/types.hpp"
#include "sage_sr/core/symbol.hpp"
#include "sage_sr/core/bridge.hpp"
#include "sage_sr/core/dataset.hpp"
#include "sage_sr/eval/interpreter.hpp"

namespace sage_sr {

struct EvaluatorConfig {
    std::vector<std::vector<float>> x_cols;   // shape: [n_vars][n_samples]
    std::vector<float>              y;         // shape: [n_samples]
    int lm_iterations = 10;
    float len_gamma = 0.99f;
    bool reward_normalize = false;
};

class Evaluator {
public:
    Evaluator(const PrimitiveSet& pset, const EvaluatorConfig& cfg);

    float evaluate(std::span<uint8_t const> prefix, RandomGenerator& rng);
    Tree  build_optimized_tree(std::span<uint8_t const> prefix);
    float fitted_reward(std::span<uint8_t const> prefix);
    void seed_fitted_tree(std::span<uint8_t const> prefix,
                          const std::vector<Scalar>& coefficients, float reward);

private:
    float compute_reward(const Eigen::VectorXd& y_pred,
                         std::span<uint8_t const> prefix) const;

    Dataset dataset_;
    Range   train_range_;
    Bridge  bridge_;
    Tree    tree_workspace_;
    InterpreterWorkspace workspace_;
    InterpreterWorkspace optimizer_workspace_;
    Eigen::VectorXd target_values_;
    double  sigma_;
    int     lm_iterations_;
    float   len_gamma_;
    bool reward_normalize_;
    struct CachedTree { Tree tree; float reward; };
    std::unordered_map<std::string, CachedTree> fitted_trees_;
};

} // namespace sage_sr
