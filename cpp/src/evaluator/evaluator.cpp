// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// source/evaluator/evaluator.cpp
#include "sage_sr/evaluator/evaluator.hpp"
#include "sage_sr/eval/interpreter.hpp"
#include "sage_sr/eval/optimizer.hpp"
#include <cmath>
#include <numeric>

namespace sage_sr {

Evaluator::Evaluator(const PrimitiveSet& pset, const EvaluatorConfig& cfg)
    : dataset_(cfg.x_cols, cfg.y)
    , train_range_{0, cfg.y.size()}
    , bridge_(pset)
    , lm_iterations_(cfg.lm_iterations)
    , len_gamma_(cfg.len_gamma)
    , reward_normalize_(cfg.reward_normalize)
{
    target_values_ = dataset_.y;

    double mean = target_values_.mean();
    double var = (target_values_.array() - mean).square().mean();
    sigma_ = std::sqrt(var);
    if (sigma_ < 1e-10) sigma_ = 1.0;
}

float Evaluator::evaluate(std::span<uint8_t const> prefix, RandomGenerator& /*rng*/)
{
    return fitted_reward(prefix);
}

Tree Evaluator::build_optimized_tree(std::span<uint8_t const> prefix)
{
    const std::string key(prefix.begin(), prefix.end());
    if (const auto it = fitted_trees_.find(key); it != fitted_trees_.end())
        return it->second.tree;
    bridge_.to_tree(prefix, tree_workspace_);
    if (tree_workspace_.num_coefficients() > 0) {
        tree_workspace_ = CoefficientOptimizer::optimize(
            tree_workspace_, dataset_, train_range_, optimizer_workspace_, lm_iterations_);
    }
    Interpreter::evaluate(tree_workspace_, dataset_, train_range_, workspace_);
    fitted_trees_.emplace(key, CachedTree{tree_workspace_, compute_reward(workspace_.result(), prefix)});
    return tree_workspace_;
}

float Evaluator::fitted_reward(std::span<uint8_t const> prefix)
{
    const std::string key(prefix.begin(), prefix.end());
    if (!fitted_trees_.contains(key)) build_optimized_tree(prefix);
    return fitted_trees_.at(key).reward;
}

void Evaluator::seed_fitted_tree(std::span<uint8_t const> prefix,
                                 const std::vector<Scalar>& coefficients, float reward)
{
    Tree tree = bridge_.to_tree(prefix);
    if (tree.num_coefficients() != coefficients.size())
        throw std::invalid_argument("Inherited coefficient count does not match expression constants");
    tree.set_coefficients(coefficients);
    fitted_trees_.insert_or_assign(std::string(prefix.begin(), prefix.end()), CachedTree{std::move(tree), reward});
}

float Evaluator::compute_reward(const Eigen::VectorXd& y_pred,
                                std::span<uint8_t const> prefix) const
{
    if (y_pred.size() != target_values_.size()) return 0.0f;

    double mse = (y_pred - target_values_).squaredNorm()
                 / static_cast<double>(y_pred.size());
    const double error = std::sqrt(mse) / (reward_normalize_ ? sigma_ : 1.0);
    if (!std::isfinite(error)) return 0.0f;
    double r = 1.0 / (1.0 + error);
    if (len_gamma_ < 1.0f)
        r *= std::pow(static_cast<double>(len_gamma_),
                      static_cast<double>(prefix.size()));
    return static_cast<float>(r);
}

} // namespace sage_sr
