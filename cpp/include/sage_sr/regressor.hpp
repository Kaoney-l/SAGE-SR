// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/regressor.hpp
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include "sage_sr/core/types.hpp"
#include "sage_sr/core/symbol.hpp"
#include "sage_sr/evaluator/evaluator.hpp"
#include "sage_sr/gp/gp_manager.hpp"
#include "sage_sr/mcts/mcts.hpp"

namespace sage_sr {

struct RegressorConfig {
    std::vector<std::string> ops = {"+", "-", "*", "/", "sin", "cos", "exp", "log", "R"};
    int   max_depth        = 6;
    int   K                = 500;
    float c                = 4.0f;
    float gamma            = 0.5f;
    float gp_rate          = 0.5f;
    float mutation_rate    = 0.1f;
    float exploration_rate = 0.2f;
    int   max_unary        = 999;
    int   max_constants    = 999;
    int   lm_iterations    = 10;
    int   max_evals        = 100000;
    double max_time_sec    = 0.0;
    float succ_error_tol   = 1e-6f;
    float len_gamma        = 0.99f;
    bool reward_normalize  = false;
    float puct_c           = 0.0f;
    // Node replacement, shrink, subtree replacement, insertion.
    float mutation_weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

struct FitResult {
    std::vector<uint8_t> best_path;
    std::vector<Scalar>  best_coefficients;
    std::string          expression;
    float                best_reward;
    int                  n_evals;
    std::vector<std::vector<uint8_t>> top_paths;
    std::vector<float>                top_rewards;
    std::vector<float>                reward_curve;
    std::vector<SearchRecord>         search_records;
};

class FittedExpression {
public:
    FittedExpression(Tree tree, std::vector<uint8_t> path, float reward, int n_variables)
        : tree_(std::move(tree)), path_(std::move(path)), reward_(reward), n_variables_(n_variables) {}
    std::string expression() const { return tree_.to_string(); }
    std::vector<Scalar> coefficients() const { return tree_.get_coefficients(); }
    const std::vector<uint8_t>& path() const { return path_; }
    float train_reward() const { return reward_; }
    std::vector<Scalar> predict(const std::vector<std::vector<float>>& x_cols) const;
private:
    Tree tree_;
    std::vector<uint8_t> path_;
    float reward_;
    int n_variables_;
};

class Regressor {
public:
    Regressor(std::vector<std::vector<float>> x_cols,
              std::vector<float> y,
              RegressorConfig cfg = {});

    FitResult fit(std::optional<uint64_t> seed = std::nullopt);

    // (paths are op-index prefix expressions, rewards their fitted rewards)
    FitResult fit(uint64_t seed,
                  const std::vector<std::vector<uint8_t>>& init_paths,
                  const std::vector<float>& init_rewards,
                  const std::vector<std::vector<Scalar>>& init_coefficients = {},
                  const std::vector<std::vector<std::string>>& init_root_ids = {});

    // constants LM-optimized exactly like the search reward (for round
    // selection on held-out data)
    std::string expression_for(const std::vector<uint8_t>& path);
    FittedExpression fit_expression(const std::vector<uint8_t>& path);
    std::vector<uint8_t> legal_actions(const std::vector<uint8_t>& prefix) const;

    const PrimitiveSet& pset() const { return *pset_; }

    void set_policy_prior(MCTS::PolicyPriorFn fn) { policy_prior_ = std::move(fn); }

private:
    RegressorConfig                cfg_;
    std::unique_ptr<PrimitiveSet>  pset_;
    Evaluator       evaluator_;
    GPManager       gp_manager_;
    MCTS::PolicyPriorFn policy_prior_;

    FitResult fit_impl(uint64_t seed,
                       const std::vector<std::vector<uint8_t>>* init_paths,
                       const std::vector<float>* init_rewards,
                       const std::vector<std::vector<Scalar>>* init_coefficients = nullptr,
                       const std::vector<std::vector<std::string>>* init_root_ids = nullptr);
};

} // namespace sage_sr
