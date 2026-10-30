// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// source/regressor.cpp
#include "sage_sr/regressor.hpp"
#include <chrono>
#include <random>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace sage_sr {

namespace {
RegressorConfig validate_config(const RegressorConfig& cfg,
                                const std::vector<std::vector<float>>& x_cols,
                                const std::vector<float>& y) {
    if (x_cols.empty() || y.empty()) throw std::invalid_argument("Training data must be nonempty");
    if (x_cols.size() + cfg.ops.size() >= 255) throw std::invalid_argument("The grammar supports at most 254 symbols");
    for (const auto& col : x_cols) {
        if (col.size() != y.size()) throw std::invalid_argument("Training input and target lengths differ");
        for (float value : col) if (!std::isfinite(value)) throw std::invalid_argument("Training inputs must be finite");
    }
    for (float value : y) if (!std::isfinite(value)) throw std::invalid_argument("Training targets must be finite");
    if (cfg.K <= 0 || cfg.max_depth <= 0 || cfg.max_evals <= 0 || cfg.lm_iterations <= 0)
        throw std::invalid_argument("Queue size, depth, evaluation budget and LM iterations must be positive");
    if (cfg.max_unary < 0 || cfg.max_constants < 0) throw std::invalid_argument("Grammar limits must be nonnegative");
    if (!std::isfinite(cfg.len_gamma) || cfg.len_gamma <= 0 || cfg.len_gamma > 1)
        throw std::invalid_argument("Parsimony coefficient must lie in (0, 1]");
    for (float value : {cfg.gp_rate, cfg.mutation_rate, cfg.exploration_rate})
        if (!std::isfinite(value) || value < 0 || value > 1) throw std::invalid_argument("Search probabilities must lie in [0, 1]");
    if (!std::isfinite(cfg.c) || cfg.c < 0 || !std::isfinite(cfg.gamma) || cfg.gamma <= 0 ||
        !std::isfinite(cfg.puct_c) || cfg.puct_c < 0)
        throw std::invalid_argument("Exploration parameters are invalid");
    return cfg;
}
}

Regressor::Regressor(std::vector<std::vector<float>> x_cols,
                     std::vector<float> y,
                     RegressorConfig cfg)
    : cfg_(validate_config(cfg, x_cols, y))
    , pset_(std::make_unique<PrimitiveSet>(make_primitive_set(cfg_.ops, static_cast<int>(x_cols.size()))))
    , evaluator_(*pset_, EvaluatorConfig{x_cols, y, cfg_.lm_iterations, cfg_.len_gamma, cfg_.reward_normalize})
    , gp_manager_(*pset_)
{}

FitResult Regressor::fit(std::optional<uint64_t> seed) {
    return fit_impl(seed.value_or(std::random_device{}()), nullptr, nullptr);
}

FitResult Regressor::fit(uint64_t seed,
                         const std::vector<std::vector<uint8_t>>& init_paths,
                         const std::vector<float>& init_rewards,
                         const std::vector<std::vector<Scalar>>& init_coefficients,
                         const std::vector<std::vector<std::string>>& init_root_ids) {
    return fit_impl(seed, &init_paths, &init_rewards, &init_coefficients, &init_root_ids);
}

FitResult Regressor::fit_impl(uint64_t s,
                              const std::vector<std::vector<uint8_t>>* init_paths,
                              const std::vector<float>* init_rewards,
                              const std::vector<std::vector<Scalar>>* init_coefficients,
                              const std::vector<std::vector<std::string>>* init_root_ids) {
    RandomGenerator rng{s};

    MCTSConfig mcts_cfg{
        .K                = cfg_.K,
        .c                = cfg_.c,
        .gamma            = cfg_.gamma,
        .gp_rate          = cfg_.gp_rate,
        .mutation_rate    = cfg_.mutation_rate,
        .exploration_rate = cfg_.exploration_rate,
        .succ_error_tol   = cfg_.succ_error_tol,
        .max_evals        = cfg_.max_evals,
        .puct_c           = cfg_.puct_c,
    };
    for (int k = 0; k < 4; ++k)
        mcts_cfg.mutation_weights[k] = cfg_.mutation_weights[k];
    MCTS mcts(*pset_, evaluator_, gp_manager_, mcts_cfg);
    if (policy_prior_) mcts.set_policy_prior(policy_prior_);
    ExpTree tree(*pset_, cfg_.max_depth, cfg_.max_unary, cfg_.max_constants);
    if (init_paths && !init_paths->empty()) {
        if (init_paths->size() != init_rewards->size())
            throw std::invalid_argument("Inherited paths and rewards must have equal lengths");
        if (init_coefficients && !init_coefficients->empty() && init_coefficients->size() != init_paths->size())
            throw std::invalid_argument("Inherited coefficient lists must match inherited paths");
        if (init_root_ids && !init_root_ids->empty() && init_root_ids->size() != init_paths->size())
            throw std::invalid_argument("Inherited ancestry lists must match inherited paths");
        for (std::size_t i = 0; i < init_paths->size(); ++i) {
            if (!tree.can_add_path((*init_paths)[i])) throw std::invalid_argument("Inherited expression violates the grammar");
            if (!std::isfinite((*init_rewards)[i]) || (*init_rewards)[i] < 0 || (*init_rewards)[i] > 1)
                throw std::invalid_argument("Inherited reward must lie in [0, 1]");
            if (init_coefficients && !init_coefficients->empty())
                evaluator_.seed_fitted_tree((*init_paths)[i], (*init_coefficients)[i], (*init_rewards)[i]);
        }
        mcts.inject_elites(tree, *init_paths, *init_rewards,
                          init_root_ids ? *init_root_ids : std::vector<std::vector<std::string>>{});
    }
    const bool has_time_limit = cfg_.max_time_sec > 0.0;
    const auto start = std::chrono::steady_clock::now();

    while (mcts.count() < cfg_.max_evals) {
        if (has_time_limit) {
            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            if (elapsed.count() >= cfg_.max_time_sec) break;
        }
        float best = mcts.search(tree, rng);
        if (1.0f - best < cfg_.succ_error_tol) break;
    }

    auto best_path = mcts.best_path();
    std::vector<std::vector<uint8_t>> top_paths;
    std::vector<float> top_rewards;
    for (auto& [p, r] : mcts.top_paths(8)) {
        top_paths.push_back(std::move(p));
        top_rewards.push_back(r);
    }
    std::vector<Scalar> best_coefficients;
    std::string expression;
    if (!best_path.empty()) {
        auto best_tree = evaluator_.build_optimized_tree(best_path);
        best_coefficients = best_tree.get_coefficients();
        expression = best_tree.to_string();
    }

    return FitResult{
        std::move(best_path),
        std::move(best_coefficients),
        std::move(expression),
        mcts.best_reward(),
        mcts.count(),
        std::move(top_paths),
        std::move(top_rewards),
        mcts.reward_curve(),
        mcts.search_records(),
    };
}

std::string Regressor::expression_for(const std::vector<uint8_t>& path) {
    if (path.empty()) return "";
    return fit_expression(path).expression();
}

FittedExpression Regressor::fit_expression(const std::vector<uint8_t>& path) {
    const ExpTree grammar(*pset_, cfg_.max_depth, cfg_.max_unary, cfg_.max_constants);
    if (path.empty() || !grammar.can_add_path(path))
        throw std::invalid_argument("Expression path must be a complete grammatical expression");
    auto tree = evaluator_.build_optimized_tree(path);
    return FittedExpression(std::move(tree), path, evaluator_.fitted_reward(path),
                            static_cast<int>(pset_->symbols.size() - cfg_.ops.size()));
}

std::vector<uint8_t> Regressor::legal_actions(const std::vector<uint8_t>& prefix) const {
    ExpTree grammar(*pset_, cfg_.max_depth, cfg_.max_unary, cfg_.max_constants);
    for (uint8_t op : prefix) {
        if (grammar.is_terminal() || !grammar.try_add_op(op))
            throw std::invalid_argument("Prefix violates the expression grammar");
    }
    return grammar.is_terminal() ? std::vector<uint8_t>{} : grammar.available_ops();
}

std::vector<Scalar> FittedExpression::predict(const std::vector<std::vector<float>>& x_cols) const {
    if (static_cast<int>(x_cols.size()) != n_variables_)
        throw std::invalid_argument("Prediction input has the wrong variable count");
    const std::size_t n = x_cols.empty() ? 0 : x_cols.front().size();
    for (const auto& col : x_cols) if (col.size() != n)
        throw std::invalid_argument("Prediction input columns must have equal lengths");
    if (n == 0) return {};
    Dataset data(x_cols, std::vector<float>(n, 0.0f));
    const auto values = Interpreter::evaluate(tree_, data, Range{0, n});
    return {values.data(), values.data() + values.size()};
}

} // namespace sage_sr
