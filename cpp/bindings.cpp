// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sage_sr/eval/timing.hpp"
#include "sage_sr/regressor.hpp"
#include <string>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace sage_sr;

PYBIND11_MODULE(_core, m) {
    m.doc() = "SAGE-SR numerical expression-search backend";

    m.def("openmp_info", [] {
        py::dict info;
#ifdef _OPENMP
        info["enabled"] = true;
        info["max_threads"] = omp_get_max_threads();
        info["num_procs"] = omp_get_num_procs();
#else
        info["enabled"] = false;
#endif
        return info;
    });

    m.def("reset_timing_stats", &reset_timing_stats);
    m.def("timing_stats", [] {
        py::dict stats;
        for (const auto& entry : timing_stats()) {
            py::dict section;
            section["calls"] = entry.calls;
            section["total_seconds"] = entry.total_seconds;
            section["average_seconds"] = entry.calls == 0
                ? 0.0
                : entry.total_seconds / static_cast<double>(entry.calls);
            stats[py::str(std::string(entry.name))] = section;
        }
        return stats;
    });

    py::class_<RegressorConfig>(m, "RegressorConfig")
        .def(py::init<>())
        .def_readwrite("ops",               &RegressorConfig::ops)
        .def_readwrite("max_depth",         &RegressorConfig::max_depth)
        .def_readwrite("K",                 &RegressorConfig::K)
        .def_readwrite("c",                 &RegressorConfig::c)
        .def_readwrite("gamma",             &RegressorConfig::gamma)
        .def_readwrite("gp_rate",           &RegressorConfig::gp_rate)
        .def_readwrite("mutation_rate",     &RegressorConfig::mutation_rate)
        .def_readwrite("exploration_rate",  &RegressorConfig::exploration_rate)
        .def_readwrite("max_unary",         &RegressorConfig::max_unary)
        .def_readwrite("max_constants",     &RegressorConfig::max_constants)
        .def_readwrite("lm_iterations",     &RegressorConfig::lm_iterations)
        .def_readwrite("max_evals",         &RegressorConfig::max_evals)
        .def_readwrite("max_time_sec",      &RegressorConfig::max_time_sec)
        .def_readwrite("succ_error_tol",    &RegressorConfig::succ_error_tol)
        .def_readwrite("len_gamma",         &RegressorConfig::len_gamma)
        .def_readwrite("reward_normalize",  &RegressorConfig::reward_normalize)
        .def_readwrite("puct_c",            &RegressorConfig::puct_c)
        .def("set_mutation_weights", [](RegressorConfig& cfg,
                                        std::vector<float> w) {
            if (w.size() != 4) throw py::value_error("Provide four mutation weights");
            for (float value : w) if (!std::isfinite(value) || value <= 0)
                throw py::value_error("Mutation weights must be finite and positive");
            for (int k = 0; k < 4; ++k)
                cfg.mutation_weights[k] = w[k];
        }, py::arg("w"));

    py::class_<SearchRecord>(m, "SearchRecord")
        .def_readonly("kind", &SearchRecord::kind)
        .def_readonly("operation", &SearchRecord::operation)
        .def_readonly("path", &SearchRecord::path)
        .def_readonly("parents", &SearchRecord::parents)
        .def_readonly("parent_rewards", &SearchRecord::parent_rewards)
        .def_readonly("root_ids", &SearchRecord::root_ids)
        .def_readonly("reward", &SearchRecord::reward)
        .def_readonly("evaluation", &SearchRecord::evaluation);

    py::class_<FittedExpression>(m, "FittedExpression")
        .def_property_readonly("expression", &FittedExpression::expression)
        .def_property_readonly("path", &FittedExpression::path)
        .def_property_readonly("coefficients", &FittedExpression::coefficients)
        .def_property_readonly("train_reward", &FittedExpression::train_reward)
        .def("predict", [](const FittedExpression& self, py::array_t<float> x) {
            if (x.ndim() != 2) throw py::value_error("Use column-major feature input [features, samples]");
            auto values = x.unchecked<2>();
            std::vector<std::vector<float>> columns(x.shape(0), std::vector<float>(x.shape(1)));
            for (py::ssize_t i = 0; i < x.shape(0); ++i)
                for (py::ssize_t j = 0; j < x.shape(1); ++j) columns[i][j] = values(i,j);
            return self.predict(columns);
        }, py::arg("x"));

    py::class_<FitResult>(m, "FitResult")
        .def_readonly("best_path",   &FitResult::best_path)
        .def_readonly("best_coefficients", &FitResult::best_coefficients)
        .def_readonly("expression",  &FitResult::expression)
        .def_readonly("best_reward", &FitResult::best_reward)
        .def_readonly("n_evals",     &FitResult::n_evals)
        .def_readonly("top_paths",   &FitResult::top_paths)
        .def_readonly("top_rewards", &FitResult::top_rewards)
        .def_readonly("reward_curve", &FitResult::reward_curve)
        .def_readonly("search_records", &FitResult::search_records);

    py::class_<Regressor>(m, "Regressor")
        .def(py::init([](
            py::array_t<float> x_train,
            py::array_t<float> y_train,
            const RegressorConfig& cfg)
        {
            if (x_train.ndim() != 2 || y_train.ndim() != 1 || x_train.shape(1) != y_train.shape(0))
                throw py::value_error("Training arrays must have shape [features, samples] and [samples]");
            auto x_buf = x_train.unchecked<2>();
            auto y_buf = y_train.unchecked<1>();
            int n_vars    = static_cast<int>(x_train.shape(0));
            int n_samples = static_cast<int>(x_train.shape(1));

            std::vector<std::vector<float>> x_cols(n_vars, std::vector<float>(n_samples));
            for (int i = 0; i < n_vars; i++)
                for (int j = 0; j < n_samples; j++)
                    x_cols[i][j] = x_buf(i, j);

            std::vector<float> y(n_samples);
            for (int j = 0; j < n_samples; j++)
                y[j] = y_buf(j);

            return Regressor(std::move(x_cols), std::move(y), cfg);
        }),
        py::arg("x_train"), py::arg("y_train"),
        py::arg("cfg") = RegressorConfig{})
        .def("fit", [](Regressor& self, py::object seed,
                       py::object init_paths, py::object init_rewards,
                       py::object init_coefficients, py::object init_root_ids) -> FitResult {
            if (init_paths.is_none() || init_rewards.is_none()) {
                if (seed.is_none()) return self.fit();
                return self.fit(seed.cast<uint64_t>());
            }
            auto paths = init_paths.cast<std::vector<std::vector<uint8_t>>>();
            auto rewards = init_rewards.cast<std::vector<float>>();
            auto coefficients = init_coefficients.is_none() ? std::vector<std::vector<Scalar>>{}
                : init_coefficients.cast<std::vector<std::vector<Scalar>>>();
            auto roots = init_root_ids.is_none() ? std::vector<std::vector<std::string>>{}
                : init_root_ids.cast<std::vector<std::vector<std::string>>>();
            return self.fit(seed.is_none() ? 0 : seed.cast<uint64_t>(), paths, rewards, coefficients, roots);
        }, py::arg("seed") = py::none(),
           py::arg("init_paths") = py::none(),
           py::arg("init_rewards") = py::none(),
           py::arg("init_coefficients") = py::none(),
           py::arg("init_root_ids") = py::none())
        // prefix from the root and must return a probability vector over op
        // indices (same indexing as the PrimitiveSet: ops first, then x0..xn).
        .def("set_policy_prior", [](Regressor& self, py::function fn) {
            self.set_policy_prior(
                [fn](const std::vector<uint8_t>& prefix) -> std::vector<float> {
                    py::gil_scoped_acquire acquire;
                    py::object out = fn(py::cast(prefix));
                    return out.cast<std::vector<float>>();
                });
        }, py::arg("fn"))
        .def("fit_expression", &Regressor::fit_expression, py::arg("path"))
        .def("legal_actions", &Regressor::legal_actions, py::arg("prefix"))
        .def("expression_for", [](Regressor& self,
                                  const std::vector<uint8_t>& path) {
            return self.expression_for(path);
        }, py::arg("path"));
}
