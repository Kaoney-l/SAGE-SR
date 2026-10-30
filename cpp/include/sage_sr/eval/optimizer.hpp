// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/eval/optimizer.hpp
#pragma once
#include "sage_sr/core/tree.hpp"
#include "sage_sr/core/dataset.hpp"
#include "sage_sr/core/types.hpp"
#include "sage_sr/eval/interpreter.hpp"

namespace sage_sr {

class CoefficientOptimizer {
public:
    // Optimize constants in tree using Levenberg-Marquardt.
    // Returns a new tree with optimized coefficient values.
    static Tree optimize(const Tree& tree, const Dataset& ds, Range range,
                         int max_iter = 10);
    static Tree optimize(const Tree& tree, const Dataset& ds, Range range,
                         InterpreterWorkspace& workspace, int max_iter);
};

} // namespace sage_sr
