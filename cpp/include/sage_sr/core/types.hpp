// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// include/sage_sr/core/types.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <random>
#include <vector>
#include <span>

namespace sage_sr {

using Scalar = double;
using Hash   = uint64_t;
using RandomGenerator = std::mt19937_64;

struct Range {
    std::size_t start{0};
    std::size_t size{0};
};

} // namespace sage_sr
