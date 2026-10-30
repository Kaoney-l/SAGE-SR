// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SAGE-SR Authors
// source/core/bridge.cpp
#include "sage_sr/core/bridge.hpp"
#include "sage_sr/eval/timing.hpp"
#include <stdexcept>

namespace sage_sr {

static int build_postfix(
    std::span<uint8_t const> prefix,
    int pos,
    const PrimitiveSet& pset,
    std::vector<Node>& nodes)
{
    if (pos >= static_cast<int>(prefix.size()))
        throw std::runtime_error("Bridge: prefix sequence too short");

    uint8_t op_idx = prefix[pos++];
    if (op_idx >= pset.symbols.size())
        throw std::invalid_argument("Prefix contains an unknown symbol index");
    const Symbol& sym = pset.symbols[op_idx];

    for (int c = 0; c < sym.arity; c++) {
        pos = build_postfix(prefix, pos, pset, nodes);
    }

    Node node(sym.node_type);
    node.arity = sym.arity;

    if (sym.node_type == NodeType::Variable) {
        node.var_index = sym.var_index;
        node.optimize = false;
    } else if (sym.node_type == NodeType::Constant) {
        node.value = 1.0;
        node.optimize = true;
    }

    nodes.push_back(node);
    return pos;
}

Tree Bridge::to_tree(std::span<uint8_t const> prefix) const
{
    Tree tree;
    to_tree(prefix, tree);
    return tree;
}

void Bridge::to_tree(std::span<uint8_t const> prefix, Tree& out) const
{
    ScopedTimer timer(TimingSection::BridgeToTree);
    auto& nodes = out.nodes();
    nodes.clear();
    nodes.reserve(prefix.size());
    if (build_postfix(prefix, 0, *pset_, nodes) != static_cast<int>(prefix.size()))
        throw std::invalid_argument("Prefix contains tokens after expression completion");
    out.update_lengths();
}

} // namespace sage_sr
