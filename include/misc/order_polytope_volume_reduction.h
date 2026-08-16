// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ORDER_POLYTOPE_VOLUME_REDUCTION_H
#define ORDER_POLYTOPE_VOLUME_REDUCTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "misc/poset.h"

struct OrderPolytopeVolumeReductionOptions {
    bool enable_chain_antichain = true;
    bool enable_parallel_decomposition = true;
    bool enable_ordinal_decomposition = true;
    bool enable_small_exact_dp = true;
    bool transitive_reduce_residuals = true;
    unsigned exact_dp_max_n = 20;
};

enum class OrderPolytopeReductionStepKind {
    ExactEmpty,
    ExactSingleton,
    ExactChain,
    ExactAntichain,
    ExactDP,
    ParallelDecomposition,
    OrdinalSumDecomposition,
    ResidualCore
};

struct OrderPolytopeReductionStep {
    OrderPolytopeReductionStepKind kind;
    unsigned input_size;
    std::vector<unsigned> block_sizes;
    double log_volume_contribution;
};

struct ReducedOrderPolytopeVolumeProblem {
    double log_volume_offset = 0.0;
    std::vector<Poset> residual_posets;
    std::vector<OrderPolytopeReductionStep> steps;

    bool is_exact() const { return residual_posets.empty(); }
};

struct MappedReducedOrderPolytopeVolumeProblem {
    ReducedOrderPolytopeVolumeProblem problem;
    // residual_vertices[r][i] is the original input-poset vertex represented
    // by coordinate i of residual_posets[r].  The parallel/ordinal reducers
    // reindex induced subposets, so consumers carrying per-coordinate data
    // must use this map rather than assuming residual indices are original.
    std::vector<std::vector<unsigned int> > residual_vertices;
};

namespace order_polytope_volume_reduction_detail {

typedef std::vector<std::vector<bool> > ReachabilityMatrix;

inline Poset make_poset(unsigned int n, Poset::RV relations) { return Poset(n, relations); }

inline ReachabilityMatrix transitive_closure(Poset const& P)
{
    unsigned int n = P.num_elem();
    ReachabilityMatrix reach(n, std::vector<bool>(n, false));

    for (unsigned int i = 0; i < P.num_relations(); ++i) {
        Poset::RT rel = P.get_relation(i);
        reach[rel.first][rel.second] = true;
    }

    for (unsigned int k = 0; k < n; ++k) {
        for (unsigned int i = 0; i < n; ++i) {
            if (!reach[i][k]) continue;
            for (unsigned int j = 0; j < n; ++j) reach[i][j] = reach[i][j] || reach[k][j];
        }
    }

    return reach;
}

inline bool comparable(ReachabilityMatrix const& reach, unsigned int i, unsigned int j)
{
    return reach[i][j] || reach[j][i];
}

inline Poset induced_poset(ReachabilityMatrix const& reach,
                           std::vector<unsigned int> const& vertices, bool transitive_reduce)
{
    Poset::RV relations;
    unsigned int m = static_cast<unsigned int>(vertices.size());

    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            if (i != j && reach[vertices[i]][vertices[j]]) relations.push_back(Poset::RT(i, j));
        }
    }

    Poset induced = make_poset(m, relations);
    return transitive_reduce ? induced.transitive_reduction() : induced;
}

template <typename EdgePredicate>
std::vector<std::vector<unsigned int> > connected_components(unsigned int n, EdgePredicate edge)
{
    std::vector<std::vector<unsigned int> > components;
    std::vector<bool> seen(n, false);

    for (unsigned int start = 0; start < n; ++start) {
        if (seen[start]) continue;

        std::vector<unsigned int> comp;
        std::queue<unsigned int> q;
        seen[start] = true;
        q.push(start);

        while (!q.empty()) {
            unsigned int u = q.front();
            q.pop();
            comp.push_back(u);

            for (unsigned int v = 0; v < n; ++v) {
                if (!seen[v] && u != v && edge(u, v)) {
                    seen[v] = true;
                    q.push(v);
                }
            }
        }

        std::sort(comp.begin(), comp.end());
        components.push_back(comp);
    }

    return components;
}

// Every pair comparable (expect == true) is a chain; no pair comparable
// (expect == false) is an antichain.
inline bool all_pairs(ReachabilityMatrix const& reach, bool expect)
{
    unsigned int n = static_cast<unsigned int>(reach.size());
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i + 1; j < n; ++j) {
            if (comparable(reach, i, j) != expect) return false;
        }
    }
    return true;
}

inline bool is_chain(ReachabilityMatrix const& reach) { return all_pairs(reach, true); }

inline bool is_antichain(ReachabilityMatrix const& reach) { return all_pairs(reach, false); }

inline unsigned long long exact_linear_extensions_dp(Poset const& P,
                                                     ReachabilityMatrix const& reach)
{
    unsigned int n = P.num_elem();
    std::vector<unsigned long long> pred_mask(n, 0);

    for (unsigned int v = 0; v < n; ++v) {
        for (unsigned int u = 0; u < n; ++u) if (reach[u][v]) pred_mask[v] |= (1ULL << u);
    }

    std::size_t states = std::size_t(1) << n;
    std::vector<unsigned long long> dp(states, 0);
    dp[0] = 1;

    for (std::size_t mask = 0; mask < states; ++mask) {
        unsigned long long count = dp[mask];
        if (count == 0) continue;

        unsigned long long mask64 = static_cast<unsigned long long>(mask);
        for (unsigned int v = 0; v < n; ++v) {
            unsigned long long bit = 1ULL << v;
            if ((mask64 & bit) == 0 && (pred_mask[v] & ~mask64) == 0) dp[mask | bit] += count;
        }
    }

    return dp[states - 1];
}

inline bool ordinal_components(ReachabilityMatrix const& reach,
                               std::vector<std::vector<unsigned int> > const& components,
                               std::vector<std::vector<unsigned int> >& ordered)
{
    unsigned int r = static_cast<unsigned int>(components.size());
    std::vector<std::vector<unsigned int> > adj(r);
    std::vector<unsigned int> indegree(r, 0);

    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = i + 1; j < r; ++j) {
            int dir = 0;

            for (unsigned int a : components[i]) {
                for (unsigned int b : components[j]) {
                    int pair_dir = reach[a][b] ? 1 : (reach[b][a] ? -1 : 0);
                    if (pair_dir == 0) return false;
                    if (dir == 0) dir = pair_dir;
                    else if (dir != pair_dir) return false;
                }
            }

            unsigned int from = dir == 1 ? i : j;
            unsigned int to = dir == 1 ? j : i;
            adj[from].push_back(to);
            ++indegree[to];
        }
    }

    std::queue<unsigned int> q;
    for (unsigned int i = 0; i < r; ++i) if (indegree[i] == 0) q.push(i);

    std::vector<unsigned int> order;
    while (!q.empty()) {
        unsigned int u = q.front();
        q.pop();
        order.push_back(u);

        for (unsigned int v : adj[u]) {
            --indegree[v];
            if (indegree[v] == 0) q.push(v);
        }
    }

    if (order.size() != r) return false;

    ordered.clear();
    for (unsigned int idx : order) ordered.push_back(components[idx]);

    for (unsigned int i = 0; i < r; ++i) {
        for (unsigned int j = i + 1; j < r; ++j) {
            for (unsigned int a : ordered[i]) {
                for (unsigned int b : ordered[j]) if (!reach[a][b]) return false;
            }
        }
    }

    return true;
}

inline std::vector<unsigned> block_sizes(std::vector<std::vector<unsigned int> > const& blocks)
{
    std::vector<unsigned> sizes;
    sizes.reserve(blocks.size());
    for (auto const& block : blocks) sizes.push_back(static_cast<unsigned>(block.size()));
    return sizes;
}

inline void add_step(ReducedOrderPolytopeVolumeProblem& result,
                     OrderPolytopeReductionStepKind kind, unsigned input_size,
                     std::vector<unsigned> const& sizes, double contribution)
{
    result.steps.push_back({kind, input_size, sizes, contribution});
}

template <bool RetainVertexMaps>
inline void reduce_recursive(Poset const& P,
                             std::vector<unsigned int> const* original_vertices,
                             OrderPolytopeVolumeReductionOptions const& options,
                             ReducedOrderPolytopeVolumeProblem& result,
                             std::vector<std::vector<unsigned int> >*
                                 residual_vertices)
{
    unsigned int n = P.num_elem();
    if constexpr (RetainVertexMaps) {
        if (!original_vertices || original_vertices->size() != n)
            throw std::invalid_argument(
                "order-polytope reduction vertex map has wrong dimension");
    }

    if (n == 0) {
        add_step(result, OrderPolytopeReductionStepKind::ExactEmpty, n, {}, 0.0);
        return;
    }
    if (n == 1) {
        add_step(result, OrderPolytopeReductionStepKind::ExactSingleton, n, {}, 0.0);
        return;
    }

    ReachabilityMatrix reach = transitive_closure(P);

    if (options.enable_chain_antichain) {
        if (is_antichain(reach)) {
            add_step(result, OrderPolytopeReductionStepKind::ExactAntichain, n, {}, 0.0);
            return;
        }

        if (is_chain(reach)) {
            double contribution = -std::lgamma(static_cast<double>(n) + 1.0);
            result.log_volume_offset += contribution;
            add_step(result, OrderPolytopeReductionStepKind::ExactChain, n, {}, contribution);
            return;
        }
    }

    if (options.enable_parallel_decomposition) {
        auto components = connected_components(n,
            [&](unsigned int i, unsigned int j) { return comparable(reach, i, j); });

        if (components.size() > 1) {
            add_step(result, OrderPolytopeReductionStepKind::ParallelDecomposition,
                     n, block_sizes(components), 0.0);

            for (auto const& comp : components) {
                Poset sub = induced_poset(reach, comp, options.transitive_reduce_residuals);
                if constexpr (RetainVertexMaps) {
                    std::vector<unsigned int> sub_vertices;
                    sub_vertices.reserve(comp.size());
                    for (unsigned int vertex : comp)
                        sub_vertices.push_back((*original_vertices)[vertex]);
                    reduce_recursive<true>(
                        sub, &sub_vertices, options, result,
                        residual_vertices);
                } else {
                    reduce_recursive<false>(
                        sub, nullptr, options, result, nullptr);
                }
            }
            return;
        }
    }

    if (options.enable_ordinal_decomposition) {
        auto components = connected_components(n,
            [&](unsigned int i, unsigned int j) { return !comparable(reach, i, j); });

        std::vector<std::vector<unsigned int> > ordered;
        if (components.size() > 1 && ordinal_components(reach, components, ordered)) {
            double contribution = -std::lgamma(static_cast<double>(n) + 1.0);
            for (auto const& block : ordered) {
                contribution += std::lgamma(static_cast<double>(block.size()) + 1.0);
            }

            result.log_volume_offset += contribution;
            add_step(result, OrderPolytopeReductionStepKind::OrdinalSumDecomposition,
                     n, block_sizes(ordered), contribution);

            for (auto const& block : ordered) {
                Poset sub = induced_poset(reach, block, options.transitive_reduce_residuals);
                if constexpr (RetainVertexMaps) {
                    std::vector<unsigned int> sub_vertices;
                    sub_vertices.reserve(block.size());
                    for (unsigned int vertex : block)
                        sub_vertices.push_back((*original_vertices)[vertex]);
                    reduce_recursive<true>(
                        sub, &sub_vertices, options, result,
                        residual_vertices);
                } else {
                    reduce_recursive<false>(
                        sub, nullptr, options, result, nullptr);
                }
            }
            return;
        }
    }

    unsigned int safe_exact_dp_max = std::min(options.exact_dp_max_n, 20u);
    if (options.enable_small_exact_dp && n <= safe_exact_dp_max) {
        unsigned long long count = exact_linear_extensions_dp(P, reach);
        double contribution = std::log(static_cast<long double>(count))
                            - std::lgamma(static_cast<double>(n) + 1.0);
        result.log_volume_offset += contribution;
        add_step(result, OrderPolytopeReductionStepKind::ExactDP, n, {}, contribution);
        return;
    }

    std::vector<unsigned int> vertices(n);
    for (unsigned int i = 0; i < n; ++i) vertices[i] = i;
    result.residual_posets.push_back(
        induced_poset(reach, vertices, options.transitive_reduce_residuals));
    if constexpr (RetainVertexMaps) {
        if (!residual_vertices)
            throw std::runtime_error(
                "order-polytope reduction is missing its vertex-map sink");
        residual_vertices->push_back(*original_vertices);
    }
    add_step(result, OrderPolytopeReductionStepKind::ResidualCore, n, {}, 0.0);
}

} // namespace order_polytope_volume_reduction_detail

inline ReducedOrderPolytopeVolumeProblem reduce_order_polytope_volume_problem(
    Poset const& P,
    OrderPolytopeVolumeReductionOptions const& options = OrderPolytopeVolumeReductionOptions())
{
    ReducedOrderPolytopeVolumeProblem result;
    order_polytope_volume_reduction_detail::reduce_recursive<false>(
        P, nullptr, options, result, nullptr);
    return result;
}

// Explicit opt-in variant for consumers carrying per-coordinate data through
// recursive residual reindexing.  The default reducer above retains its prior
// result type and allocation behavior.
inline MappedReducedOrderPolytopeVolumeProblem
reduce_order_polytope_volume_problem_with_vertex_maps(
    Poset const& P,
    OrderPolytopeVolumeReductionOptions const& options =
        OrderPolytopeVolumeReductionOptions())
{
    MappedReducedOrderPolytopeVolumeProblem result;
    std::vector<unsigned int> original_vertices(P.num_elem());
    for (unsigned int i = 0; i < P.num_elem(); ++i) original_vertices[i] = i;
    order_polytope_volume_reduction_detail::reduce_recursive<true>(
        P, &original_vertices, options, result.problem,
        &result.residual_vertices);
    return result;
}

#endif // ORDER_POLYTOPE_VOLUME_REDUCTION_H
