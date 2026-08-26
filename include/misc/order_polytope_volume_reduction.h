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
#include <queue>
#include <vector>

#include "misc/poset.h"

struct OrderPolytopeVolumeReductionOptions {
    // Set to zero to leave all non-structural components for sampling.
    // Values above 20 are capped because 21! does not fit in uint64_t.
    unsigned exact_dp_max_n = 20;
};

struct ReducedOrderPolytopeVolumeProblem {
    double log_volume_offset = 0.0;
    std::vector<Poset> residual_posets;
    // residual_vertices[r][i] maps residual coordinate i to an input vertex.
    std::vector<std::vector<unsigned int> > residual_vertices;

    bool is_exact() const { return residual_posets.empty(); }
};

namespace order_volume_reduction_detail {

typedef std::vector<std::vector<bool> > ReachabilityMatrix;

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

inline ReachabilityMatrix induced_reachability(
    ReachabilityMatrix const& reach,
    std::vector<unsigned int> const& vertices)
{
    unsigned int m = static_cast<unsigned int>(vertices.size());
    ReachabilityMatrix induced(m, std::vector<bool>(m, false));

    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            induced[i][j] = reach[vertices[i]][vertices[j]];
        }
    }
    return induced;
}

// Construct the Hasse diagram directly from an already closed relation.
inline Poset reduced_poset(ReachabilityMatrix const& reach)
{
    unsigned int const n = static_cast<unsigned int>(reach.size());
    Poset::RV relations;

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (!reach[i][j]) continue;
            bool is_cover = true;
            for (unsigned int k = 0; k < n; ++k) {
                if (reach[i][k] && reach[k][j]) {
                    is_cover = false;
                    break;
                }
            }
            if (is_cover) relations.push_back(Poset::RT(i, j));
        }
    }

    return Poset(n, relations);
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
                if (!seen[v] && edge(u, v)) {
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

inline unsigned long long exact_linear_extensions_dp(
    ReachabilityMatrix const& reach)
{
    unsigned int n = static_cast<unsigned int>(reach.size());
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

inline std::vector<std::vector<unsigned int> > ordinal_components(
    ReachabilityMatrix const& reach,
    std::vector<std::vector<unsigned int> > const& components)
{
    std::vector<std::vector<unsigned int> > ordered = components;
    std::sort(ordered.begin(), ordered.end(),
              [&](std::vector<unsigned int> const& lhs,
                  std::vector<unsigned int> const& rhs) {
                  return reach[lhs.front()][rhs.front()];
              });
    return ordered;
}

inline void reduce_recursive(ReachabilityMatrix const& reach,
                             std::vector<unsigned int> const& original_vertices,
                             OrderPolytopeVolumeReductionOptions const& options,
                             ReducedOrderPolytopeVolumeProblem& problem)
{
    unsigned int n = static_cast<unsigned int>(reach.size());

    if (is_antichain(reach)) return;

    if (is_chain(reach)) {
        problem.log_volume_offset -=
            std::lgamma(static_cast<double>(n) + 1.0);
        return;
    }

    auto components = connected_components(n,
        [&](unsigned int i, unsigned int j) { return comparable(reach, i, j); });

    if (components.size() > 1) {
        for (auto const& comp : components) {
            std::vector<unsigned int> sub_vertices;
            sub_vertices.reserve(comp.size());
            for (unsigned int vertex : comp)
                sub_vertices.push_back(original_vertices[vertex]);
            reduce_recursive(
                induced_reachability(reach, comp), sub_vertices,
                options, problem);
        }
        return;
    }

    components = connected_components(n,
        [&](unsigned int i, unsigned int j) { return !comparable(reach, i, j); });

    if (components.size() > 1) {
        std::vector<std::vector<unsigned int> > const ordered =
            ordinal_components(reach, components);
        double contribution = -std::lgamma(static_cast<double>(n) + 1.0);
        for (auto const& block : ordered) {
            contribution += std::lgamma(static_cast<double>(block.size()) + 1.0);
        }

        problem.log_volume_offset += contribution;

        for (auto const& block : ordered) {
            std::vector<unsigned int> sub_vertices;
            sub_vertices.reserve(block.size());
            for (unsigned int vertex : block)
                sub_vertices.push_back(original_vertices[vertex]);
            reduce_recursive(
                induced_reachability(reach, block), sub_vertices,
                options, problem);
        }
        return;
    }

    unsigned int safe_exact_dp_max = std::min(options.exact_dp_max_n, 20u);
    if (n <= safe_exact_dp_max) {
        unsigned long long count = exact_linear_extensions_dp(reach);
        double contribution = std::log(static_cast<long double>(count))
                            - std::lgamma(static_cast<double>(n) + 1.0);
        problem.log_volume_offset += contribution;
        return;
    }

    problem.residual_posets.push_back(reduced_poset(reach));
    problem.residual_vertices.push_back(original_vertices);
}

} // namespace order_volume_reduction_detail

inline ReducedOrderPolytopeVolumeProblem reduce_order_polytope_volume_problem(
    Poset const& P,
    OrderPolytopeVolumeReductionOptions const& options = OrderPolytopeVolumeReductionOptions())
{
    ReducedOrderPolytopeVolumeProblem result;
    std::vector<unsigned int> original_vertices(P.num_elem());
    for (unsigned int i = 0; i < P.num_elem(); ++i)
        original_vertices[i] = i;
    order_volume_reduction_detail::reduce_recursive(
        order_volume_reduction_detail::transitive_closure(P),
        original_vertices, options, result);
    return result;
}

#endif // ORDER_POLYTOPE_VOLUME_REDUCTION_H
