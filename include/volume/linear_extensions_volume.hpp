// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
#define VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "convex_bodies/orderpolytope.h"
#include "misc/order_polytope_volume_reduction.h"
#include "misc/poset.h"
#include "random_walks/random_walks.hpp"
#include "volume/sampling_policies.hpp"
#include "volume/order_polytope_diagonal_cooling.hpp"
#include "volume/volume_cooling_gaussians.hpp"


template <typename NT>
struct linear_extension_estimate {
    NT volume_estimate, log_volume, log_factorial, log_extensions;
    unsigned int n, input_relations, reduced_relations;
    bool reduction_exact;
    unsigned int residual_count;
    std::vector<unsigned int> residual_sizes;
    NT log_volume_offset;
    std::vector<OrderPolytopeReductionStep> reduction_steps;
};


template <typename WalkTypePolicy, typename RandomNumberGenerator, typename Point,
          typename NT = typename Point::FT>
linear_extension_estimate<NT>
estimate_log_linear_extensions(Poset const& poset,
                               RandomNumberGenerator& rng,
                               NT error = NT(0.1),
                               unsigned int walk_length = 1,
                               OrderPolytopeVolumeReductionOptions const& reduction_options =
                                   OrderPolytopeVolumeReductionOptions())
{
    unsigned int n = poset.num_elem();

    linear_extension_estimate<NT> result;
    result.n = n;
    result.input_relations = poset.num_relations();
    result.reduced_relations = poset.transitive_reduction().num_relations();
    result.log_factorial = std::lgamma(NT(n + 1));

    ReducedOrderPolytopeVolumeProblem reduced_problem =
        reduce_order_polytope_volume_problem(poset, reduction_options);
    result.reduction_exact = reduced_problem.is_exact();
    result.residual_count = static_cast<unsigned int>(reduced_problem.residual_posets.size());
    result.log_volume_offset = static_cast<NT>(reduced_problem.log_volume_offset);
    result.reduction_steps = reduced_problem.steps;
    for (auto const& residual : reduced_problem.residual_posets) {
        result.residual_sizes.push_back(residual.num_elem());
    }

    NT log_volume = static_cast<NT>(reduced_problem.log_volume_offset);

    // Independent relative errors compound through the volume product, so
    // split the target across the residuals to keep the total near `error`.
    NT const per_residual_error = result.residual_count > 1
        ? error / std::sqrt(static_cast<NT>(result.residual_count))
        : error;

    for (auto const& residual : reduced_problem.residual_posets) {
        OrderPolytope<Point> OP(residual);
        double vol = volume_cooling_gaussians<WalkTypePolicy>(
            OP, rng, per_residual_error, walk_length);

        if (vol <= 0.0 || !std::isfinite(vol)) {
            NT const nan = std::numeric_limits<NT>::quiet_NaN();
            result.volume_estimate = nan;
            result.log_volume = nan;
            result.log_extensions = nan;
            return result;
        }

        log_volume += std::log(static_cast<NT>(vol));
    }

    result.log_volume = log_volume;
    result.volume_estimate = std::exp(log_volume);
    result.log_extensions = result.log_volume + result.log_factorial;
    return result;
}


template <typename RandomNumberGenerator, typename Point, typename NT = typename Point::FT>
linear_extension_estimate<NT>
estimate_log_linear_extensions_spherical_order_hmc(
    Poset const& poset,
    RandomNumberGenerator& rng,
    NT error = NT(0.1),
    unsigned int walk_length = 1,
    OrderPolytopeVolumeReductionOptions const& reduction_options =
        OrderPolytopeVolumeReductionOptions())
{
    return estimate_log_linear_extensions
        <OrderPolytopeGaussianHamiltonianMonteCarloExactWalk,
         RandomNumberGenerator, Point, NT>(
            poset, rng, error, walk_length, reduction_options);
}


// Explicit opt-in counting path for a user-provided diagonal rounding shape.
// The supplied metric is expressed in the original input-poset coordinates;
// reduction vertex maps are used to restrict it to every residual core.
template <typename RandomNumberGenerator, typename Point>
linear_extension_estimate<typename Point::FT>
estimate_log_linear_extensions_diagonal_order_hmc(
    Poset const& poset,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    typename Point::FT error = typename Point::FT(0.1),
    unsigned int walk_length = 1,
    OrderPolytopeVolumeReductionOptions const& reduction_options =
        OrderPolytopeVolumeReductionOptions(),
    OrderPolytopeDiagonalRoundingDiagnostics<typename Point::FT>* diagnostics = nullptr)
{
    typedef typename Point::FT NT;
    if (metric.dimension() != poset.num_elem())
        throw std::invalid_argument(
            "diagonal rounding metric dimension does not match input poset");
    if (!std::isfinite(error) || error <= NT(0) || walk_length == 0)
        throw std::invalid_argument(
            "diagonal rounding error and walk length must be positive");

    OrderPolytopeDiagonalRoundingDiagnostics<NT>* sink = diagnostics;
    std::chrono::steady_clock::time_point estimator_start;
    if (sink) {
        *sink = OrderPolytopeDiagonalRoundingDiagnostics<NT>();
        estimator_start = std::chrono::steady_clock::now();
    }

    // Bind the raw user center/shape to the actual input poset even if this
    // estimate reduces exactly and never constructs a cooling walk.
    OrderPolytope<Point> input_polytope(poset);
    OrderPolytopeDiagonalRounding<Point> input_metric(
        input_polytope, metric.center(), metric.shape_diag(), true);
    if (sink)
        sink->metric_setup_microseconds = input_metric.setup_microseconds();

    unsigned int const n = poset.num_elem();
    linear_extension_estimate<NT> result;
    result.n = n;
    result.input_relations = poset.num_relations();
    result.reduced_relations = poset.transitive_reduction().num_relations();
    result.log_factorial = std::lgamma(NT(n + 1));

    MappedReducedOrderPolytopeVolumeProblem const mapped_problem =
        reduce_order_polytope_volume_problem_with_vertex_maps(
            poset, reduction_options);
    ReducedOrderPolytopeVolumeProblem const& reduced_problem =
        mapped_problem.problem;
    if (reduced_problem.residual_posets.size() !=
        mapped_problem.residual_vertices.size())
        throw std::runtime_error(
            "order-polytope reduction returned inconsistent residual maps");

    result.reduction_exact = reduced_problem.is_exact();
    result.residual_count =
        static_cast<unsigned int>(reduced_problem.residual_posets.size());
    result.log_volume_offset = static_cast<NT>(reduced_problem.log_volume_offset);
    result.reduction_steps = reduced_problem.steps;
    for (auto const& residual : reduced_problem.residual_posets)
        result.residual_sizes.push_back(residual.num_elem());

    NT log_volume = static_cast<NT>(reduced_problem.log_volume_offset);
    NT const per_residual_error = result.residual_count > 1
        ? error / std::sqrt(static_cast<NT>(result.residual_count))
        : error;
    if (sink)
        sink->residual_cooling.reserve(result.residual_count);

    for (unsigned int residual_index = 0;
         residual_index < result.residual_count; ++residual_index) {
        Poset const& residual = reduced_problem.residual_posets[residual_index];
        OrderPolytopeDiagonalRounding<Point> residual_metric = input_metric.restrict_to(
            residual, mapped_problem.residual_vertices[residual_index]);
        if (sink)
            sink->metric_setup_microseconds += residual_metric.setup_microseconds();

        OrderPolytope<Point> polytope(residual);
        OrderPolytopeDiagonalCoolingDiagnostics<NT>* cooling_diagnostics = nullptr;
        if (sink) {
            sink->residual_cooling.emplace_back();
            cooling_diagnostics = &sink->residual_cooling.back();
        }
        auto const volume_result =
            volume_cooling_gaussians_order_polytope_diagonal_result(
                polytope, rng, residual_metric,
                static_cast<double>(per_residual_error), walk_length,
                cooling_diagnostics);
        if (sink)
            sink->metric_setup_microseconds +=
                cooling_diagnostics->metric_setup_microseconds;
        if (!std::isfinite(volume_result.log_volume))
            throw std::runtime_error(
                "diagonal rounding cooling returned non-finite log volume");
        log_volume += volume_result.log_volume;
    }

    result.log_volume = log_volume;
    result.volume_estimate = std::exp(log_volume);
    result.log_extensions = result.log_volume + result.log_factorial;
    if (!std::isfinite(result.log_extensions))
        throw std::runtime_error(
            "diagonal rounding estimator returned non-finite log count");

    if (sink) {
        auto const estimator_end = std::chrono::steady_clock::now();
        sink->total_estimator_microseconds =
            std::chrono::duration<double, std::micro>(
                estimator_end - estimator_start).count();
    }
    return result;
}

#endif // VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
