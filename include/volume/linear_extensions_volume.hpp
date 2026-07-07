// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
#define VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "convex_bodies/orderpolytope.h"
#include "misc/order_polytope_volume_reduction.h"
#include "misc/poset.h"
#include "random_walks/random_walks.hpp"
#include "volume/sampling_policies.hpp"
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

#endif // VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
