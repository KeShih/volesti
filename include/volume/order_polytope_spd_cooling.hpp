// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_ORDER_POLYTOPE_SPD_COOLING_HPP
#define VOLUME_ORDER_POLYTOPE_SPD_COOLING_HPP

#include "convex_bodies/order_polytope_spd_rounding.hpp"
#include "random_walks/order_polytope_spd_gaussian_hmc_exact_walk.hpp"
#include "volume/order_polytope_diagonal_cooling.hpp"

template <typename Point, typename RandomNumberGenerator>
GaussianCoolingLogVolume<typename Point::FT>
volume_cooling_gaussians_log(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    OrderPolytopeSPDRounding<Point> const& metric,
    double error = 0.1,
    unsigned int walk_length = 1,
    unsigned long long max_samples =
        order_rounding_cooling_detail::default_max_samples)
{
    if (metric.is_diagonal()) {
        typename OrderPolytope<Point>::VT const shape_diag =
            metric.shape().diagonal();
        OrderPolytopeDiagonalRounding<Point> const diagonal_metric(
            input_polytope, metric.center(), shape_diag);
        return volume_cooling_gaussians_log(
            input_polytope, rng, diagonal_metric, error, walk_length,
            max_samples);
    }

    typedef order_polytope_rounding_detail::SPDBody<Point> Polytope;
    Polytope polytope(input_polytope, metric);
    return order_rounding_cooling_detail::run<
        OrderPolytopeSPDExactHMCWalk>(
            polytope, rng, metric, error, walk_length, max_samples);
}

#endif // VOLUME_ORDER_POLYTOPE_SPD_COOLING_HPP
