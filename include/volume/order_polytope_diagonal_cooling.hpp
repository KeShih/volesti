// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP
#define VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "volume/volume_cooling_gaussians_log.hpp"

namespace order_rounding_cooling_detail {

static constexpr unsigned long long default_max_samples = 100000000ULL;

template <typename WalkPolicy, typename Polytope,
          typename RandomNumberGenerator, typename Metric>
GaussianCoolingLogVolume<typename Polytope::PointType::FT>
run(Polytope& polytope, RandomNumberGenerator& rng, Metric const& metric,
    double error, unsigned int walk_length,
    unsigned long long max_samples)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename WalkPolicy::template Walk<
        Polytope, RandomNumberGenerator> WalkType;

    if (!std::isfinite(error) || error <= 0.0 || walk_length == 0 ||
        max_samples == 0)
        throw std::invalid_argument(
            "rounded OrderPolytope cooling parameters must be positive");
    unsigned int const n = polytope.dimension();
    if (n == 0) return {true, NT(0)};
    if (!volume_cooling_gaussians_log_detail::
            schedule_parameters_fit(n))
        throw std::overflow_error(
            "rounded OrderPolytope dimension exceeds the schedule range");
    gaussian_annealing_parameters<NT> parameters(n);
    std::vector<NT> a_values;
    volume_cooling_gaussians_log_detail::ScheduleSampleBudget sample_budget(
        max_samples);

    auto const quadratic = [&](Point const& sample) {
        return metric.centered_quadratic_statistic(sample);
    };
    auto const prepare_walk = [](WalkType&, NT) { return true; };
    if (!volume_cooling_gaussians_log_detail::bounded_annealing_schedule
            <WalkType>(
                polytope, polytope.facet_metric_distances(),
                parameters.ratio, parameters.C, parameters.frac,
                parameters.N, walk_length, NT(error), a_values, rng,
                sample_budget, quadratic, prepare_walk))
        throw std::runtime_error(
            "rounded OrderPolytope cooling could not construct its schedule");

    unsigned int const W = parameters.W;
    unsigned int const phase_count =
        static_cast<unsigned int>(a_values.size() - 1);
    Point point(n);
    NT log_volume = metric.gaussian_log_normalizer(a_values.front());

    for (unsigned int phase = 0; phase < phase_count; ++phase) {
        NT const current_a = a_values[phase];
        NT const next_a = a_values[phase + 1];
        WalkType walk(polytope, point, current_a, rng);
        NT const current_error = NT(error) / std::sqrt(NT(phase_count));
        NT const window_tolerance = current_error < NT(2)
            ? -std::log(NT(1) - current_error / NT(2))
            : std::numeric_limits<NT>::max();

        auto sample_log_weight = [&]() -> NT {
            walk.apply(polytope, point, current_a, walk_length, rng);
            return (current_a - next_a) * quadratic(point);
        };

        NT log_ratio;
        unsigned long long samples = 0;
        bool const stage_success =
            volume_cooling_gaussians_log_detail::accumulate_stage<NT>(
                W, window_tolerance, sample_budget.remaining,
                sample_log_weight, log_ratio, samples);
        sample_budget.consume(samples);
        if (!stage_success)
            throw std::runtime_error(
                "rounded OrderPolytope cooling ratio estimator failed");
        log_volume += log_ratio;
        if (!std::isfinite(log_volume))
            throw std::runtime_error(
                "rounded OrderPolytope log volume is non-finite");
    }
    return {true, log_volume};
}

} // namespace order_rounding_cooling_detail

template <typename Point, typename RandomNumberGenerator>
GaussianCoolingLogVolume<typename Point::FT>
volume_cooling_gaussians_log(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    double error = 0.1,
    unsigned int walk_length = 1,
    unsigned long long max_samples =
        order_rounding_cooling_detail::default_max_samples)
{
    typedef order_polytope_rounding_detail::DiagonalBody<Point> Polytope;
    Polytope polytope(input_polytope, metric);
    return order_rounding_cooling_detail::run<
        OrderPolytopeDiagonalExactHMCWalk>(
            polytope, rng, metric, error, walk_length, max_samples);
}

#endif // VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP
