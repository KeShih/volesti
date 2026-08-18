// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP
#define VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "sampling/random_point_generators.hpp"
#include "volume/math_helpers.hpp"
#include "volume/sampling_policies.hpp"
#include "volume/volume_cooling_gaussians.hpp"

template <typename NT>
struct OrderPolytopeDiagonalCoolingResult {
    NT volume_estimate;
    NT log_volume;
};

namespace order_polytope_diagonal_cooling_detail {

template <typename NT>
inline void get_first_gaussian_from_metric_distances(
    std::vector<NT> const& distances,
    NT frac,
    NT error,
    std::vector<NT>& a_values)
{
    if (distances.empty() || !std::isfinite(frac) || frac <= NT(0) ||
        !std::isfinite(error) || error <= NT(0))
        throw std::invalid_argument(
            "diagonal cooling initial-Gaussian inputs are invalid");
    for (NT distance : distances)
        if (!std::isfinite(distance) || distance <= NT(0))
            throw std::invalid_argument(
                "diagonal cooling requires finite positive facet distances");

    NT const tol = std::is_same<float, NT>::value ? NT(0.001) : NT(0.0000001);
    NT lower = NT(0), upper = NT(1);
    unsigned int const max_iterations = 10000;
    static NT const pi = std::acos(NT(-1));
    static NT const log_two = std::log(NT(2));
    static NT const half_log_pi = NT(0.5) * std::log(pi);
    NT const target_tail = frac * error;
    if (!std::isfinite(target_tail) || target_tail <= NT(0))
        throw std::invalid_argument(
            "diagonal cooling tail tolerance has unusable scale");

    auto tail_sum = [&](NT a) {
        if (!std::isfinite(a) || a <= NT(0))
            throw std::runtime_error(
                "diagonal cooling initial Gaussian parameter is invalid");
        NT sum = NT(0);
        for (NT distance : distances) {
            NT const exponent = -a * distance * distance
                              - log_two - std::log(distance)
                              - half_log_pi - NT(0.5) * std::log(a);
            NT const term = std::exp(exponent);
            if (std::isnan(term))
                throw std::runtime_error(
                    "diagonal cooling initial Gaussian tail is NaN");
            sum += term;
        }
        return sum;
    };

    unsigned int iteration = 0;
    for (; iteration < max_iterations; ++iteration) {
        if (tail_sum(upper) <= target_tail) break;
        upper *= NT(10);
        if (!std::isfinite(upper))
            throw std::runtime_error(
                "diagonal cooling could not bracket the initial Gaussian");
    }
    if (iteration == max_iterations)
        throw std::runtime_error(
            "diagonal cooling could not bracket the initial Gaussian");

    iteration = 0;
    for (; iteration < max_iterations; ++iteration) {
        NT const gap = upper - lower;
        if (!(gap > tol * std::abs(upper))) break;
        NT const middle = lower + (upper - lower) / NT(2);
        if (middle == lower || middle == upper) break;
        if (tail_sum(middle) < target_tail) upper = middle;
        else lower = middle;
    }
    if (iteration == max_iterations)
        throw std::runtime_error(
            "diagonal cooling initial-Gaussian bisection did not converge");
    // `upper` is always the tail-bound-satisfying side of the bracket.
    // Returning the midpoint can violate the requested tail bound at very
    // small scales.
    NT const a0 = upper;
    if (!std::isfinite(a0) || a0 <= NT(0))
        throw std::runtime_error(
            "diagonal cooling produced an invalid initial Gaussian");
    if (tail_sum(a0) > target_tail)
        throw std::runtime_error(
            "diagonal cooling initial Gaussian does not satisfy its tail bound");
    a_values.push_back(a0);
}

template <typename NT>
inline unsigned int checked_schedule_sample_count(NT frac, NT error)
{
    NT const denominator = (NT(1) - frac) * error;
    if (!std::isfinite(denominator) || denominator <= NT(0))
        throw std::invalid_argument(
            "diagonal cooling schedule tolerance has unusable scale");
    NT const raw_steps = NT(150) / denominator;
    long double const raw_steps_wide = static_cast<long double>(raw_steps);
    long double const max_steps_wide =
        static_cast<long double>(std::numeric_limits<unsigned int>::max())
        - 1.0L;
    if (!std::isfinite(raw_steps) || raw_steps < NT(0) ||
        raw_steps_wide > max_steps_wide)
        throw std::invalid_argument(
            "diagonal cooling schedule sample count exceeds supported range");
    return static_cast<unsigned int>(raw_steps) + 1u;
}

template <typename Point, typename Metric, typename NT>
inline NT ratio_weight(Point const& centered_point,
                       NT next_a,
                       NT current_a,
                       Metric const& metric)
{
    NT const q = metric.centered_quadratic_statistic(centered_point);
    NT const exponent = (current_a - next_a) * q;
    if (!std::isfinite(exponent))
        throw std::runtime_error(
            "diagonal cooling Gaussian-ratio exponent is non-finite");
    NT const weight = std::exp(exponent);
    if (!std::isfinite(weight) || weight <= NT(0))
        throw std::runtime_error(
            "diagonal cooling Gaussian-ratio weight is invalid");
    return weight;
}

template <typename RandomPointGenerator, typename Polytope, typename Point,
          typename Metric, typename NT, typename RandomNumberGenerator>
NT get_next_gaussian(Polytope& polytope,
                     Point& point,
                     NT a,
                     unsigned int N,
                     NT ratio,
                     NT C,
                     unsigned int walk_length,
                     Metric const& metric,
                     RandomNumberGenerator& rng)
{
    if (!std::isfinite(a) || a <= NT(0) || !std::isfinite(ratio) ||
        ratio < NT(0) || ratio >= NT(1) || N == 0)
        throw std::invalid_argument(
            "diagonal cooling schedule inputs are invalid");

    NT last_ratio = NT(0.1);
    NT k = NT(1);
    NT const tol = NT(0.00001);
    std::vector<NT> weights(N, NT(0));
    std::list<Point> samples;
    PushBackWalkPolicy push_back_policy;
    RandomPointGenerator::apply(polytope, point, a, N, walk_length,
                                samples, push_back_policy, rng);

    unsigned int const max_iterations = 10000;
    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
        NT const next_a = a * std::pow(ratio, k);
        if (!std::isfinite(next_a) || next_a < NT(0))
            throw std::runtime_error(
                "diagonal cooling produced an invalid next Gaussian");

        auto weight = weights.begin();
        for (auto const& sample : samples) {
            *weight = ratio_weight(sample, next_a, a, metric);
            ++weight;
        }
        std::pair<NT, NT> const mean_variance = get_mean_variance(weights);
        if (!std::isfinite(mean_variance.first) || mean_variance.first <= NT(0) ||
            !std::isfinite(mean_variance.second) || mean_variance.second < NT(0))
            throw std::runtime_error(
                "diagonal cooling schedule statistics are invalid");

        NT const coefficient_of_variation_squared =
            (mean_variance.second / mean_variance.first) /
            mean_variance.first;
        if (!std::isfinite(coefficient_of_variation_squared))
            throw std::runtime_error(
                "diagonal cooling schedule variation is non-finite");
        if (coefficient_of_variation_squared >= C ||
            mean_variance.first / last_ratio < NT(1) + tol) {
            if (k != NT(1)) k /= NT(2);
            NT const result = a * std::pow(ratio, k);
            if (!std::isfinite(result) || result < NT(0))
                throw std::runtime_error(
                    "diagonal cooling schedule result is invalid");
            return result;
        }
        k *= NT(2);
        if (!std::isfinite(k))
            throw std::runtime_error(
                "diagonal cooling schedule exponent overflowed");
        last_ratio = mean_variance.first;
    }
    throw std::runtime_error(
        "diagonal cooling schedule did not converge");
}

template <typename WalkType, typename RandomPointGenerator,
          typename Polytope, typename Metric, typename NT,
          typename RandomNumberGenerator>
void compute_annealing_schedule(Polytope& polytope,
                                NT ratio,
                                NT C,
                                NT frac,
                                unsigned int N,
                                unsigned int walk_length,
                                NT error,
                                Metric const& metric,
                                std::vector<NT>& a_values,
                                RandomNumberGenerator& rng)
{
    typedef typename Polytope::PointType Point;
    get_first_gaussian_from_metric_distances(
        polytope.rounding_facet_metric_distances(), frac, error, a_values);

    NT const a_stop = NT(0);
    NT const tol = NT(0.001);
    unsigned int const total_steps =
        checked_schedule_sample_count(frac, error);
    Point point(polytope.dimension());

    unsigned int const max_phases = 10000;
    for (unsigned int phase = 0; phase < max_phases; ++phase) {
        NT const current_a = a_values.back();
        NT const next_a = get_next_gaussian<RandomPointGenerator>(
            polytope, point, current_a, N, ratio, C, walk_length, metric, rng);

        NT ratio_sum = NT(0);
        WalkType walk(polytope, point, current_a, rng);
        for (unsigned int j = 0; j < total_steps; ++j) {
            walk.apply(polytope, point, current_a, walk_length, rng);
            ratio_sum += ratio_weight(point, next_a, current_a, metric);
            if (!std::isfinite(ratio_sum))
                throw std::runtime_error(
                    "diagonal cooling schedule ratio sum is non-finite");
        }
        NT const mean = ratio_sum / NT(total_steps);
        if (next_a > NT(0) && mean > NT(1) + tol) {
            a_values.push_back(next_a);
        } else if (next_a <= NT(0)) {
            a_values.push_back(a_stop);
            return;
        } else {
            // The legacy schedule replaces the current phase by zero.  When
            // this happens at phase zero there is no preceding Gaussian to
            // normalize from; retain a0 and estimate the direct a0 -> 0 ratio.
            if (a_values.size() == 1) a_values.push_back(a_stop);
            else a_values.back() = a_stop;
            return;
        }
    }
    throw std::runtime_error(
        "diagonal cooling exceeded the maximum number of phases");
}

} // namespace order_polytope_diagonal_cooling_detail

template <bool EnableDiagnostics, typename WalkPolicy,
          typename Point, typename RandomNumberGenerator>
OrderPolytopeDiagonalCoolingResult<typename Point::FT>
volume_cooling_gaussians_order_polytope_diagonal_implementation(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    double error = 0.1,
    unsigned int walk_length = 1,
    OrderPolytopeDiagonalCoolingDiagnostics<typename Point::FT>* diagnostics = nullptr)
{
    typedef typename Point::FT NT;
    typedef DiagonalRoundingOrderPolytope<Point, EnableDiagnostics> Polytope;
    typedef typename WalkPolicy::template Walk<
        Polytope, RandomNumberGenerator> WalkType;
    typedef GaussianRandomPointGenerator<WalkType> RandomPointGenerator;

    if (!std::isfinite(error) || error <= 0.0 || walk_length == 0)
        throw std::invalid_argument(
            "diagonal cooling error and walk length must be positive");
    if (input_polytope.dimension() != metric.dimension())
        throw std::invalid_argument(
            "diagonal cooling polytope/metric dimension mismatch");

    OrderPolytopeDiagonalCoolingDiagnostics<NT>* sink = diagnostics;
    if constexpr (EnableDiagnostics) {
        if (!sink)
            throw std::invalid_argument(
                "diagonal cooling diagnostics implementation requires a sink");
    } else {
        sink = nullptr;
    }
    std::chrono::steady_clock::time_point cooling_start;
    if (sink) {
        *sink = OrderPolytopeDiagonalCoolingDiagnostics<NT>();
    }

    Polytope polytope(input_polytope, metric, sink);
    if (sink)
        cooling_start = std::chrono::steady_clock::now();
    unsigned int const n = polytope.dimension();
    if (n == 0) {
        if (sink) {
            auto const cooling_end = std::chrono::steady_clock::now();
            sink->cooling_microseconds =
                std::chrono::duration<double, std::micro>(
                    cooling_end - cooling_start).count();
        }
        return {NT(1), NT(0)};
    }
    gaussian_annealing_parameters<NT> parameters(n);
    std::vector<NT> a_values;

    if (sink)
        sink->active_stage = OrderPolytopeDiagonalCoolingStage::Schedule;
    order_polytope_diagonal_cooling_detail::compute_annealing_schedule
        <WalkType, RandomPointGenerator>(
            polytope, parameters.ratio, parameters.C, parameters.frac,
            parameters.N, walk_length, NT(error), metric, a_values, rng);

    if (a_values.size() < 2 || !std::isfinite(a_values.front()) ||
        a_values.front() <= NT(0))
        throw std::runtime_error(
            "diagonal cooling produced an invalid annealing schedule");
    if (sink) {
        sink->initial_a0 = a_values.front();
        sink->cooling_phase_count =
            static_cast<unsigned int>(a_values.size() - 1);
        sink->active_stage = OrderPolytopeDiagonalCoolingStage::Ratio;
    }

    unsigned int const W = parameters.W;
    unsigned int const phase_count =
        static_cast<unsigned int>(a_values.size() - 1);
    std::vector<NT> last_W_template(W, NT(0));
    Point point(n);
    NT log_volume = metric.gaussian_log_normalizer(a_values.front());

    for (unsigned int phase = 0; phase < phase_count; ++phase) {
        NT const current_a = a_values[phase];
        NT const next_a = a_values[phase + 1];
        WalkType walk(polytope, point, current_a, rng);

        bool done = false;
        NT sum = NT(0), iterations = NT(0);
        NT const current_error = NT(error) / std::sqrt(NT(phase_count));
        NT min_value = std::numeric_limits<NT>::min();
        NT max_value = std::numeric_limits<NT>::max();
        unsigned int min_index = W - 1, max_index = W - 1, index = 0;
        std::vector<NT> last_W = last_W_template;

        unsigned long long const max_iterations = 100000000ULL;
        for (unsigned long long step = 0; !done; ++step) {
            if (step >= max_iterations)
                throw std::runtime_error(
                    "diagonal cooling ratio estimator did not converge");
            walk.apply(polytope, point, current_a, walk_length, rng);
            iterations += NT(1);
            sum += order_polytope_diagonal_cooling_detail::ratio_weight(
                point, next_a, current_a, metric);
            if (!std::isfinite(sum))
                throw std::runtime_error(
                    "diagonal cooling ratio sum is non-finite");
            NT const value = sum / iterations;
            last_W[index] = value;

            if (value <= min_value) {
                min_value = value;
                min_index = index;
            } else if (min_index == index) {
                auto const it = std::min_element(last_W.begin(), last_W.end());
                min_value = *it;
                min_index = static_cast<unsigned int>(
                    std::distance(last_W.begin(), it));
            }
            if (value >= max_value) {
                max_value = value;
                max_index = index;
            } else if (max_index == index) {
                auto const it = std::max_element(last_W.begin(), last_W.end());
                max_value = *it;
                max_index = static_cast<unsigned int>(
                    std::distance(last_W.begin(), it));
            }
            if (max_value > NT(0) && std::isfinite(max_value) &&
                (max_value - min_value) / max_value <= current_error / NT(2))
                done = true;

            index = index % W + 1;
            if (index == W) index = 0;
        }

        NT const ratio_estimate = sum / iterations;
        if (!std::isfinite(ratio_estimate) || ratio_estimate <= NT(0))
            throw std::runtime_error(
                "diagonal cooling produced an invalid ratio estimate");
        log_volume += std::log(ratio_estimate);
        if (!std::isfinite(log_volume))
            throw std::runtime_error(
                "diagonal cooling log volume is non-finite");
    }

    if (sink) {
        auto const cooling_end = std::chrono::steady_clock::now();
        sink->cooling_microseconds =
            std::chrono::duration<double, std::micro>(
                cooling_end - cooling_start).count();
    }

    NT const volume = std::exp(log_volume);
    if (!std::isfinite(volume))
        throw std::runtime_error(
            "diagonal cooling volume estimate is non-finite");
    return {volume, log_volume};
}

template <typename Point, typename RandomNumberGenerator>
OrderPolytopeDiagonalCoolingResult<typename Point::FT>
volume_cooling_gaussians_order_polytope_diagonal_result(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    double error = 0.1,
    unsigned int walk_length = 1,
    OrderPolytopeDiagonalCoolingDiagnostics<typename Point::FT>* diagnostics = nullptr)
{
    if (diagnostics)
        return volume_cooling_gaussians_order_polytope_diagonal_implementation<
            true, OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk>(
            input_polytope, rng, metric, error, walk_length, diagnostics);
    return volume_cooling_gaussians_order_polytope_diagonal_implementation<
        false, OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk>(
        input_polytope, rng, metric, error, walk_length, nullptr);
}

template <typename Point, typename RandomNumberGenerator>
double volume_cooling_gaussians_order_polytope_diagonal(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    double error = 0.1,
    unsigned int walk_length = 1,
    OrderPolytopeDiagonalCoolingDiagnostics<typename Point::FT>* diagnostics = nullptr)
{
    return static_cast<double>(
        volume_cooling_gaussians_order_polytope_diagonal_result(
            input_polytope, rng, metric, error, walk_length, diagnostics)
            .volume_estimate);
}

#endif // VOLUME_ORDER_POLYTOPE_DIAGONAL_COOLING_HPP
