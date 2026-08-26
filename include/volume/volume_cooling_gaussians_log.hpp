// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP
#define VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "volume/sampling_policies.hpp"
#include "volume/volume_cooling_gaussians.hpp"

// Log-domain stochastic Gaussian-cooling volume estimation with bounded
// scheduling and a finite sampling budget:
//
//     log vol = (n/2) log(pi / a_0) + sum_i log rho_i,
//
//     rho_i = Z_{i+1} / Z_i = E_{x ~ pi_i}[ g_{i+1}(x) / g_i(x) ],
//
//     log rho_i = logsumexp(log w) - log(N),   log w = (a_i - a_{i+1}) |x|^2
//
// This is the log of the arithmetic mean, not mean(log w).  For 0 <= eps < 2,
// the sliding-window criterion
// (max - min)/max <= eps/2 has the log-domain equivalent
// log_max - log_min <= -log(1 - eps/2); eps >= 2 is trivially satisfied.
// Success means completion within numerical and resource guards, not certified
// convergence or accuracy.

template <typename NT>
struct GaussianCoolingLogVolume
{
    bool success;
    NT log_volume;
};

namespace volume_cooling_gaussians_log_detail {

static constexpr unsigned long long default_max_samples = 10000000ULL;
static constexpr unsigned int default_max_steps = 10000U;

// A failed consume leaves the budget unchanged.
struct ScheduleSampleBudget
{
    explicit ScheduleSampleBudget(unsigned long long limit)
        : remaining(limit) {}

    bool consume(unsigned long long amount)
    {
        if (amount > remaining) return false;
        remaining -= amount;
        return true;
    }

    unsigned long long remaining;
};

template <typename NT>
inline NT log_add_exp(NT lhs, NT rhs)
{
    if (lhs == -std::numeric_limits<NT>::infinity()) return rhs;
    NT const shift = std::max(lhs, rhs);
    return shift + std::log(
        std::exp(lhs - shift) + std::exp(rhs - shift));
}

template <typename Walk, typename NT>
void set_walk_delta(Walk&, NT) {}

template <typename Polytope, typename RandomNumberGenerator, typename NT>
void set_walk_delta(
    GaussianBallWalk::Walk<Polytope, RandomNumberGenerator>& walk, NT delta)
{
    walk.update_delta(delta);
}

template <typename NT>
bool schedule_sample_count(NT frac, NT error, unsigned int& count)
{
    long double const denominator =
        (1.0L - static_cast<long double>(frac)) *
        static_cast<long double>(error);
    if (!std::isfinite(denominator) || denominator <= 0.0L) return false;
    long double const raw = 150.0L / denominator + 1.0L;
    if (!std::isfinite(raw) ||
        raw > static_cast<long double>(
                  std::numeric_limits<unsigned int>::max()))
        return false;
    count = static_cast<unsigned int>(raw);
    return true;
}

template <typename NT>
bool gaussian_tail_sum(std::vector<NT> const& dists,
                       NT a,
                       NT& sum)
{
    sum = NT(0);
    if (dists.empty() || !std::isfinite(a) || !(a > NT(0)))
        return false;
    auto const root = std::sqrt(M_PI * a);
    if (!std::isfinite(root) || !(root > 0)) return false;

    for (auto const& dist : dists) {
        if (!std::isfinite(dist) || !(dist > NT(0))) return false;
        // The double exponent preserves promotion for NT=float.
        auto const squared_distance = std::pow(dist, 2.0);
        auto const exponent = -a * squared_distance;
        auto const denominator = 2.0 * dist * root;
        if (!std::isfinite(squared_distance) ||
            !std::isfinite(exponent) || !std::isfinite(denominator) ||
            !(denominator > 0))
            return false;
        auto const term = std::exp(exponent) / denominator;
        auto const next_sum = sum + term;
        if (!std::isfinite(term) || !std::isfinite(next_sum))
            return false;
        sum = static_cast<NT>(next_sum);
        if (!std::isfinite(sum)) return false;
    }
    return true;
}

inline bool schedule_parameters_fit(unsigned int d)
{
    if (d == 0) return false;
    typedef unsigned long long Wide;
    Wide const dw = static_cast<Wide>(d);
    if (dw > std::numeric_limits<Wide>::max() / dw) return false;
    Wide const square = dw * dw;
    Wide const uint_max = static_cast<Wide>(
        std::numeric_limits<unsigned int>::max());
    if (square > (uint_max - Wide(800)) / Wide(6))
        return false;
    return true;
}

// Bounded upper-bracketing and bisection for the initial Gaussian.
template <typename NT>
bool bounded_first_gaussian(std::vector<NT> const& dists,
                            NT frac,
                            NT error,
                            unsigned int max_bound_steps,
                            unsigned int max_bisection_steps,
                            NT& first_a)
{
    if (!std::isfinite(frac) || !std::isfinite(error) ||
        !(frac > NT(0)) || !(error > NT(0)) ||
        max_bound_steps == 0 || max_bisection_steps == 0)
        return false;

    NT const target = frac * error;
    NT const tol = std::is_same<float, NT>::value ? NT(0.001) : NT(0.0000001);
    if (!std::isfinite(target) || !(target > NT(0))) return false;

    NT lower = NT(0);
    NT upper = NT(1);
    bool bracketed = false;
    for (unsigned int i = 0; i < max_bound_steps; ++i) {
        NT sum;
        if (!gaussian_tail_sum(dists, upper, sum)) return false;
        if (sum > target) {
            NT const next_upper = upper * NT(10);
            if (!std::isfinite(next_upper) || !(next_upper > upper))
                return false;
            upper = next_upper;
        } else {
            bracketed = true;
            break;
        }
    }
    if (!bracketed) return false;

    unsigned int step = 0;
    while (upper - lower > tol * std::abs(upper) &&
           step < max_bisection_steps) {
        NT const mid = lower + (upper - lower) / NT(2);
        if (!std::isfinite(mid)) return false;
        if (mid == lower || mid == upper) break;

        NT sum;
        if (!gaussian_tail_sum(dists, mid, sum)) return false;
        if (sum < target)
            upper = mid;
        else
            lower = mid;
        ++step;
    }
    if (!std::isfinite(upper - lower) ||
        upper - lower > tol * std::abs(upper))
        return false;

    first_a = lower + (upper - lower) / NT(2);
    return std::isfinite(first_a) && first_a > NT(0);
}

template <typename NT>
bool log_weight_moments(std::vector<NT> const& log_weights,
                        NT& log_mean,
                        NT& cv2)
{
    if (log_weights.empty()) return false;
    NT const shift = *std::max_element(log_weights.begin(), log_weights.end());
    if (!std::isfinite(shift)) return false;

    NT sum = NT(0), sum2 = NT(0);
    for (NT const log_weight : log_weights) {
        if (!std::isfinite(log_weight)) return false;
        NT const weight = std::exp(log_weight - shift);
        sum += weight;
        sum2 += weight * weight;
    }

    NT const count = static_cast<NT>(log_weights.size());
    log_mean = shift + std::log(sum / count);
    cv2 = std::max(NT(0), count * sum2 / (sum * sum) - NT(1));
    return std::isfinite(log_mean) && std::isfinite(cv2);
}

template <typename NT, typename MomentEvaluator>
bool bounded_next_gaussian_parameter(NT a,
                                     NT ratio,
                                     NT C,
                                     unsigned int max_search_steps,
                                     MomentEvaluator&& evaluate_moments,
                                     NT& next_a)
{
    if (!std::isfinite(a) || !(a > NT(0)) ||
        !std::isfinite(ratio) || ratio < NT(0) || !(ratio < NT(1)) ||
        !std::isfinite(C) || !(C > NT(0)) || max_search_steps == 0)
        return false;

    NT last_log_mean = std::log(NT(0.1));
    NT k = NT(1);
    NT previous_candidate = std::numeric_limits<NT>::quiet_NaN();
    NT const tol = NT(0.00001);

    for (unsigned int step = 0; step < max_search_steps; ++step) {
        NT const candidate = a * std::pow(ratio, k);
        if (!std::isfinite(candidate) || !(candidate < a))
            return false;
        if (step > 0 && candidate == previous_candidate) {
            // Repeated represented candidates have identical cached moments;
            // return the candidate, which is already known to be below a.
            next_a = candidate;
            return true;
        }
        previous_candidate = candidate;

        NT log_mean, cv2;
        if (!evaluate_moments(candidate, log_mean, cv2))
            return false;

        if (cv2 >= C || log_mean - last_log_mean < std::log1p(tol)) {
            if (k != NT(1)) k /= NT(2);
            next_a = a * std::pow(ratio, k);
            return std::isfinite(next_a) && next_a < a;
        }

        NT const next_k = NT(2) * k;
        if (!std::isfinite(next_k) || !(next_k > k)) return false;
        k = next_k;
        last_log_mean = log_mean;
    }
    return false;
}

template
<
    typename WalkType,
    typename Polytope,
    typename NT,
    typename RandomNumberGenerator,
    typename QuadraticStatistic,
    typename PrepareWalk
>
bool bounded_next_gaussian_with_statistic(
    Polytope& P, typename Polytope::PointType& p,
    NT a, unsigned int N, NT ratio, NT C,
    unsigned int walk_length, RandomNumberGenerator& rng,
    unsigned int max_search_steps, ScheduleSampleBudget& budget,
    NT& next_a, QuadraticStatistic const& quadratic_statistic,
    PrepareWalk const& prepare_walk)
{
    if (N == 0 || walk_length == 0 || !budget.consume(N)) return false;

    WalkType walk(P, p, a, rng);
    if (!prepare_walk(walk, a)) return false;
    std::vector<NT> quadratics;
    quadratics.reserve(N);
    for (unsigned int sample = 0; sample < N; ++sample) {
        walk.apply(P, p, a, walk_length, rng);
        if (!p.getCoefficients().allFinite()) return false;
        NT const quadratic = quadratic_statistic(p);
        if (!std::isfinite(quadratic) || quadratic < NT(0)) return false;
        quadratics.push_back(quadratic);
    }

    std::vector<NT> log_weights(N);
    auto evaluate_moments = [&](NT candidate, NT& log_mean, NT& cv2) {
        auto log_weight = log_weights.begin();
        for (NT const quadratic : quadratics) {
            *log_weight = (a - candidate) * quadratic;
            ++log_weight;
        }
        return log_weight_moments(log_weights, log_mean, cv2);
    };

    return bounded_next_gaussian_parameter<NT>(
        a, ratio, C, max_search_steps, evaluate_moments, next_a);
}

// Bounded annealing schedule; invalid arithmetic or exhausted guards return
// false.  Finite inputs use the standard constants and schedule decisions.
template
<
    typename WalkType,
    typename Polytope,
    typename NT,
    typename RandomNumberGenerator,
    typename QuadraticStatistic,
    typename PrepareWalk
>
bool bounded_annealing_schedule(Polytope& P,
                                std::vector<NT> const& facet_distances,
                                NT ratio,
                                NT C,
                                NT frac,
                                unsigned int N,
                                unsigned int walk_length,
                                NT error,
                                std::vector<NT>& a_vals,
                                RandomNumberGenerator& rng,
                                ScheduleSampleBudget& budget,
                                QuadraticStatistic const& quadratic_statistic,
                                PrepareWalk const& prepare_walk)
{
    typedef typename Polytope::PointType Point;

    a_vals.clear();
    if (!std::isfinite(ratio) || ratio < NT(0) || !(ratio < NT(1)) ||
        !std::isfinite(C) || !(C > NT(0)) ||
        !std::isfinite(frac) || !(frac > NT(0)) || !(frac < NT(1)) ||
        N == 0 || walk_length == 0 ||
        !std::isfinite(error) || !(error > NT(0)) || budget.remaining == 0)
        return false;

    NT first_a;
    if (!bounded_first_gaussian(facet_distances, frac, error,
                                default_max_steps, default_max_steps, first_a))
        return false;
    a_vals.push_back(first_a);

    unsigned int total_steps;
    if (!schedule_sample_count(frac, error, total_steps)) return false;

    Point p(P.dimension());
    NT const decision_tol = NT(0.001);

    for (unsigned int stage = 0; stage < default_max_steps; ++stage) {
        NT const current_a = a_vals.back();
        NT next_a;
        if (!bounded_next_gaussian_with_statistic<WalkType>(
                P, p, current_a, N, ratio, C, walk_length, rng,
                default_max_steps, budget, next_a, quadratic_statistic,
                prepare_walk))
            return false;
        if (!budget.consume(total_steps)) return false;

        NT log_current_sum = -std::numeric_limits<NT>::infinity();
        WalkType walk(P, p, current_a, rng);
        if (!prepare_walk(walk, current_a)) return false;

        for (unsigned int j = 0; j < total_steps; ++j) {
            walk.apply(P, p, current_a, walk_length, rng);
            if (!p.getCoefficients().allFinite()) return false;
            NT const quadratic = quadratic_statistic(p);
            NT const log_weight = (current_a - next_a) * quadratic;
            if (!std::isfinite(quadratic) || quadratic < NT(0)) return false;
            if (!std::isfinite(log_weight)) return false;
            log_current_sum = log_add_exp(log_current_sum, log_weight);
        }

        NT const log_current_mean =
            log_current_sum - std::log(NT(total_steps));
        if (!std::isfinite(log_current_mean)) return false;
        if (next_a > NT(0) &&
            log_current_mean > std::log1p(decision_tol)) {
            a_vals.push_back(next_a);
        } else if (next_a == NT(0)) {
            a_vals.push_back(NT(0));
            return true;
        } else {
            a_vals.back() = NT(0);
            return a_vals.size() > 1;
        }
    }
    return false;
}

// Accumulate one log-weight per call.  Non-finite values or budget exhaustion
// return false rather than indicating convergence.
template <typename NT, typename LogWeightSampler>
bool accumulate_stage(unsigned int W,
                      NT window_tol,
                      unsigned long long max_samples,
                      LogWeightSampler&& sample_log_weight,
                      NT& log_ratio,
                      unsigned long long& samples)
{
    samples = 0;
    if (W == 0 || !std::isfinite(window_tol) || window_tol < NT(0))
        return false;
    if (static_cast<unsigned long long>(W) > max_samples)
        return false;

    NT shift = -std::numeric_limits<NT>::infinity();
    NT scaled_sum = NT(0);
    std::vector<NT> last_W(W, NT(0));
    NT min_val = std::numeric_limits<NT>::lowest();
    NT max_val = std::numeric_limits<NT>::max();
    unsigned int min_index = W - 1;
    unsigned int max_index = W - 1;
    unsigned int index = 0;

    while (samples < max_samples) {
        NT const log_w = sample_log_weight();
        ++samples;
        if (!std::isfinite(log_w)) return false;

        if (shift == -std::numeric_limits<NT>::infinity()) {
            shift = log_w;
            scaled_sum = NT(1);
        } else if (log_w <= shift) {
            NT const scaled_weight = std::exp(log_w - shift);
            scaled_sum += scaled_weight;
        } else {
            NT const old_scale = std::exp(shift - log_w);
            scaled_sum = scaled_sum * old_scale + NT(1);
            shift = log_w;
        }
        NT const count = NT(samples);
        NT const val = shift + std::log(scaled_sum / count);
        if (!std::isfinite(val)) return false;

        last_W[index] = val;
        if (val <= min_val) {
            min_val = val;
            min_index = index;
        } else if (min_index == index) {
            auto it = std::min_element(last_W.begin(), last_W.end());
            min_val = *it;
            min_index = static_cast<unsigned int>(it - last_W.begin());
        }

        if (val >= max_val) {
            max_val = val;
            max_index = index;
        } else if (max_index == index) {
            auto it = std::max_element(last_W.begin(), last_W.end());
            max_val = *it;
            max_index = static_cast<unsigned int>(it - last_W.begin());
        }

        if (samples >= W) {
            NT const spread = max_val - min_val;
            if (!std::isfinite(spread)) return false;
            if (spread <= window_tol) {
                log_ratio = val;
                return true;
            }
        }

        if (++index == W) index = 0;
    }
    return false;
}

} // namespace volume_cooling_gaussians_log_detail

template
<
    typename WalkTypePolicy,
    typename Polytope,
    typename RandomNumberGenerator
>
GaussianCoolingLogVolume<typename Polytope::PointType::FT>
volume_cooling_gaussians_log(
    Polytope& Pin,
    RandomNumberGenerator& rng,
    double const& error,
    unsigned int const& walk_length,
    unsigned long long max_samples)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename WalkTypePolicy::template Walk
                                              <
                                                    Polytope,
                                                    RandomNumberGenerator
                                              > WalkType;

    GaussianCoolingLogVolume<NT> out{
        false, std::numeric_limits<NT>::quiet_NaN()};

    if (!std::isfinite(error) || error <= 0.0 || walk_length == 0 ||
        max_samples == 0)
        throw std::invalid_argument(
            "log Gaussian cooling: invalid error, walk length, or sample budget");

    auto P(Pin); // copy: the polytope is shifted to its Chebychev center
    unsigned int const n = P.dimension();
    if (n == 0) {
        out.success = true;
        out.log_volume = NT(0);
        return out;
    }
    if (!volume_cooling_gaussians_log_detail::
            schedule_parameters_fit(n))
        return out;
    gaussian_annealing_parameters<NT> parameters(n);

    auto InnerBall = P.ComputeInnerBall();
    if (!std::isfinite(InnerBall.second) || InnerBall.second < 0.0) return out;

    Point c = InnerBall.first;
    if (!c.getCoefficients().allFinite()) return out;
    NT const radius = InnerBall.second;
    P.shift(c.getCoefficients());

    // Build the schedule under local termination and resource guards.
    std::vector<NT> a_vals;
    NT const ratio = parameters.ratio;
    NT const C = parameters.C;
    unsigned int const N = parameters.N;
    volume_cooling_gaussians_log_detail::ScheduleSampleBudget budget(
        max_samples);
    auto const quadratic = [](Point const& point) {
        return point.squared_length();
    };
    auto const prepare_walk = [&](WalkType& walk, NT current_a) {
        NT const denominator =
            std::sqrt(std::max(NT(1), current_a) * NT(n));
        NT const delta = NT(4) * radius / denominator;
        if (!std::isfinite(denominator) || !std::isfinite(delta))
            return false;
        volume_cooling_gaussians_log_detail::set_walk_delta(walk, delta);
        return true;
    };

    if (!volume_cooling_gaussians_log_detail::bounded_annealing_schedule
            <WalkType>(P, P.get_dists(radius), ratio, C, parameters.frac, N,
                       walk_length, NT(error), a_vals, rng, budget, quadratic,
                       prepare_walk))
        return out;

    unsigned int const W = parameters.W;
    unsigned int const mm = a_vals.size() - 1;

    NT log_vol = (NT(n) / NT(2)) * std::log(NT(M_PI) / a_vals[0]);
    if (!std::isfinite(log_vol)) return out;

    Point p(n);   // the origin: the shifted Chebychev center

    for (unsigned int i = 0; i < mm; ++i)
    {
        NT const a_curr = a_vals[i];
        NT const a_next = a_vals[i + 1];
        NT const curr_eps = error / std::sqrt(NT(mm));
        // Exact log-domain form of (max-min)/max <= eps/2.  For curr_eps >= 2
        // the criterion is automatic, avoiding an invalid logarithm.
        NT const window_tol = curr_eps < NT(2)
            ? -std::log(NT(1) - curr_eps / NT(2))
            : std::numeric_limits<NT>::max();

        WalkType walk(P, p, a_curr, rng);
        if (!prepare_walk(walk, a_curr)) return out;

        auto sample_log_weight = [&]() -> NT {
            walk.apply(P, p, a_curr, walk_length, rng);
            if (!p.getCoefficients().allFinite())
                return std::numeric_limits<NT>::quiet_NaN();
            NT const value = quadratic(p);
            if (!std::isfinite(value) || value < NT(0))
                return std::numeric_limits<NT>::quiet_NaN();
            return (a_curr - a_next) * value;
        };

        NT log_ratio;
        unsigned long long its;
        bool const stage_success =
            volume_cooling_gaussians_log_detail::accumulate_stage<NT>(
                W, window_tol, budget.remaining, sample_log_weight,
                log_ratio, its);
        budget.consume(its);
        if (!stage_success)
            return out;
        log_vol += log_ratio;
        if (!std::isfinite(log_vol))
            return out;
    }

    out.log_volume = log_vol;
    out.success = true;
    return out;
}

template
<
    typename WalkTypePolicy,
    typename Polytope,
    typename RandomNumberGenerator
>
GaussianCoolingLogVolume<typename Polytope::PointType::FT>
volume_cooling_gaussians_log(Polytope& Pin,
                             RandomNumberGenerator& rng,
                             double const& error = 0.1,
                             unsigned int const& walk_length = 1)
{
    return volume_cooling_gaussians_log<WalkTypePolicy>(
        Pin, rng, error, walk_length,
        volume_cooling_gaussians_log_detail::default_max_samples);
}

#endif // VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP
