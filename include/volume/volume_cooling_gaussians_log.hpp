// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP
#define VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "volume/volume_cooling_gaussians.hpp"

// Log-domain Gaussian-cooling volume estimation with per-stage
// diagnostics.  Same algorithm as volume_cooling_gaussians — the
// annealing schedule, the walk protocol, the sliding-window convergence
// test and its parameters are reused unchanged — but every quantity
// lives in the log domain:
//
//     log vol = (n/2) log(pi / a_0) + sum_i log rho_i,
//
//     rho_i = Z_{i+1} / Z_i = E_{x ~ pi_i}[ g_{i+1}(x) / g_i(x) ],
//
//     log rho_i = logsumexp(log w) - log(N),   log w = (a_i - a_{i+1}) |x|^2
//
// (the log of the arithmetic mean of the weights, never mean(log w),
// which would estimate the geometric mean and bias the ratio low).  The
// sliding-window termination criterion (max - min)/max <= eps/2 has the
// exact log-domain equivalent log_max - log_min <= -log(1 - eps/2).
// The result never materializes exp(log_volume), so it stays finite
// where the raw volume of a high-dimensional body under- or overflows.

template <typename NT>
struct CoolingStageDiagnostics
{
    NT a_curr;                  // annealing parameter of the sampled stage
    NT a_next;                  // parameter of the next stage
    NT log_ratio;               // logsumexp(log w) - log(samples)
    NT cv2;                     // sample CV^2 of the weights: E[w^2]/E[w]^2 - 1
    unsigned long long samples; // samples spent on this ratio
};

template <typename NT>
struct GaussianCoolingLogVolume
{
    bool success;
    NT log_volume;              // log of the estimated volume
    NT log_first_gaussian;      // (n/2) log(pi / a_0)
    std::vector<CoolingStageDiagnostics<NT>> stages;
    unsigned long long total_samples;
    unsigned long long violations;     // containment failures among samples
                                       // (counted iff count_violations)
    double schedule_seconds;
    double sampling_seconds;
};

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
                             unsigned int const& walk_length = 1,
                             bool count_violations = false)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename WalkTypePolicy::template Walk
                                              <
                                                    Polytope,
                                                    RandomNumberGenerator
                                              > WalkType;
    typedef GaussianRandomPointGenerator<WalkType> RandomPointGenerator;

    GaussianCoolingLogVolume<NT> out;
    out.success = false;
    out.log_volume = std::numeric_limits<NT>::quiet_NaN();
    out.log_first_gaussian = std::numeric_limits<NT>::quiet_NaN();
    out.total_samples = 0;
    out.violations = 0;
    out.schedule_seconds = 0;
    out.sampling_seconds = 0;

    auto P(Pin); // copy: the polytope is shifted to its Chebychev center
    unsigned int const n = P.dimension();
    gaussian_annealing_parameters<NT> parameters(n);

    auto InnerBall = P.ComputeInnerBall();
    if (InnerBall.second < 0.0) return out;

    Point c = InnerBall.first;
    NT const radius = InnerBall.second;
    P.shift(c.getCoefficients());

    // Annealing schedule (identical to the linear-domain routine).
    std::vector<NT> a_vals;
    NT const ratio = parameters.ratio;
    NT const C = parameters.C;
    unsigned int const N = parameters.N;

    auto const t_sched0 = std::chrono::steady_clock::now();
    compute_annealing_schedule
    <
        WalkType,
        RandomPointGenerator
    >(P, ratio, C, parameters.frac, N, walk_length, radius, error, a_vals, rng);
    out.schedule_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_sched0).count();

    unsigned int const W = parameters.W;
    unsigned int const mm = a_vals.size() - 1;

    NT log_vol = (NT(n) / NT(2)) * std::log(NT(M_PI) / a_vals[0]);
    out.log_first_gaussian = log_vol;

    NT const neg_inf = -std::numeric_limits<NT>::infinity();
    auto log_add_exp = [&](NT a, NT b) {
        if (a == neg_inf) return b;
        if (b == neg_inf) return a;
        NT const m = std::max(a, b);
        return m + std::log(std::exp(a - m) + std::exp(b - m));
    };

    Point p(n);   // the origin: the shifted Chebychev center
    auto const t_samp0 = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < mm; ++i)
    {
        NT const a_curr = a_vals[i];
        NT const a_next = a_vals[i + 1];
        NT const curr_eps = error / std::sqrt(NT(mm));
        // Exact log-domain form of (max - min)/max <= eps/2.  For
        // curr_eps >= 2 the linear criterion (max-min)/max <= eps/2 is
        // trivially satisfied (the ratio never exceeds 1 for positive
        // running means), so the log form saturates to "always done"
        // instead of feeding a negative argument to log (which would be
        // NaN and loop forever).
        NT const window_tol = curr_eps < NT(2)
            ? -std::log(NT(1) - curr_eps / NT(2))
            : std::numeric_limits<NT>::max();

        bool done = false;
        unsigned long long its = 0;
        NT lse = neg_inf;    // logsumexp of the weights
        NT lse2 = neg_inf;   // logsumexp of the squared weights
        std::vector<NT> last_W(W, NT(0));
        NT min_val = std::numeric_limits<NT>::lowest();
        NT max_val = std::numeric_limits<NT>::max();
        unsigned int min_index = W - 1;
        unsigned int max_index = W - 1;
        unsigned int index = 0;

        WalkType walk(P, p, a_curr, rng);
        update_delta<WalkType>
                ::apply(walk, 4.0 * radius
                         / std::sqrt(std::max(NT(1.0), a_curr) * NT(n)));

        while (!done)
        {
            walk.apply(P, p, a_curr, walk_length, rng);
            if (count_violations && P.is_in(p, NT(1e-7)) != -1)
                ++out.violations;

            ++its;
            NT const log_w = (a_curr - a_next) * p.squared_length();
            lse = log_add_exp(lse, log_w);
            lse2 = log_add_exp(lse2, NT(2) * log_w);
            NT const val = lse - std::log(NT(its));   // running log-mean

            last_W[index] = val;
            if (val <= min_val)
            {
                min_val = val;
                min_index = index;
            } else if (min_index == index)
            {
                auto it = std::min_element(last_W.begin(), last_W.end());
                min_val = *it;
                min_index = std::distance(last_W.begin(), it);
            }

            if (val >= max_val)
            {
                max_val = val;
                max_index = index;
            } else if (max_index == index)
            {
                auto it = std::max_element(last_W.begin(), last_W.end());
                max_val = *it;
                max_index = std::distance(last_W.begin(), it);
            }

            if (its >= W && max_val - min_val <= window_tol)
                done = true;

            index = index % W + 1;
            if (index == W) index = 0;
        }

        NT const log_ratio = lse - std::log(NT(its));
        CoolingStageDiagnostics<NT> stage;
        stage.a_curr = a_curr;
        stage.a_next = a_next;
        stage.log_ratio = log_ratio;
        // CV^2 = E[w^2]/E[w]^2 - 1 in the log domain.
        stage.cv2 = std::exp((lse2 - std::log(NT(its)))
                             - NT(2) * log_ratio) - NT(1);
        stage.samples = its;
        out.stages.push_back(stage);
        out.total_samples += its;

        log_vol += log_ratio;
    }

    out.sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_samp0).count();
    out.log_volume = log_vol;
    out.success = true;
    return out;
}

#endif // VOLUME_VOLUME_COOLING_GAUSSIANS_LOG_HPP
