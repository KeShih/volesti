// VolEsti (volume computation and sampling library)

// Copyright (c) 2026 Ke Shi
// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef TRIGONOMETRIC_EQUATION_SOLVERS_HPP
#define TRIGONOMETRIC_EQUATION_SOLVERS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace trig_solvers_detail {

// First time of the form base_time + k*period (integer k >= 0) reaching lower,
// tolerating a time_tol undershoot before stepping one period forward.
template <typename NT>
NT first_periodic_candidate(NT const& base_time, NT const& period,
                            NT const& lower, NT const& time_tol)
{
    NT k = std::ceil((lower - base_time) / period);
    if (k < NT(0)) k = NT(0);
    NT candidate = base_time + k * period;
    if (candidate < lower - time_tol)
        candidate += period;
    return candidate;
}

} // namespace trig_solvers_detail

// Solve a*cos(omega*t) + b*sin(omega*t) = c for the smallest t >= min_time.
// Degenerate identities are reported as no hit because they have no isolated root.
template <typename NT>
std::pair<NT, bool> first_trigonometric_solution(
    NT const& a, NT const& b, NT const& c, NT const& omega, NT const& min_time,
    NT const& time_tol = NT(1e-12), NT const& value_tol = NT(1e-12))
{
    assert(omega > NT(0));
    assert(min_time >= NT(0));

    NT const no_hit = std::numeric_limits<NT>::max();
    static NT const pi = std::acos(NT(-1));
    NT const two_pi = NT(2) * pi;
    NT const lower = std::max(min_time, NT(0));

    NT const radius = std::sqrt(a * a + b * b);
    NT const value_scale = std::max({NT(1), std::abs(radius), std::abs(c)});
    NT const value_eps = value_tol * value_scale;

    if (radius <= value_eps || std::abs(c) > radius + value_eps)
        return std::make_pair(no_hit, false);

    NT ratio = c / radius;
    if (ratio > NT(1)) ratio = NT(1);
    if (ratio < NT(-1)) ratio = NT(-1);

    NT const phase = std::atan2(b, a);
    NT const angle = std::acos(ratio);
    NT const period = two_pi / omega;

    NT best = no_hit;
    bool found = false;

    for (int sign = -1; sign <= 1; sign += 2) {
        NT const root_angle = phase + sign * angle;
        NT const base_time = root_angle / omega;
        NT const candidate = trig_solvers_detail::first_periodic_candidate(
            base_time, period, lower, time_tol);
        if (candidate < best) {
            best = candidate;
            found = true;
        }
    }

    return std::make_pair(best, found);
}


// Faster variant with precomputed inv_omega and period (uses 2x atan2).
template <typename NT>
std::pair<NT, bool> first_trigonometric_solution(
    NT const& a, NT const& b, NT const& c, NT const& inv_omega, NT const& period,
    NT const& min_time, NT const& time_tol, NT const& value_tol)
{
    assert(min_time >= NT(0));

    NT const no_hit = std::numeric_limits<NT>::max();
    NT const lower = std::max(min_time, NT(0));

    NT const R_sq = a * a + b * b;
    NT const value_scale = std::max(NT(1), std::max(std::abs(a) + std::abs(b), std::abs(c)));
    NT const value_eps = value_tol * value_scale;

    if (R_sq <= value_eps * value_eps)
        return {no_hit, false};

    NT const c_sq = c * c;
    NT const disc = R_sq - c_sq;

    if (disc < -value_eps * value_scale)
        return {no_hit, false};

    if (disc <= value_eps * value_scale) {
        // Tangent case: double root at maximum (+R) or minimum (-R).
        // phase = atan2(b,a) locates the maximum; for c < 0 the tangent
        // is at the minimum, half a period later.
        NT const phase = std::atan2(b, a);
        static NT const pi = std::acos(NT(-1));
        NT root_angle = (c >= NT(0)) ? phase : phase + pi;
        NT const base_time = root_angle * inv_omega;
        return {trig_solvers_detail::first_periodic_candidate(
                    base_time, period, lower, time_tol), true};
    }

    NT const D = std::sqrt(disc);
    NT t1 = std::atan2(b * c - a * D, a * c + b * D) * inv_omega;
    NT t2 = std::atan2(b * c + a * D, a * c - b * D) * inv_omega;

    NT best = no_hit;
    bool found = false;

    for (int idx = 0; idx < 2; ++idx) {
        NT base_time = (idx == 0) ? t1 : t2;
        NT const candidate = trig_solvers_detail::first_periodic_candidate(
            base_time, period, lower, time_tol);
        if (candidate < best) {
            best = candidate;
            found = true;
        }
    }

    return {best, found};
}

#endif
