// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
#define VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "convex_bodies/orderpolytope.h"
#include "misc/order_polytope_volume_reduction.h"
#include "misc/poset.h"
#include "random_walks/random_walks.hpp"
#include "volume/sampling_policies.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_gaussians_log.hpp"


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


// ---------------------------------------------------------------------
// Backend-selectable Gaussian-cooling integration.
//
// volume_cooling_gaussians drives its walk through the
// (P, p, a_i, rng) constructor and apply(P, p, a_i, walk_length, rng):
// every annealing stage samples the spherical Gaussian exp(-a_i |x|^2)
// on the (shifted) polytope.  The adapters below map each stage onto
// the general-Gaussian walks with the stage's correct target: precision
// A = 2 a_i I, center 0, and the backend's mass policy.  For these
// common-frequency targets the quarter-period leg cap is exact, so the
// modal adapter sets it explicitly (arbitrary multi-frequency targets
// must not use it - the modal walk's own default is diameter-capped).

struct OrderPolytopeMatchedDiagonalCoolingWalk
{
    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk Policy;

        Walk(Polytope &P, Point const& p, NT const& a_i,
             RandomNumberGenerator &rng)
            : _inner(std::make_shared<Inner>(P, p,
                     typename Policy::parameters<NT>(
                         VT(VT::Constant(P.dimension(), NT(2) * a_i)),
                         VT(VT::Zero(P.dimension())),
                         std::sqrt(NT(2) * a_i)),
                     rng))
        {}

        inline void apply(Polytope const& P, Point& p, NT const&,
                          unsigned int const& walk_length,
                          RandomNumberGenerator &rng)
        {
            _inner->apply(P, p, walk_length, rng);
        }

        // shared_ptr: the cooling driver's generic update_delta no-op
        // takes walks by value, and the inner walk is move-only.
        typedef typename Policy::template Walk<Polytope, RandomNumberGenerator> Inner;
        std::shared_ptr<Inner> _inner;
    };
};

struct OrderPolytopeMatchedDenseCoolingWalk
{
    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::MT MT;
        typedef typename Polytope::VT VT;
        typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk Policy;

        Walk(Polytope &P, Point const& p, NT const& a_i,
             RandomNumberGenerator &rng)
            : _inner(std::make_shared<Inner>(P, p,
                     typename Policy::parameters<NT>(
                         MT(NT(2) * a_i * MT::Identity(P.dimension(),
                                                       P.dimension())),
                         VT(VT::Zero(P.dimension())),
                         std::sqrt(NT(2) * a_i),
                         GaussianHMCBackend::MatchedDenseAngleFullScan),
                     rng))
        {}

        inline void apply(Polytope const& P, Point& p, NT const&,
                          unsigned int const& walk_length,
                          RandomNumberGenerator &rng)
        {
            _inner->apply(P, p, walk_length, rng);
        }

        typedef typename Policy::template Walk<Polytope, RandomNumberGenerator> Inner;
        std::shared_ptr<Inner> _inner;
    };
};

struct OrderPolytopeModalFallbackCoolingWalk
{
    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::MT MT;
        typedef typename Polytope::VT VT;
        typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk Policy;

        static typename Policy::parameters<NT> stage_params(Polytope &P,
                                                            NT const& a_i)
        {
            unsigned int const n = P.dimension();
            typename Policy::parameters<NT> params(
                MT(NT(2) * a_i * MT::Identity(n, n)),
                MT(MT::Identity(n, n)),
                VT(VT::Zero(n)));
            // This stage target is common-frequency (all omega_k equal
            // sqrt(2 a_i)), so the quarter-period cap is exact here.
            NT const omega = std::sqrt(NT(2) * a_i);
            params.m_L = std::min(NT(1),
                NT(1.57079632679489661923) / omega);
            params.set_L = true;
            return params;
        }

        Walk(Polytope &P, Point const& p, NT const& a_i,
             RandomNumberGenerator &rng)
            : _inner(std::make_shared<Inner>(P, p, stage_params(P, a_i), rng))
        {}

        inline void apply(Polytope const& P, Point& p, NT const&,
                          unsigned int const& walk_length,
                          RandomNumberGenerator &rng)
        {
            _inner->apply(P, p, walk_length, rng);
        }

        typedef typename Policy::template Walk<Polytope, RandomNumberGenerator> Inner;
        std::shared_ptr<Inner> _inner;
    };
};


// Log-domain aggregation: ratio and repeated-run estimates are combined
// as log(mean(exp(log_w))) = logsumexp(log_w) - log(N), never as
// mean(log_w) (which estimates the log of the geometric mean and is
// biased low for the arithmetic mean the estimators need).
template <typename NT>
NT log_sum_exp(std::vector<NT> const& xs)
{
    if (xs.empty()) return -std::numeric_limits<NT>::infinity();
    NT const m = *std::max_element(xs.begin(), xs.end());
    if (!std::isfinite(m)) return m;
    NT s = NT(0);
    for (NT const& x : xs) s += std::exp(x - m);
    return m + std::log(s);
}

template <typename NT>
NT log_mean_exp(std::vector<NT> const& xs)
{
    // The empty mean is exp-space zero: -inf, not (-inf) - log(0) = NaN.
    if (xs.empty()) return -std::numeric_limits<NT>::infinity();
    return log_sum_exp(xs) - std::log(static_cast<NT>(xs.size()));
}


struct LinearExtensionOptions
{
    // Auto behaves as SphericalEventQueue: the annealing targets are
    // spherical Gaussians, for which the purpose-built order-polytope
    // event-queue walk is exact and extensively validated, so it is the
    // conservative default *for this order-polytope-specific wrapper*.
    // Generic volume calls elsewhere in the library are untouched; the
    // classic generic sampler remains available via use_generic_cdhr.
    GaussianHMCBackend backend = GaussianHMCBackend::Auto;
    // Ignore `backend` and drive the cooling with the generic
    // GaussianCDHRWalk (the library's traditional default sampler).
    bool use_generic_cdhr = false;
    // Count containment violations among all ratio samples (cheap; off
    // by default to keep the sampling loop lean).
    bool count_violations = false;
    double error = 0.1;
    unsigned int walk_length = 1;
    unsigned int repetitions = 1;
    OrderPolytopeVolumeReductionOptions reduction;
};

template <typename NT>
struct LogVolumeEstimate
{
    NT log_volume;              // logsumexp-combined across repetitions
    NT se_log_volume;           // spread of per-run log volumes (0 if 1 run)
    NT log_factorial;           // lgamma(n + 1)
    NT log_extensions;          // log_volume + log_factorial
    unsigned int repetitions;
    GaussianHMCBackend backend;
    bool used_generic_cdhr;
    // Profile-counter deltas over all repetitions; zero unless compiled
    // with VOLESTI_HMC_PROFILE.
    unsigned long long reflections;
    unsigned long long boundary_solves;
    // Aggregated sampling diagnostics over all repetitions/residuals.
    unsigned long long total_samples;
    unsigned long long violations;      // iff count_violations
    double schedule_seconds;
    double sampling_seconds;
    // Per-stage cooling diagnostics of the LAST residual of the LAST
    // repetition (a_i, log-ratio, samples, weight CV^2) for benchmarks.
    std::vector<CoolingStageDiagnostics<NT>> stages;
    // Reduction diagnostics of the last repetition.
    unsigned int n;
    unsigned int residual_count;
    bool reduction_exact;
    NT log_volume_offset;
};

// One repetition: poset reduction, then log-domain Gaussian cooling on
// every residual (log-ratios accumulated as logsumexp(log w) - log N,
// per stage, never through a raw volume).
template <typename WalkTypePolicy, typename RandomNumberGenerator,
          typename Point, typename NT>
NT reduced_log_volume_once(Poset const& poset,
                           RandomNumberGenerator& rng,
                           LinearExtensionOptions const& options,
                           LogVolumeEstimate<NT>& out)
{
    ReducedOrderPolytopeVolumeProblem const reduced =
        reduce_order_polytope_volume_problem(poset, options.reduction);
    out.residual_count =
        static_cast<unsigned int>(reduced.residual_posets.size());
    out.reduction_exact = reduced.is_exact();
    out.log_volume_offset = static_cast<NT>(reduced.log_volume_offset);

    NT log_volume = static_cast<NT>(reduced.log_volume_offset);
    NT const per_residual_error = out.residual_count > 1
        ? NT(options.error) / std::sqrt(NT(out.residual_count))
        : NT(options.error);

    for (auto const& residual : reduced.residual_posets) {
        OrderPolytope<Point> OP(residual);
        auto const r = volume_cooling_gaussians_log<WalkTypePolicy>(
            OP, rng, double(per_residual_error), options.walk_length,
            options.count_violations);
        if (!r.success || !std::isfinite(r.log_volume))
            return std::numeric_limits<NT>::quiet_NaN();
        log_volume += r.log_volume;
        out.total_samples += r.total_samples;
        out.violations += r.violations;
        out.schedule_seconds += r.schedule_seconds;
        out.sampling_seconds += r.sampling_seconds;
        out.stages = r.stages;
    }
    return log_volume;
}

// Backend-selectable log-count wrapper: log_count = log_volume +
// lgamma(n + 1), with every ratio accumulated in the log domain.
// Additive - the templated estimate_log_linear_extensions API above is
// unchanged.
template <typename RandomNumberGenerator, typename Point,
          typename NT = typename Point::FT>
LogVolumeEstimate<NT>
estimate_log_linear_extensions(Poset const& poset,
                               RandomNumberGenerator& rng,
                               LinearExtensionOptions const& options)
{
    LogVolumeEstimate<NT> out;
    out.repetitions = std::max(1u, options.repetitions);
    out.backend = options.backend;
    out.used_generic_cdhr = options.use_generic_cdhr;
    out.n = poset.num_elem();
    out.log_factorial = std::lgamma(NT(poset.num_elem() + 1));
    out.reflections = 0;
    out.boundary_solves = 0;
    out.total_samples = 0;
    out.violations = 0;
    out.schedule_seconds = 0;
    out.sampling_seconds = 0;
    out.residual_count = 0;
    out.reduction_exact = false;
    out.log_volume_offset = NT(0);

    auto run_once = [&]() -> NT {
        if (options.use_generic_cdhr)
            return reduced_log_volume_once
                <GaussianCDHRWalk, RandomNumberGenerator, Point, NT>(
                    poset, rng, options, out);
        switch (options.backend) {
            case GaussianHMCBackend::MatchedDiagonalEventQueue:
                return reduced_log_volume_once
                    <OrderPolytopeMatchedDiagonalCoolingWalk,
                     RandomNumberGenerator, Point, NT>(poset, rng, options, out);
            case GaussianHMCBackend::MatchedDenseAngleFullScan:
                return reduced_log_volume_once
                    <OrderPolytopeMatchedDenseCoolingWalk,
                     RandomNumberGenerator, Point, NT>(poset, rng, options, out);
            case GaussianHMCBackend::ModalDenseFallback:
                return reduced_log_volume_once
                    <OrderPolytopeModalFallbackCoolingWalk,
                     RandomNumberGenerator, Point, NT>(poset, rng, options, out);
            default:   // Auto, SphericalEventQueue
                return reduced_log_volume_once
                    <OrderPolytopeGaussianHamiltonianMonteCarloExactWalk,
                     RandomNumberGenerator, Point, NT>(poset, rng, options, out);
        }
    };

#ifdef VOLESTI_HMC_PROFILE
    unsigned long long const refl0 =
        hmc_profile_counters::n_reflections.load();
    unsigned long long const solv0 =
        hmc_profile_counters::n_trig_calls.load();
#endif

    std::vector<NT> log_vols;
    for (unsigned int r = 0; r < out.repetitions; ++r)
        log_vols.push_back(run_once());

#ifdef VOLESTI_HMC_PROFILE
    out.reflections = hmc_profile_counters::n_reflections.load() - refl0;
    out.boundary_solves = hmc_profile_counters::n_trig_calls.load() - solv0;
#endif

    out.log_volume = log_mean_exp(log_vols);
    NT se = NT(0);
    if (log_vols.size() > 1) {
        NT mean = NT(0);
        for (NT const& lv : log_vols) mean += lv;
        mean /= NT(log_vols.size());
        for (NT const& lv : log_vols) se += (lv - mean) * (lv - mean);
        se = std::sqrt(se / (NT(log_vols.size() - 1) * NT(log_vols.size())));
    }
    out.se_log_volume = se;
    out.log_extensions = out.log_volume + out.log_factorial;
    return out;
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
