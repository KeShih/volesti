// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

// Non-CI benchmark for the order-polytope Gaussian HMC backends:
//
//     spherical  - SphericalEventQueue, A = a I (the PR-0 walk)
//     diag-EQ    - MatchedDiagonalEventQueue, nonuniform diagonal A
//     dense-scan - MatchedDenseAngleFullScan, same diagonal A
//
// Reports us/step, facet solves per reflection (the recomputation count
// the diagonal event queue is supposed to shrink), heap pops and stale
// events per reflection, and the containment-violation rate.
//
// Build with -DVOLESTI_HMC_PROFILE (set on the CMake target).

#include <chrono>
#include <cstdio>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"

#include "random_walks/random_walks.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef OrderPolytope<Point> OP_t;
typedef typename OP_t::MT MT;
typedef typename OP_t::VT VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 127> RNGType;
typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk MatchedPolicy;
typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk SphericalPolicy;

struct CounterSnapshot {
    unsigned long long trig_calls, reflections, heap_pops, stale;
    static CounterSnapshot take()
    {
        return {hmc_profile_counters::n_trig_calls.load(),
                hmc_profile_counters::n_reflections.load(),
                hmc_profile_counters::n_heap_remove.load(),
                hmc_profile_counters::n_stale_purge.load()};
    }
};

template <typename WalkT>
void run_one(char const* label, OP_t& OP, WalkT& walk, RNGType& rng,
             unsigned int steps)
{
    unsigned int const warmup = steps / 10;
    Point p(OP.dimension());
    p = Point(OP.inner_point());
    for (unsigned int s = 0; s < warmup; ++s) walk.apply(OP, p, 1, rng);

    CounterSnapshot const c0 = CounterSnapshot::take();
    unsigned int violations = 0;
    auto const t0 = std::chrono::steady_clock::now();
    for (unsigned int s = 0; s < steps; ++s) {
        walk.apply(OP, p, 1, rng);
        if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
    }
    auto const t1 = std::chrono::steady_clock::now();
    CounterSnapshot const c1 = CounterSnapshot::take();

    double const us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                    / double(steps);
    double const refl = double(c1.reflections - c0.reflections);
    double const solves_per_refl =
        refl > 0 ? double(c1.trig_calls - c0.trig_calls) / refl : 0.0;
    double const pops_per_refl =
        refl > 0 ? double(c1.heap_pops - c0.heap_pops) / refl : 0.0;
    double const stale_per_refl =
        refl > 0 ? double(c1.stale - c0.stale) / refl : 0.0;

    std::printf("  %-11s %9.2f us/step %8.1f refl/step %10.2f solves/refl "
                "%8.2f pops/refl %7.3f stale/refl %8.5f%% violations\n",
                label, us, refl / double(steps), solves_per_refl,
                pops_per_refl, stale_per_refl,
                100.0 * double(violations) / double(steps));
}

void bench_poset(char const* name, unsigned int n, Poset::RV rels,
                 unsigned int steps)
{
    Poset poset(n, rels);
    OP_t OP(poset);
    std::printf("%s: n=%u, true facets=%u, %u steps\n",
                name, n, OP.num_true_facets(), steps);

    NT const a = NT(1);
    NT const omega_sph = std::sqrt(NT(2) * a);

    VT a_diag(n), center(n);
    for (unsigned int i = 0; i < n; ++i) {
        a_diag(i) = NT(0.5) + NT(7.5) * NT(i) / NT(n > 1 ? n - 1 : 1);
        center(i) = NT(0.5);
    }
    NT const omega = NT(1.5);

    {
        Point start(OP.inner_point());
        RNGType rng(n);
        typename SphericalPolicy::template Walk<OP_t, RNGType> walk(
            OP, start, a, rng);
        // the spherical walk's apply takes (and ignores) an a_i argument
        struct Adapter {
            typename SphericalPolicy::template Walk<OP_t, RNGType>& w;
            NT a;
            void apply(OP_t const& P, Point& p, unsigned int len, RNGType& r)
            { w.apply(P, p, a, len, r); }
        } adapter{walk, a};
        run_one("spherical", OP, adapter, rng, steps);
    }
    {
        Point start(OP.inner_point());
        RNGType rng(n);
        typename MatchedPolicy::parameters<NT> params(a_diag, center, omega);
        typename MatchedPolicy::template Walk<OP_t, RNGType> walk(
            OP, start, params, rng);
        run_one("diag-EQ", OP, walk, rng, steps);
    }
    {
        Point start(OP.inner_point());
        RNGType rng(n);
        typename MatchedPolicy::parameters<NT> params(
            MT(a_diag.asDiagonal()), center, omega,
            GaussianHMCBackend::MatchedDenseAngleFullScan);
        typename MatchedPolicy::template Walk<OP_t, RNGType> walk(
            OP, start, params, rng);
        run_one("dense-scan", OP, walk, rng, steps);
    }
    std::printf("\n");
}

int main()
{
    {
        Poset::RV chain;
        for (unsigned int i = 0; i + 1 < 30; ++i) chain.push_back({i, i + 1});
        bench_poset("chain-30", 30, chain, 20000);
    }
    {
        Poset::RV chain;
        for (unsigned int i = 0; i + 1 < 50; ++i) chain.push_back({i, i + 1});
        bench_poset("chain-50", 50, chain, 20000);
    }
    {
        Poset::RV rels;
        bench_poset("antichain-16", 16, rels, 20000);
    }
    {
        // Sparse layered poset on 16 elements.
        Poset::RV rels;
        for (unsigned int i = 0; i < 4; ++i) {
            rels.push_back({i, 4 + i});
            rels.push_back({i, 4 + (i + 1) % 4});
            rels.push_back({4 + i, 8 + i});
            rels.push_back({8 + i, 12 + i});
        }
        bench_poset("layered-16", 16, rels, 20000);
    }
    return 0;
}
