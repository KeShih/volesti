// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include <cmath>
#include <iostream>
#include <limits>

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"

#include "random_walks/random_walks.hpp"
#include "sampling/sampling.hpp"

typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk MatchedPolicy;
typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk SphericalPolicy;

// Mean of pi(x) ~ exp(-prec/2 (x - c)^2) truncated to [0, 1], by Simpson.
template <typename NT>
NT truncated_normal_mean(NT prec, NT c)
{
    int const N = 2000;
    NT const h = NT(1) / NT(N);
    NT num = NT(0), den = NT(0);
    for (int i = 0; i <= N; ++i) {
        NT const x = NT(i) * h;
        NT const w = (i == 0 || i == N) ? NT(1) : (i % 2 ? NT(4) : NT(2));
        NT const f = std::exp(-NT(0.5) * prec * (x - c) * (x - c));
        num += w * f * x;
        den += w * f;
    }
    return num / den;
}


// A. Spherical equivalence: for A = 2a I, c = 0 and omega = sqrt(2a) the
// matched mass is M = I and the dense full-scan backend must agree with
// the spherical event-queue backend leg by leg.
template <typename NT>
void call_test_matched_spherical_equivalence() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 29> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;
    typedef typename SphericalPolicy::template Walk<OP_t, RNGType> SWalk;

    struct Case {
        unsigned int n;
        RV rels;
        NT a;
        std::vector<NT> p, v;
        NT T;
    };
    std::vector<Case> cases = {
        // PR-0 exact fixture: two simultaneous lower facets (omega = 1).
        {2, RV{}, NT(0.5), {NT(0.5), NT(0.5)}, {NT(-0.375), NT(-0.375)},
         NT(1.57079632679489661923)},
        // PR-0 exact fixture: cover + bound simultaneous on a 3-chain.
        {3, RV{{0, 1}, {1, 2}}, NT(0.5),
         {NT(0.25), NT(0.375), NT(0.5)}, {NT(0.34375), NT(0.25), NT(0.875)},
         NT(1.57079632679489661923)},
        // Generic multi-reflection leg, omega != 1.
        {3, RV{{0, 1}, {1, 2}}, NT(1.25),
         {NT(0.3), NT(0.4), NT(0.6)}, {NT(0.9), NT(-0.7), NT(0.2)}, NT(1.3)},
        {4, RV{{0, 1}, {0, 2}, {1, 3}, {2, 3}}, NT(2),
         {NT(0.2), NT(0.45), NT(0.55), NT(0.8)},
         {NT(-0.8), NT(0.6), NT(-0.4), NT(1.1)}, NT(2.6)},
    };

    for (auto const& tc : cases) {
        RV rels = tc.rels;
        Poset poset(tc.n, rels);
        OP_t OP(poset);
        NT const omega = std::sqrt(NT(2) * tc.a);

        Point start(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i)
            start.set_coord(i, tc.p[i]);
        Point v0(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i)
            v0.set_coord(i, tc.v[i]);

        RNGType rng_m(tc.n);
        // Explicit dense backend: under Auto a spherical A = 2a I is (now)
        // detected as diagonal and routed to the diagonal event queue,
        // which has its own reduction test; this one pins the full scan.
        typename MatchedPolicy::parameters<NT> params(
            MT(NT(2) * tc.a * MT::Identity(tc.n, tc.n)), VT(VT::Zero(tc.n)),
            omega, GaussianHMCBackend::MatchedDenseAngleFullScan);
        MWalk mwalk(OP, start, params, rng_m);
        CHECK(mwalk.backend() == GaussianHMCBackend::MatchedDenseAngleFullScan);

        RNGType rng_s(tc.n);
        SWalk swalk(OP, start, tc.a, rng_s);

        Point pm = start, vm = v0;
        Point ps = start;
        CHECK(mwalk.apply_leg(OP, pm, vm, tc.T));
        CHECK(swalk.apply_leg(OP, ps, v0, tc.T));
        Point vs = swalk.velocity();

        for (unsigned int i = 0; i < tc.n; ++i) {
            CHECK(std::abs(pm[i] - ps[i]) < NT(1e-12));
            CHECK(std::abs(vm[i] - vs[i]) < NT(1e-12));
        }
        CHECK(OP.is_in(pm, NT(1e-9)) == -1);
    }
}


// Identity policy under Auto must route to the spherical event queue and
// behave identically to using that walk directly.
template <typename NT>
void call_test_matched_identity_delegation() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 31> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;
    typedef typename SphericalPolicy::template Walk<OP_t, RNGType> SWalk;

    RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);

    NT const a = NT(1.25);
    NT const omega = std::sqrt(NT(2) * a);
    Point start(3, {NT(0.3), NT(0.5), NT(0.7)});

    RNGType rng_m(3);
    typename MatchedPolicy::parameters<NT> params(omega);
    MWalk mwalk(OP, start, params, rng_m);
    CHECK(mwalk.backend() == GaussianHMCBackend::SphericalEventQueue);

    RNGType rng_s(3);
    SWalk swalk(OP, start, a, rng_s);

    Point pm = start, vm(3, {NT(0.9), NT(-0.7), NT(0.2)});
    Point ps = start, vs0(3, {NT(0.9), NT(-0.7), NT(0.2)});
    CHECK(mwalk.apply_leg(OP, pm, vm, NT(1.3)));
    CHECK(swalk.apply_leg(OP, ps, vs0, NT(1.3)));
    Point vs = swalk.velocity();
    for (unsigned int i = 0; i < 3; ++i) {
        CHECK(std::abs(pm[i] - ps[i]) < NT(1e-15));
        CHECK(std::abs(vm[i] - vs[i]) < NT(1e-15));
    }

    // Dense precision must never run on the incident-only event queue.
    typename MatchedPolicy::parameters<NT> bad(
        typename OP_t::MT(OP_t::MT::Identity(3, 3)),
        typename OP_t::VT(OP_t::VT::Zero(3)), NT(1),
        GaussianHMCBackend::SphericalEventQueue);
    RNGType rng_b(3);
    CHECK_THROWS_AS(MWalk(OP, start, bad, rng_b), std::invalid_argument);
}


// B. Matched reflection invariants on every facet type, for a dense SPD A:
// n^T v flips sign, the kinetic form v^T A v is preserved, and the
// reflection is an involution.
template <typename NT>
void call_test_matched_reflection_invariants() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 37> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    // Facet normals follow the walk's build order: lower walls (by
    // coordinate), upper walls, then covers.
    auto run = [&](unsigned int n, RV rels, std::vector<VT> const& normals) {
        Poset poset(n, rels);
        OP_t OP(poset);
        MT A(n, n);
        for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
                A(i, j) = (i == j ? NT(2) : NT(0)) + NT(0.5) / NT(1 + i + j);

        Point start(OP.inner_point());
        RNGType rng(n);
        typename MatchedPolicy::parameters<NT> params(
            A, VT(VT::Constant(n, NT(0.5))), NT(1.1));
        MWalk walk(OP, start, params, rng);

        REQUIRE(walk.num_event_facets() == normals.size());

        for (unsigned int fid = 0; fid < normals.size(); ++fid) {
            for (unsigned int trial = 0; trial < 3; ++trial) {
                VT v(n);
                for (unsigned int i = 0; i < n; ++i)
                    v(i) = NT(0.7) - NT(0.3) * NT(i) + NT(0.1) * NT(fid)
                         + NT(0.37) * NT(trial) * (i % 2 ? NT(-1) : NT(1));

                NT const nv_before = normals[fid].dot(v);
                NT const K_before = v.dot(A * v);
                VT v1 = v;
                walk.reflect_velocity(fid, v1);

                // Pin the exact map, not just its invariants: the metric
                // reflection with an independently solved w = A^{-1} n.
                VT const w = A.llt().solve(normals[fid]);
                VT const v_ref =
                    v - (NT(2) * nv_before / normals[fid].dot(w)) * w;
                for (unsigned int i = 0; i < n; ++i)
                    CHECK(std::abs(v1(i) - v_ref(i)) < NT(1e-13));

                CHECK(std::abs(normals[fid].dot(v1) + nv_before) < NT(1e-12));
                CHECK(std::abs(v1.dot(A * v1) - K_before) <
                      NT(1e-12) * std::max(NT(1), std::abs(K_before)));

                VT v2 = v1;
                walk.reflect_velocity(fid, v2);
                for (unsigned int i = 0; i < n; ++i)
                    CHECK(std::abs(v2(i) - v(i)) < NT(1e-12));
            }
        }
    };

    {
        // Antichain n = 3: 3 lower + 3 upper wall facets.
        unsigned int const n = 3;
        std::vector<VT> normals;
        for (unsigned int i = 0; i < n; ++i) {
            VT e(VT::Zero(n)); e(i) = NT(-1); normals.push_back(e);
        }
        for (unsigned int i = 0; i < n; ++i) {
            VT e(VT::Zero(n)); e(i) = NT(1); normals.push_back(e);
        }
        run(n, RV{}, normals);
    }
    {
        // Chain 0 < 1: lb(0), ub(1), cover e_0 - e_1.
        unsigned int const n = 2;
        std::vector<VT> normals;
        VT n0(VT::Zero(n)); n0(0) = NT(-1); normals.push_back(n0);
        VT n1(VT::Zero(n)); n1(1) = NT(1);  normals.push_back(n1);
        VT n2(n); n2 << NT(1), NT(-1);      normals.push_back(n2);
        run(n, RV{{0, 1}}, normals);
    }
}


// C. Feasibility and Hamiltonian conservation with a dense SPD A over
// chain, antichain, diamond and a sparse random-like poset.
template <typename NT>
void call_test_matched_feasibility() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 41> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    std::vector<std::pair<unsigned int, RV>> posets;
    {
        RV chain;
        for (unsigned int i = 0; i + 1 < 6; ++i) chain.push_back({i, i + 1});
        posets.push_back({6, chain});
    }
    posets.push_back({4, RV{}});
    posets.push_back({4, RV{{0, 1}, {0, 2}, {1, 3}, {2, 3}}});
    posets.push_back({8, RV{{0, 2}, {1, 2}, {2, 5}, {3, 5}, {1, 4},
                            {4, 6}, {5, 7}}});

    for (auto& pr : posets) {
        unsigned int const n = pr.first;
        Poset poset(n, pr.second);
        OP_t OP(poset);

        // Dense SPD precision with genuinely coupled cover reflections:
        // for the Kac-Murdock-Szego matrix A(i,j) = 2 * 0.5^|i-j| the
        // difference vectors e_u - e_v are NOT eigenvectors (unlike
        // I + rho * ones, whose cover reflections degenerate to the
        // spherical velocity swap), so A^{-1} n is not parallel to n.
        MT A(n, n);
        for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
                A(i, j) = NT(2) * std::pow(NT(0.5), std::abs(int(i) - int(j)));
        VT center(VT::Constant(n, NT(0.5)));
        typename MatchedPolicy::parameters<NT> params(A, center, NT(1.3));

        Point p(OP.inner_point());
        RNGType rng(n);
        MWalk walk(OP, p, params, rng);

        // Hamiltonian conservation over one deterministic leg.
        Point pl = p;
        Point vl(n);
        for (unsigned int i = 0; i < n; ++i)
            vl.set_coord(i, NT(0.8) - NT(0.4) * NT(i % 3));
        NT const H0 = walk.hamiltonian(pl, vl);
        CHECK(walk.apply_leg(OP, pl, vl, NT(1.1)));
        NT const H1 = walk.hamiltonian(pl, vl);
        CHECK(std::abs(H1 - H0) < NT(1e-9) * std::max(NT(1), std::abs(H0)));
        CHECK(OP.is_in(pl, NT(1e-9)) == -1);

        // Random walk feasibility.
        unsigned int violations = 0;
        for (unsigned int step = 0; step < 600; ++step) {
            walk.apply(OP, p, 1, rng);
            if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
        }
        CHECK(violations == 0);
        CHECK(walk.hard_negative_slack_count() == 0);
    }
}


// D. Distribution smoke test: with a diagonal A on an antichain (a box)
// the coordinates are independent truncated normals; compare sample means
// against 1D quadrature references.
template <typename NT>
void call_test_matched_distribution_means() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 43> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    MT A(MT::Zero(2, 2));
    A(0, 0) = NT(6);
    A(1, 1) = NT(3);
    VT center(2);
    center << NT(0.3), NT(0.7);

    typename MatchedPolicy::parameters<NT> params(A, center, NT(1));
    Point p(OP.inner_point());
    RNGType rng(2);
    MWalk walk(OP, p, params, rng);

    // 50000 samples put the 0.015 tolerance at ~5-6 batch-means standard
    // errors of this walk's autocorrelated chain: robust to RNG-stream
    // shifts from refactors, still sensitive to real distribution errors.
    unsigned int const N = 50000;
    NT sum0 = NT(0), sum1 = NT(0);
    for (unsigned int step = 0; step < N; ++step) {
        walk.apply(OP, p, 1, rng);
        sum0 += p[0];
        sum1 += p[1];
    }
    NT const ref0 = truncated_normal_mean(A(0, 0), center(0));
    NT const ref1 = truncated_normal_mean(A(1, 1), center(1));
    CHECK(std::abs(sum0 / NT(N) - ref0) < NT(0.015));
    CHECK(std::abs(sum1 / NT(N) - ref1) < NT(0.015));
    CHECK(walk.hard_negative_slack_count() == 0);
}


// E. Dense velocity coupling: a reflection on one wall changes *every*
// velocity coordinate through A^{-1} n.  The whole leg is recomputed
// independently here from the closed form, so a backend that only updated
// facets incident to the reflected coordinate would end up elsewhere.
template <typename NT>
void call_test_matched_dense_coupling() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 47> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    MT A(2, 2);
    A << NT(2), NT(-1.2), NT(-1.2), NT(2);
    VT center(VT::Constant(2, NT(0.5)));
    NT const omega = NT(1);

    typename MatchedPolicy::parameters<NT> params(A, center, omega);
    Point start(2, {NT(0.2), NT(0.5)});
    RNGType rng(2);
    MWalk walk(OP, start, params, rng);

    Point pw = start;
    Point vw(2, {NT(-0.6), NT(0.05)});
    NT const T = NT(0.9);
    CHECK(walk.apply_leg(OP, pw, vw, T));

    // Independent reference: alpha/beta closed form, first hit of the
    // lower wall of coordinate 0 by phase inversion, matched reflection
    // via an explicit Cholesky solve, then free flight to T.
    VT alpha(2), beta(2);
    alpha << start[0] - center(0), start[1] - center(1);
    beta << NT(-0.6) / omega, NT(0.05) / omega;

    // x0(theta) = c0 + R cos(theta - phi) = 0.
    NT const R = std::sqrt(alpha(0) * alpha(0) + beta(0) * beta(0));
    NT const phi = std::atan2(beta(0), alpha(0));
    NT const dacos = std::acos(-center(0) / R);
    NT theta_star = phi - dacos;
    if (!(theta_star > NT(1e-12)))
        theta_star = phi + dacos;
    REQUIRE(theta_star > NT(0));
    REQUIRE(theta_star < T * omega);

    NT const cs = std::cos(theta_star), sn = std::sin(theta_star);
    VT X = alpha * cs + beta * sn;
    VT Y = beta * cs - alpha * sn;
    REQUIRE(std::abs(center(0) + X(0)) < NT(1e-12));   // on the wall

    VT n(2);
    n << NT(-1), NT(0);
    VT const w = A.llt().solve(n);
    NT const q = n.dot(w);
    VT const Y_before = Y;
    Y -= (NT(2) * n.dot(Y) / q) * w;

    // The dense coupling is real: coordinate 1's velocity must move.
    CHECK(std::abs(Y(1) - Y_before(1)) > NT(0.1));

    VT alpha2 = X * cs - Y * sn;
    VT beta2  = X * sn + Y * cs;
    NT const ce = std::cos(T * omega), se = std::sin(T * omega);
    VT const x_end = center + alpha2 * ce + beta2 * se;
    VT const v_end = (beta2 * ce - alpha2 * se) * omega;

    // The reference trajectory must stay strictly inside after the hit,
    // so exactly one reflection happens in [0, T].
    for (NT th = theta_star + NT(1e-3); th < T * omega; th += NT(1e-3)) {
        VT const x = center + alpha2 * std::cos(th) + beta2 * std::sin(th);
        REQUIRE(x(0) > NT(0));
        REQUIRE(x(0) < NT(1));
        REQUIRE(x(1) > NT(0));
        REQUIRE(x(1) < NT(1));
    }

    CHECK(std::abs(pw[0] - x_end(0)) < NT(1e-10));
    CHECK(std::abs(pw[1] - x_end(1)) < NT(1e-10));
    CHECK(std::abs(vw[0] - v_end(0)) < NT(1e-10));
    CHECK(std::abs(vw[1] - v_end(1)) < NT(1e-10));
}



// Momentum-refresh covariance: v = omega L^{-T} xi must give
// cov(v) = omega^2 A^{-1}.  A non-diagonal A distinguishes the two
// Cholesky factors (solving with L instead of L^T gives (L^T L)^{-1},
// off by ~0.3 entrywise here), so this pins the factor orientation the
// rest of the suite cannot see.
template <typename NT>
void call_test_matched_refresh_covariance() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 53> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    MT A(2, 2);
    A << NT(2), NT(-1.2), NT(-1.2), NT(2);
    NT const omega = NT(1.7);
    typename MatchedPolicy::parameters<NT> params(
        A, VT(VT::Constant(2, NT(0.5))), omega);
    Point start(2, {NT(0.5), NT(0.5)});
    RNGType rng(2);
    MWalk walk(OP, start, params, rng);

    unsigned int const N = 20000;
    MT cov(MT::Zero(2, 2));
    NT mean_K = NT(0);
    for (unsigned int k = 0; k < N; ++k) {
        Point v = walk.sample_velocity(rng);
        VT const vc = v.getCoefficients();
        cov += vc * vc.transpose();
        mean_K += walk.kinetic(v);
    }
    cov /= NT(N);
    mean_K /= NT(N);

    MT const cov_ref(omega * omega * A.llt().solve(MT::Identity(2, 2)));
    for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
            CHECK(std::abs(cov(i, j) - cov_ref(i, j)) < NT(0.05));

    // E[K] = n/2 for v ~ N(0, M^{-1}) with K = 1/2 v^T M v.
    CHECK(std::abs(mean_K - NT(1)) < NT(0.03));
}


// Cover-facet reflection with a non-constant center and an A whose
// A^{-1}(e_u - e_v) is NOT parallel to the normal: pins the center term
// of cover offsets (C = b - rel_scale (c_u - c_v)) and the genuinely
// coupled cover reflection, both invisible to the other fixtures.  The
// normalized polytope run must land on the same end state: the cover
// crossing x_u = x_v and the unnormalized reflection normal are both
// invariant to row scaling.
template <typename NT>
void call_test_matched_cover_center_reflection() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 59> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    MT A(2, 2);
    A << NT(2), NT(-0.5), NT(-0.5), NT(1);
    VT center(2);
    center << NT(0.2), NT(0.7);
    NT const omega = NT(1);
    NT const T = NT(1.2);

    // Independent closed-form reference: hit of x0 - x1 = 0 by phase
    // inversion, matched reflection by an explicit Cholesky solve, free
    // flight to T.
    VT alpha(2), beta(2);
    alpha << NT(0.15) - center(0), NT(0.55) - center(1);
    beta << NT(0.5), NT(-0.3);

    NT const g_a = alpha(0) - alpha(1);            // cos coefficient
    NT const g_b = beta(0) - beta(1);              // sin coefficient
    NT const g_c = -(center(0) - center(1));       // rhs of the crossing
    NT const R = std::sqrt(g_a * g_a + g_b * g_b);
    NT const phi = std::atan2(g_b, g_a);
    NT const dacos = std::acos(g_c / R);
    NT theta_star = phi - dacos;
    if (!(theta_star > NT(1e-12)))
        theta_star = phi + dacos;
    REQUIRE(theta_star > NT(0));
    REQUIRE(theta_star < T * omega);
    // outgoing: d(x0 - x1)/dtheta > 0 at the hit
    REQUIRE(-g_a * std::sin(theta_star) + g_b * std::cos(theta_star) > NT(0));

    NT const cs = std::cos(theta_star), sn = std::sin(theta_star);
    VT X = alpha * cs + beta * sn;
    VT Y = beta * cs - alpha * sn;
    REQUIRE(std::abs((center(0) + X(0)) - (center(1) + X(1))) < NT(1e-12));

    VT n(2);
    n << NT(1), NT(-1);
    VT const w = A.llt().solve(n);
    VT const Y_before = Y;
    Y -= (NT(2) * n.dot(Y) / n.dot(w)) * w;
    // The coupling is genuine: the tangential (mean) component moves too.
    CHECK(std::abs((Y(0) + Y(1)) - (Y_before(0) + Y_before(1))) > NT(0.05));

    VT const alpha2 = X * cs - Y * sn;
    VT const beta2  = X * sn + Y * cs;
    NT const ce = std::cos(T * omega), se = std::sin(T * omega);
    VT const x_end = center + alpha2 * ce + beta2 * se;
    VT const v_end = (beta2 * ce - alpha2 * se) * omega;

    // Exactly one reflection in [0, T]: the post-hit path stays inside.
    for (NT th = theta_star + NT(1e-3); th < T * omega; th += NT(1e-3)) {
        VT const x = center + alpha2 * std::cos(th) + beta2 * std::sin(th);
        REQUIRE(x(0) > NT(0));
        REQUIRE(x(1) < NT(1));
        REQUIRE(x(0) < x(1));
    }

    RV rels{{0, 1}};
    for (bool normalized : {false, true}) {
        Poset poset(2, rels);
        OP_t OP(poset);
        if (normalized) OP.normalize();

        typename MatchedPolicy::parameters<NT> params(A, center, omega);
        Point start(2, {NT(0.15), NT(0.55)});
        RNGType rng(2);
        MWalk walk(OP, start, params, rng);

        Point pw = start;
        Point vw(2, {NT(0.5), NT(-0.3)});
        CHECK(walk.apply_leg(OP, pw, vw, T));
        for (unsigned int i = 0; i < 2; ++i) {
            CHECK(std::abs(pw[i] - x_end(i)) < NT(1e-10));
            CHECK(std::abs(vw[i] - v_end(i)) < NT(1e-10));
        }
        CHECK(OP.is_in(pw, NT(1e-9)) == -1);
        CHECK(walk.hard_negative_slack_count() == 0);
    }
}


// Reflection-budget abort: apply_leg must return false and leave the
// caller's (p, v) untouched, and the walk must remain usable afterwards.
template <typename NT>
void call_test_matched_abort_restore() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 61> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    typename MatchedPolicy::parameters<NT> params(
        MT(MT::Identity(2, 2)), VT(VT::Zero(2)), NT(1));
    params.rho = 1;
    params.set_rho = true;

    Point start(2, {NT(0.5), NT(0.5)});
    RNGType rng(2);
    MWalk walk(OP, start, params, rng);

    // Two simultaneous lower-facet reflections exceed rho = 1.
    Point p = start;
    Point v(2, {NT(-0.375), NT(-0.375)});
    NT const T = NT(1.57079632679489661923);
    CHECK_FALSE(walk.apply_leg(OP, p, v, T));
    CHECK(p[0] == NT(0.5));
    CHECK(p[1] == NT(0.5));
    CHECK(v[0] == NT(-0.375));
    CHECK(v[1] == NT(-0.375));

    // A reflection-free leg still works on the same walk object.
    Point p2 = start;
    Point v2(2, {NT(0.1), NT(-0.1)});
    CHECK(walk.apply_leg(OP, p2, v2, NT(0.4)));
    CHECK(OP.is_in(p2, NT(1e-12)) == -1);
}


// Solver stress regimes the basic suite does not reach: a Gaussian
// center far outside the polytope (every leg reflects around an exterior
// orbit center; facet offsets change sign regimes) and an annealing-scale
// omega (angle_eps grows to ~3e-8 while chunks stay quarter-period).
template <typename NT>
void call_test_matched_stress_regimes() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 67> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    // Exterior center, dense A.
    {
        RV rels;
        Poset poset(2, rels);
        OP_t OP(poset);

        MT A(2, 2);
        A << NT(2), NT(-0.5), NT(-0.5), NT(1);
        VT center(2);
        center << NT(1.5), NT(-0.4);
        typename MatchedPolicy::parameters<NT> params(A, center, NT(1));

        Point p(OP.inner_point());
        RNGType rng(2);
        MWalk walk(OP, p, params, rng);

        Point pl = p;
        Point vl(2, {NT(0.4), NT(0.6)});
        NT const H0 = walk.hamiltonian(pl, vl);
        CHECK(walk.apply_leg(OP, pl, vl, NT(1.4)));
        NT const H1 = walk.hamiltonian(pl, vl);
        CHECK(std::abs(H1 - H0) < NT(1e-9) * std::max(NT(1), std::abs(H0)));

        unsigned int violations = 0;
        for (unsigned int step = 0; step < 400; ++step) {
            walk.apply(OP, p, 1, rng);
            if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
        }
        CHECK(violations == 0);
        CHECK(walk.hard_negative_slack_count() == 0);
    }

    // Annealing-scale omega on a chain: the stationary mass concentrates
    // at sigma ~ 1/omega around a center pressed against the corner.
    {
        RV rels{{0, 1}, {1, 2}};
        Poset poset(3, rels);
        OP_t OP(poset);

        NT const omega = NT(300);
        MT A(3, 3);
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                A(i, j) = omega * omega
                        * NT(2) * std::pow(NT(0.5), std::abs(int(i) - int(j)));
        VT center(3);
        center << NT(0.02), NT(0.03), NT(0.05);
        typename MatchedPolicy::parameters<NT> params(A, center, omega);

        Point p(OP.inner_point());
        RNGType rng(3);
        MWalk walk(OP, p, params, rng);

        unsigned int violations = 0;
        for (unsigned int step = 0; step < 400; ++step) {
            walk.apply(OP, p, 1, rng);
            if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
        }
        CHECK(violations == 0);
        CHECK(walk.hard_negative_slack_count() == 0);
    }
}



// ---- PR-2: MatchedDiagonalEventQueue ---------------------------------

// A. Weighted diagonal reflection formula on every facet type, plus
// backend routing and the conservative diagonal detector.
template <typename NT>
void call_test_matched_diag_weighted_reflection() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 71> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    RV rels{{0, 1}};
    Poset poset(2, rels);
    OP_t OP(poset);

    VT a_diag(2);
    a_diag << NT(4), NT(1);
    VT center(2);
    center << NT(0.2), NT(0.7);
    NT const omega = NT(1.3);

    typename MatchedPolicy::parameters<NT> params(a_diag, center, omega);
    Point start(2, {NT(0.15), NT(0.55)});
    RNGType rng(2);
    MWalk walk(OP, start, params, rng);
    CHECK(walk.backend() == GaussianHMCBackend::MatchedDiagonalEventQueue);
    REQUIRE(walk.num_event_facets() == 3);   // lb(0), ub(1), cover

    // Cover facet (fid 2), a_u != a_v:
    //   den = 1/4 + 1 = 1.25,  s = (v_u - v_v)/den = 1.2/1.25 = 0.96,
    //   v_u+ = 0.9 - 2*0.96/4 = 0.42,   v_v+ = -0.3 + 2*0.96/1 = 1.62.
    {
        VT v(2);
        v << NT(0.9), NT(-0.3);
        VT v1 = v;
        walk.reflect_velocity(2, v1);
        CHECK(std::abs(v1(0) - NT(0.42)) < NT(1e-13));
        CHECK(std::abs(v1(1) - NT(1.62)) < NT(1e-13));
        // n^T v flips; kinetic form v^T A v is preserved; involution.
        CHECK(std::abs((v1(0) - v1(1)) + (v(0) - v(1))) < NT(1e-13));
        NT const K0 = NT(4) * v(0) * v(0) + v(1) * v(1);
        NT const K1 = NT(4) * v1(0) * v1(0) + v1(1) * v1(1);
        CHECK(std::abs(K1 - K0) < NT(1e-12) * K0);
        VT v2 = v1;
        walk.reflect_velocity(2, v2);
        CHECK(std::abs(v2(0) - v(0)) < NT(1e-13));
        CHECK(std::abs(v2(1) - v(1)) < NT(1e-13));
    }
    // Wall facets flip exactly one coordinate.
    {
        VT v(2);
        v << NT(0.9), NT(-0.3);
        VT v1 = v;
        walk.reflect_velocity(0, v1);   // lb(0)
        CHECK(v1(0) == NT(-0.9));
        CHECK(v1(1) == NT(-0.3));
        VT v3 = v;
        walk.reflect_velocity(1, v3);   // ub(1)
        CHECK(v3(0) == NT(0.9));
        CHECK(v3(1) == NT(0.3));
    }

    // The diagonal reflection agrees with the dense matched backend for
    // the same (diagonal) precision.
    {
        typename MatchedPolicy::parameters<NT> pdense(
            MT(a_diag.asDiagonal()), center, omega,
            GaussianHMCBackend::MatchedDenseAngleFullScan);
        RNGType rng_d(2);
        MWalk dense(OP, start, pdense, rng_d);
        CHECK(dense.backend() == GaussianHMCBackend::MatchedDenseAngleFullScan);
        for (unsigned int fid = 0; fid < 3; ++fid) {
            VT va(2), vb(2);
            va << NT(0.7), NT(-0.4);
            vb = va;
            walk.reflect_velocity(fid, va);
            dense.reflect_velocity(fid, vb);
            CHECK(std::abs(va(0) - vb(0)) < NT(1e-13));
            CHECK(std::abs(va(1) - vb(1)) < NT(1e-13));
        }
    }

    // Routing: an exactly-diagonal dense matrix is detected; a nearly
    // diagonal one is NOT, and forcing the diagonal queue on it throws.
    {
        MT D(MT::Zero(2, 2));
        D(0, 0) = NT(4); D(1, 1) = NT(1);
        typename MatchedPolicy::parameters<NT> pd(D, center, omega);
        RNGType r1(2);
        MWalk w1(OP, start, pd, r1);
        CHECK(w1.backend() == GaussianHMCBackend::MatchedDiagonalEventQueue);

        MT Dn = D;
        Dn(0, 1) = Dn(1, 0) = NT(1e-3);
        typename MatchedPolicy::parameters<NT> pn(Dn, center, omega);
        RNGType r2(2);
        MWalk w2(OP, start, pn, r2);
        CHECK(w2.backend() == GaussianHMCBackend::MatchedDenseAngleFullScan);

        typename MatchedPolicy::parameters<NT> pf(
            Dn, center, omega, GaussianHMCBackend::MatchedDiagonalEventQueue);
        RNGType r3(2);
        CHECK_THROWS_AS(MWalk(OP, start, pf, r3), std::invalid_argument);
    }

    // Detector hardening: malformed and extreme inputs must never be
    // classified diagonal (they fall through to the dense branch's loud
    // validation), and huge diagonals must not overflow the bound into
    // waving a strongly coupled matrix through.
    {
        CHECK(MatchedPolicy::is_diagonal_matrix<MT, NT>(
            MT(MT::Identity(3, 3))));

        MT nonsquare(MT::Zero(3, 2));
        nonsquare(0, 0) = NT(2); nonsquare(1, 1) = NT(3);
        CHECK_FALSE(MatchedPolicy::is_diagonal_matrix<MT, NT>(nonsquare));
        typename MatchedPolicy::parameters<NT> pns(nonsquare, center, omega);
        RNGType r4(2);
        CHECK_THROWS_AS(MWalk(OP, start, pns, r4), std::invalid_argument);

        MT huge(2, 2);
        huge << NT(1e160), NT(5e159), NT(5e159), NT(1e160);
        CHECK_FALSE(MatchedPolicy::is_diagonal_matrix<MT, NT>(huge));

        MT withnan(MT::Identity(2, 2));
        withnan(0, 1) = std::numeric_limits<NT>::quiet_NaN();
        CHECK_FALSE(MatchedPolicy::is_diagonal_matrix<MT, NT>(withnan));
    }
}


// B. Reduction to spherical: for A = 2a I (uniform diagonal), c = 0 and
// omega = sqrt(2a), the diagonal event queue must reproduce the spherical
// walk on the PR-0 exact corner fixtures.
template <typename NT>
void call_test_matched_diag_reduction_to_spherical() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 73> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;
    typedef typename SphericalPolicy::template Walk<OP_t, RNGType> SWalk;

    NT const a = NT(0.5);            // omega = 1, A = I
    NT const T = NT(1.57079632679489661923);

    // Antichain fixture: both lower facets simultaneous, end at 0.375.
    {
        RV rels;
        Poset poset(2, rels);
        OP_t OP(poset);
        Point start(2, {NT(0.5), NT(0.5)});

        typename MatchedPolicy::parameters<NT> params(
            VT(VT::Constant(2, NT(1))), VT(VT::Zero(2)), NT(1));
        RNGType rng(2);
        MWalk walk(OP, start, params, rng);
        CHECK(walk.backend() == GaussianHMCBackend::MatchedDiagonalEventQueue);

        Point p = start, v(2, {NT(-0.375), NT(-0.375)});
        CHECK(walk.apply_leg(OP, p, v, T));
        CHECK(std::abs(p[0] - NT(0.375)) < NT(1e-9));
        CHECK(std::abs(p[1] - NT(0.375)) < NT(1e-9));
    }

    // Chain fixture: cover + bound simultaneous, weighted coefficients
    // r_u = r_v = 1 reduce to the swap; compare against the spherical
    // walk and the exact end state.
    {
        RV rels{{0, 1}, {1, 2}};
        Poset poset(3, rels);
        OP_t OP(poset);
        Point start(3, {NT(0.25), NT(0.375), NT(0.5)});

        typename MatchedPolicy::parameters<NT> params(
            VT(VT::Constant(3, NT(1))), VT(VT::Zero(3)), NT(1));
        RNGType rng(3);
        MWalk walk(OP, start, params, rng);

        RNGType rng_s(3);
        SWalk swalk(OP, start, a, rng_s);

        Point pm = start, vm(3, {NT(0.34375), NT(0.25), NT(0.875)});
        Point ps = start, vs0(3, {NT(0.34375), NT(0.25), NT(0.875)});
        CHECK(walk.apply_leg(OP, pm, vm, T));
        CHECK(swalk.apply_leg(OP, ps, vs0, T));
        Point vs = swalk.velocity();
        for (unsigned int i = 0; i < 3; ++i) {
            CHECK(std::abs(pm[i] - ps[i]) < NT(1e-12));
            CHECK(std::abs(vm[i] - vs[i]) < NT(1e-12));
        }
        CHECK(std::abs(pm[0] - NT(0.25)) < NT(1e-9));
        CHECK(std::abs(pm[1] - NT(0.34375)) < NT(1e-9));
        CHECK(std::abs(pm[2] - NT(0.725)) < NT(1e-9));
    }
}


// C. Diagonal event queue vs dense full scan: identical deterministic
// legs (nonuniform diagonal, non-constant center, multi-chunk) must land
// on the same end states across poset families.
template <typename NT>
void call_test_matched_diag_vs_dense() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 79> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    std::vector<std::pair<unsigned int, RV>> posets;
    {
        RV chain;
        for (unsigned int i = 0; i + 1 < 6; ++i) chain.push_back({i, i + 1});
        posets.push_back({6, chain});
    }
    posets.push_back({4, RV{}});
    posets.push_back({4, RV{{0, 1}, {0, 2}, {1, 3}, {2, 3}}});
    posets.push_back({8, RV{{0, 2}, {1, 2}, {2, 5}, {3, 5}, {1, 4},
                            {4, 6}, {5, 7}}});

    NT const omega = NT(1.2);
    std::vector<NT> const Ts = {NT(0.7), NT(1.3), NT(2.1), NT(0.9)};

    for (auto& pr : posets) {
        unsigned int const n = pr.first;
        Poset poset(n, pr.second);
        OP_t OP(poset);

        VT a_diag(n), center(n);
        for (unsigned int i = 0; i < n; ++i) {
            a_diag(i) = NT(1) + NT(0.6) * NT(i % 3);
            center(i) = NT(0.35) + NT(0.05) * NT(i % 2);
        }

        typename MatchedPolicy::parameters<NT> pq(a_diag, center, omega);
        typename MatchedPolicy::parameters<NT> pd(
            MT(a_diag.asDiagonal()), center, omega,
            GaussianHMCBackend::MatchedDenseAngleFullScan);

        Point start(OP.inner_point());
        RNGType rng_q(n), rng_d(n);
        MWalk wq(OP, start, pq, rng_q);
        MWalk wd(OP, start, pd, rng_d);
        CHECK(wq.backend() == GaussianHMCBackend::MatchedDiagonalEventQueue);
        CHECK(wd.backend() == GaussianHMCBackend::MatchedDenseAngleFullScan);

        Point pq_p = start, pd_p = start;
        for (unsigned int leg = 0; leg < Ts.size(); ++leg) {
            Point vq(n), vd(n);
            for (unsigned int i = 0; i < n; ++i) {
                NT const vi = NT(0.9) - NT(0.5) * NT((i + leg) % 3)
                            + NT(0.15) * NT(leg);
                vq.set_coord(i, vi);
                vd.set_coord(i, vi);
            }
            CHECK(wq.apply_leg(OP, pq_p, vq, Ts[leg]));
            CHECK(wd.apply_leg(OP, pd_p, vd, Ts[leg]));
            for (unsigned int i = 0; i < n; ++i) {
                CHECK(std::abs(pq_p[i] - pd_p[i]) < NT(1e-9));
                CHECK(std::abs(vq[i] - vd[i]) < NT(1e-9));
            }
            CHECK(OP.is_in(pq_p, NT(1e-9)) == -1);
        }
        CHECK(wq.hard_negative_slack_count() == 0);
        CHECK(wd.hard_negative_slack_count() == 0);
    }
}


#ifdef VOLESTI_HMC_PROFILE
// D. Local invalidation: a reflection recomputes only the facets incident
// to the affected coordinate(s), not all facets.  Chain-12 has 13 true
// facets; a wall hit touches coordinate {0} (2 incident facets), a cover
// hit touches {0, 1} (3 incident facets).  Centering the Gaussian on the
// start point freezes every coordinate with zero velocity, so the legs
// are exactly one-reflection by construction.
template <typename NT>
void call_test_matched_diag_local_invalidation() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 83> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    unsigned int const n = 12;
    RV chain;
    for (unsigned int i = 0; i + 1 < n; ++i) chain.push_back({i, i + 1});
    Poset poset(n, chain);
    OP_t OP(poset);
    unsigned int const F = OP.num_true_facets();
    REQUIRE(F == 13);

    Point start(OP.inner_point());
    VT center(n);
    for (unsigned int i = 0; i < n; ++i) center(i) = start[i];
    VT a_diag(VT::Constant(n, NT(1)));
    a_diag(1) = NT(4);

    typename MatchedPolicy::parameters<NT> params(a_diag, center, NT(1));
    RNGType rng(n);
    MWalk walk(OP, start, params, rng);
    CHECK(walk.backend() == GaussianHMCBackend::MatchedDiagonalEventQueue);

    NT const T = NT(0.8);
    auto run_leg = [&](std::vector<NT> const& vcoords,
                       unsigned long long& solves,
                       unsigned long long& reflections) {
        Point p = start;
        Point v(n);
        for (unsigned int i = 0; i < n; ++i) v.set_coord(i, vcoords[i]);
        unsigned long long const c0 = hmc_profile_counters::n_trig_calls.load();
        unsigned long long const r0 = hmc_profile_counters::n_reflections.load();
        CHECK(walk.apply_leg(OP, p, v, T));
        CHECK(OP.is_in(p, NT(1e-9)) == -1);
        solves = hmc_profile_counters::n_trig_calls.load() - c0;
        reflections = hmc_profile_counters::n_reflections.load() - r0;
    };

    std::vector<NT> v_none(n, NT(0));
    std::vector<NT> v_wall(n, NT(0));
    v_wall[0] = NT(-0.2);              // hits lb(0) once
    std::vector<NT> v_cover(n, NT(0));
    v_cover[0] = NT(0.06);             // hits cover(0,1) once
    v_cover[1] = NT(-0.06);

    unsigned long long s_none, r_none, s_wall, r_wall, s_cover, r_cover;
    run_leg(v_none, s_none, r_none);
    run_leg(v_wall, s_wall, r_wall);
    run_leg(v_cover, s_cover, r_cover);

    CHECK(r_none == 0);
    CHECK(r_wall == 1);
    CHECK(r_cover == 1);

    // The wall hit re-solves the 2 facets incident to coordinate 0, the
    // cover hit the 3 facets incident to {0, 1} - plus a bounded handful
    // of pop-time re-solves - never the full 13-facet rescan the dense
    // backend would do.
    unsigned long long const extra_wall = s_wall - s_none;
    unsigned long long const extra_cover = s_cover - s_none;
    CHECK(extra_wall >= 1);
    CHECK(extra_wall <= 5);
    CHECK(extra_cover >= 1);
    CHECK(extra_cover <= 7);
    CHECK(extra_wall < F);
    CHECK(extra_cover < F);
}
#endif


// E. Long-run feasibility with a strongly nonuniform diagonal precision
// and an off-center Gaussian on chain and sparse posets.
template <typename NT>
void call_test_matched_diag_long_run() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 89> RNGType;
    typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;

    std::vector<std::pair<unsigned int, RV>> posets;
    {
        RV chain;
        for (unsigned int i = 0; i + 1 < 16; ++i) chain.push_back({i, i + 1});
        posets.push_back({16, chain});
    }
    posets.push_back({8, RV{{0, 2}, {1, 2}, {2, 5}, {3, 5}, {1, 4},
                            {4, 6}, {5, 7}}});

    for (auto& pr : posets) {
        unsigned int const n = pr.first;
        Poset poset(n, pr.second);
        OP_t OP(poset);

        VT a_diag(n), center(n);
        for (unsigned int i = 0; i < n; ++i) {
            a_diag(i) = (i % 2 == 0) ? NT(0.5) : NT(8);
            center(i) = NT(0.1) + NT(0.8) * NT(i) / NT(n);
        }
        typename MatchedPolicy::parameters<NT> params(a_diag, center, NT(1.5));

        Point p(OP.inner_point());
        RNGType rng(n);
        MWalk walk(OP, p, params, rng);

        unsigned int violations = 0;
        for (unsigned int step = 0; step < 1200; ++step) {
            walk.apply(OP, p, 1, rng);
            if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
        }
        CHECK(violations == 0);
        CHECK(walk.hard_negative_slack_count() == 0);
    }
}


TEST_CASE("matched_spherical_equivalence") {
    call_test_matched_spherical_equivalence<double>();
}

TEST_CASE("matched_identity_delegation") {
    call_test_matched_identity_delegation<double>();
}

TEST_CASE("matched_reflection_invariants") {
    call_test_matched_reflection_invariants<double>();
}

TEST_CASE("matched_feasibility") {
    call_test_matched_feasibility<double>();
}

TEST_CASE("matched_distribution_means") {
    call_test_matched_distribution_means<double>();
}

TEST_CASE("matched_dense_coupling") {
    call_test_matched_dense_coupling<double>();
}

TEST_CASE("matched_refresh_covariance") {
    call_test_matched_refresh_covariance<double>();
}

TEST_CASE("matched_cover_center_reflection") {
    call_test_matched_cover_center_reflection<double>();
}

TEST_CASE("matched_abort_restore") {
    call_test_matched_abort_restore<double>();
}

TEST_CASE("matched_stress_regimes") {
    call_test_matched_stress_regimes<double>();
}

TEST_CASE("matched_diag_weighted_reflection") {
    call_test_matched_diag_weighted_reflection<double>();
}

TEST_CASE("matched_diag_reduction_to_spherical") {
    call_test_matched_diag_reduction_to_spherical<double>();
}

TEST_CASE("matched_diag_vs_dense") {
    call_test_matched_diag_vs_dense<double>();
}

#ifdef VOLESTI_HMC_PROFILE
TEST_CASE("matched_diag_local_invalidation") {
    call_test_matched_diag_local_invalidation<double>();
}
#endif

TEST_CASE("matched_diag_long_run") {
    call_test_matched_diag_long_run<double>();
}
