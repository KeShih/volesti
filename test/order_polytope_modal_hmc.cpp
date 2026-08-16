// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include <cmath>
#include <iostream>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"

#include "random_walks/random_walks.hpp"
#include "random_walks/order_polytope_hmc_modal.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef OrderPolytope<Point> OP_t;
typedef typename OP_t::MT MT;
typedef typename OP_t::VT VT;
typedef OrderPolytopeModalGaussianTrajectory<OP_t> Modal;

// Deterministic small SPD matrix: diag-dominant with fixed off-diagonals.
static MT test_spd(unsigned int n, NT diag, NT coup)
{
    MT A(n, n);
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            A(i, j) = (i == j) ? diag + NT(0.3) * NT(i)
                               : coup / NT(1 + std::abs(int(i) - int(j)));
    return NT(0.5) * (A + MT(A.transpose()));
}

TEST_CASE("modal_spherical_reduction") {
    // A = 2a I, M = I: every mode oscillates at omega = sqrt(2a) and the
    // trajectory reduces to the spherical closed form.
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);

    NT const a = NT(1.25);
    NT const omega = std::sqrt(NT(2) * a);
    Modal traj(OP, MT(NT(2) * a * MT::Identity(3, 3)), MT(MT::Identity(3, 3)),
               VT(VT::Zero(3)));

    for (unsigned int k = 0; k < 3; ++k)
        CHECK(std::abs(traj.frequencies()(k) - omega) < NT(1e-12));

    VT x(3), v(3);
    x << NT(0.3), NT(0.45), NT(0.7);
    v << NT(0.9), NT(-0.6), NT(0.2);
    traj.set_state(x, v);

    for (NT t : {NT(0.0), NT(0.3), NT(0.8), NT(1.7), NT(2.9)}) {
        VT const ref = x * std::cos(omega * t)
                     + v * (std::sin(omega * t) / omega);
        VT const pos = traj.position(t);
        VT const vel = traj.velocity(t);
        VT const vref = v * std::cos(omega * t)
                      - x * (omega * std::sin(omega * t));
        for (unsigned int i = 0; i < 3; ++i) {
            CHECK(std::abs(pos(i) - ref(i)) < NT(1e-12));
            CHECK(std::abs(vel(i) - vref(i)) < NT(1e-12));
            CHECK(std::abs(traj.coordinate(i, t) - ref(i)) < NT(1e-12));
            CHECK(std::abs(traj.velocity_coordinate(i, t) - vref(i)) < NT(1e-12));
        }
    }
}

TEST_CASE("modal_matched_reduction") {
    // Matched mass M = A / omega^2: all generalized frequencies equal
    // omega and the trajectory reduces to the common-frequency closed
    // form c + (x - c) cos(omega t) + v/omega sin(omega t).
    Poset::RV rels;
    Poset poset(4, rels);
    OP_t OP(poset);

    MT const A = test_spd(4, NT(2), NT(0.5));
    NT const omega = NT(1.7);
    VT center(4);
    center << NT(0.4), NT(0.5), NT(0.6), NT(0.45);
    Modal traj(OP, A, MT(A / (omega * omega)), center);

    for (unsigned int k = 0; k < 4; ++k)
        CHECK(std::abs(traj.frequencies()(k) - omega) < NT(1e-9));

    VT x(4), v(4);
    x << NT(0.2), NT(0.7), NT(0.5), NT(0.35);
    v << NT(0.6), NT(-0.8), NT(0.1), NT(0.4);
    traj.set_state(x, v);

    for (NT t : {NT(0.2), NT(0.9), NT(1.6)}) {
        VT const ref = center + (x - center) * std::cos(omega * t)
                     + v * (std::sin(omega * t) / omega);
        VT const pos = traj.position(t);
        for (unsigned int i = 0; i < 4; ++i)
            CHECK(std::abs(pos(i) - ref(i)) < NT(1e-9));
    }
}

TEST_CASE("modal_energy_conservation") {
    // Arbitrary SPD A and M (not matched): H = U + K constant along the
    // exact flow.
    Poset::RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    Poset poset(4, rels);
    OP_t OP(poset);

    MT const A = test_spd(4, NT(3), NT(0.8));
    MT const M = test_spd(4, NT(1.5), NT(-0.4));
    VT center(VT::Constant(4, NT(0.5)));
    Modal traj(OP, A, M, center);

    VT x(4), v(4);
    x << NT(0.3), NT(0.55), NT(0.4), NT(0.75);
    v << NT(1.1), NT(-0.5), NT(0.7), NT(-0.9);
    traj.set_state(x, v);

    NT const H0 = traj.energy(x, v);
    for (NT t = NT(0); t <= NT(3); t += NT(0.11)) {
        NT const H = traj.energy(traj.position(t), traj.velocity(t));
        CHECK(std::abs(H - H0) < NT(1e-10) * std::max(NT(1), std::abs(H0)));
    }
}

TEST_CASE("modal_finite_difference") {
    // d position / dt == velocity, checked by central differences.
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);

    MT const A = test_spd(3, NT(2.5), NT(0.6));
    MT const M = test_spd(3, NT(1.2), NT(0.3));
    Modal traj(OP, A, M, VT(VT::Constant(3, NT(0.4))));

    VT x(3), v(3);
    x << NT(0.25), NT(0.5), NT(0.8);
    v << NT(-0.7), NT(0.4), NT(0.9);
    traj.set_state(x, v);

    NT const h = NT(1e-6);
    for (NT t : {NT(0.15), NT(0.7), NT(1.9)}) {
        VT const fd = (traj.position(t + h) - traj.position(t - h)) / (NT(2) * h);
        VT const vel = traj.velocity(t);
        for (unsigned int i = 0; i < 3; ++i)
            CHECK(std::abs(fd(i) - vel(i)) < NT(1e-6));
        for (unsigned int f = 0; f < traj.num_facets(); ++f) {
            NT const gfd = (traj.facet_value(f, t + h)
                            - traj.facet_value(f, t - h)) / (NT(2) * h);
            CHECK(std::abs(gfd - traj.facet_derivative(f, t)) < NT(1e-6));
        }
    }
}

TEST_CASE("modal_facet_values") {
    // facet_value must equal the direct coordinate formulas for every
    // true facet type (lower wall, upper wall, cover).
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);       // facets: lb(0), ub(2), cover(0,1), cover(1,2)

    MT const A = test_spd(3, NT(2), NT(0.5));
    MT const M = test_spd(3, NT(1), NT(0.2));
    Modal traj(OP, A, M, VT(VT::Constant(3, NT(0.45))));
    REQUIRE(traj.num_facets() == 4);

    VT x(3), v(3);
    x << NT(0.2), NT(0.5), NT(0.75);
    v << NT(0.8), NT(-0.3), NT(0.5);
    traj.set_state(x, v);

    for (NT t : {NT(0.0), NT(0.4), NT(1.3)}) {
        NT const x0 = traj.coordinate(0, t);
        NT const x1 = traj.coordinate(1, t);
        NT const x2 = traj.coordinate(2, t);
        CHECK(std::abs(traj.facet_value(0, t) - (-x0)) < NT(1e-12));       // lb(0)
        CHECK(std::abs(traj.facet_value(1, t) - (x2 - NT(1))) < NT(1e-12)); // ub(2)
        CHECK(std::abs(traj.facet_value(2, t) - (x0 - x1)) < NT(1e-12));   // cover
        CHECK(std::abs(traj.facet_value(3, t) - (x1 - x2)) < NT(1e-12));   // cover
    }

    // Rejections: non-SPD and non-symmetric inputs fail loudly.
    MT bad = A;
    bad(0, 1) += NT(0.2);
    CHECK_THROWS_AS(Modal(OP, bad, M, VT(VT::Zero(3))), std::invalid_argument);
    MT negA(MT::Identity(3, 3));
    negA(2, 2) = NT(-1);
    CHECK_THROWS_AS(Modal(OP, negA, M, VT(VT::Zero(3))), std::invalid_argument);
}


// ---- Part 2: dense modal boundary oracle ------------------------------

TEST_CASE("modal_oracle_spherical_agreement") {
    // A = 2a I, M = I, c = 0: the modal oracle must agree with the
    // independent full-scan trigonometric oracle of OrderPolytope.
    NT const a = NT(1.25);
    NT const omega = std::sqrt(NT(2) * a);

    struct Case { unsigned int n; Poset::RV rels; std::vector<NT> x, v; };
    std::vector<Case> cases = {
        {3, {{0, 1}, {1, 2}}, {NT(0.2), NT(0.5), NT(0.8)},
         {NT(0.9), NT(-0.4), NT(0.3)}},
        {2, {}, {NT(0.4), NT(0.6)}, {NT(-0.8), NT(0.7)}},
        {4, {{0, 1}, {0, 2}, {1, 3}, {2, 3}},
         {NT(0.2), NT(0.45), NT(0.55), NT(0.8)},
         {NT(0.5), NT(-0.6), NT(0.8), NT(-0.3)}},
    };

    for (auto const& tc : cases) {
        Poset::RV rels = tc.rels;
        Poset poset(tc.n, rels);
        OP_t OP(poset);

        Modal traj(OP, MT(NT(2) * a * MT::Identity(tc.n, tc.n)),
                   MT(MT::Identity(tc.n, tc.n)), VT(VT::Zero(tc.n)));
        VT x(tc.n), v(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i) { x(i) = tc.x[i]; v(i) = tc.v[i]; }
        traj.set_state(x, v);

        auto const hit = first_hit_modal_dense(traj, NT(10), -1);
        REQUIRE(hit.facet >= 0);

        Point px(tc.n), pv(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i) {
            px.set_coord(i, x(i));
            pv.set_coord(i, v(i));
        }
        int prev = -1;
        auto const full = OP.trigonometric_positive_intersect(px, pv, omega, prev);
        CHECK(std::abs(hit.t - full.first) < NT(1e-9));
        CHECK(std::abs(traj.facet_value(hit.facet, hit.t)) < NT(1e-8));
    }
}

TEST_CASE("modal_oracle_matched_agreement") {
    // Matched mass: the facet values are single-frequency, so an
    // independent closed-form scan (built directly from n.alpha, n.beta)
    // must agree with the modal oracle.
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);

    MT const A = test_spd(3, NT(2), NT(0.5));
    NT const omega = NT(1.4);
    VT center(3);
    center << NT(0.3), NT(0.5), NT(0.65);
    Modal traj(OP, A, MT(A / (omega * omega)), center);

    VT x(3), v(3);
    x << NT(0.25), NT(0.45), NT(0.7);
    v << NT(0.8), NT(-0.5), NT(0.6);
    traj.set_state(x, v);

    auto const hit = first_hit_modal_dense(traj, NT(6), -1);
    REQUIRE(hit.facet >= 0);

    // Closed-form facet values on the common-frequency trajectory:
    // rows lb(0), ub(2), cover(0,1), cover(1,2).
    VT const alpha = x - center;
    VT const beta = v / omega;
    auto gref = [&](unsigned int f, NT t) {
        NT af, bf, cf;
        switch (f) {
            case 0: af = -alpha(0); bf = -beta(0); cf = NT(0) + center(0); break;
            case 1: af = alpha(2); bf = beta(2); cf = NT(1) - center(2); break;
            case 2: af = alpha(0) - alpha(1); bf = beta(0) - beta(1);
                    cf = -(center(0) - center(1)); break;
            default: af = alpha(1) - alpha(2); bf = beta(1) - beta(2);
                     cf = -(center(1) - center(2)); break;
        }
        return af * std::cos(omega * t) + bf * std::sin(omega * t) - cf;
    };
    NT t_ref = std::numeric_limits<NT>::max();
    for (unsigned int f = 0; f < 4; ++f) {
        NT tp = NT(0), gp = gref(f, NT(0));
        for (NT t = NT(1e-4); t <= NT(6); t += NT(1e-4)) {
            NT const g = gref(f, t);
            if (gp <= NT(0) && g > NT(0)) {
                NT lo = tp, hi = t;
                for (int it = 0; it < 80; ++it) {
                    NT const mid = NT(0.5) * (lo + hi);
                    if (gref(f, mid) > NT(0)) hi = mid; else lo = mid;
                }
                t_ref = std::min(t_ref, NT(0.5) * (lo + hi));
                break;
            }
            tp = t; gp = g;
        }
    }
    CHECK(std::abs(hit.t - t_ref) < NT(1e-8));
}

TEST_CASE("modal_oracle_arbitrary_spd") {
    // Arbitrary unrelated SPD A, M on n = 3..6 posets: the returned hit
    // satisfies its facet equation and no facet is violated before it.
    struct Case { unsigned int n; Poset::RV rels; };
    std::vector<Case> cases = {
        {3, {{0, 1}, {1, 2}}},
        {4, {{0, 1}, {0, 2}, {1, 3}, {2, 3}}},
        {5, {{0, 2}, {1, 2}, {2, 4}, {3, 4}}},
        {6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}},
    };

    for (auto const& tc : cases) {
        Poset::RV rels = tc.rels;
        Poset poset(tc.n, rels);
        OP_t OP(poset);

        MT const A = test_spd(tc.n, NT(3), NT(0.7));
        MT const M = test_spd(tc.n, NT(1.4), NT(-0.35));
        VT center(VT::Constant(tc.n, NT(0.5)));
        Modal traj(OP, A, M, center);

        Point inner(OP.inner_point());
        VT x(tc.n), v(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i) {
            x(i) = inner[i];
            v(i) = NT(0.9) - NT(0.6) * NT(i % 3);
        }
        traj.set_state(x, v);

        auto const hit = first_hit_modal_dense(traj, NT(8), -1);
        REQUIRE(hit.facet >= 0);
        CHECK(std::isfinite(hit.t));
        CHECK(hit.t > NT(0));
        CHECK(std::abs(traj.facet_value(hit.facet, hit.t)) < NT(1e-8));
        CHECK(traj.facet_derivative(hit.facet, hit.t) > NT(0));

        for (unsigned int s = 1; s < 100; ++s) {
            NT const t = hit.t * NT(0.999) * NT(s) / NT(100);
            for (unsigned int f = 0; f < traj.num_facets(); ++f)
                CHECK(traj.facet_value(f, t) <= NT(1e-9));
        }
    }
}

TEST_CASE("modal_oracle_repeated_facet") {
    Poset::RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    MT const A = test_spd(2, NT(2), NT(0.4));
    MT const M = test_spd(2, NT(1), NT(0.15));
    VT center(VT::Constant(2, NT(0.5)));
    Modal traj(OP, A, M, center);

    // On lb(0) (x0 = 0) moving outward: with no last-hit exclusion this
    // is an immediate hit at t = 0 (facet 0 is lb(0) in build order).
    VT x(2), v(2);
    x << NT(0), NT(0.5);
    v << NT(-0.4), NT(0.2);
    traj.set_state(x, v);
    auto const hit0 = first_hit_modal_dense(traj, NT(5), -1);
    CHECK(hit0.facet == 0);
    CHECK(hit0.t == NT(0));

    // Same state moving inward with lb(0) as the just-reflected facet:
    // the residual root is suppressed and the returned hit is a genuine
    // later crossing of some facet, strictly past t_eps.
    v(0) = NT(0.4);
    traj.set_state(x, v);
    auto const hit1 = first_hit_modal_dense(traj, NT(5), 0);
    REQUIRE(hit1.facet >= 0);
    CHECK(hit1.t > NT(1e-6));
    CHECK(std::abs(traj.facet_value(hit1.facet, hit1.t)) < NT(1e-8));
}

TEST_CASE("modal_oracle_near_simultaneous") {
    // Two disjoint facets crossing at the same time: after reflecting on
    // the first and re-querying from the hit state, the second facet's
    // zero-time crossing must be returned, not discarded as residual.
    Poset::RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    NT const a = NT(0.5);
    Modal traj(OP, MT(NT(2) * a * MT::Identity(2, 2)),
               MT(MT::Identity(2, 2)), VT(VT::Zero(2)));

    VT x(2), v(2);
    x << NT(0.5), NT(0.5);
    v << NT(-0.375), NT(-0.375);
    traj.set_state(x, v);

    auto const hit_a = first_hit_modal_dense(traj, NT(3), -1);
    REQUIRE(hit_a.facet >= 0);
    REQUIRE(hit_a.t > NT(0));

    // Advance to the hit, reflect the hit facet (a wall with M = I:
    // flip its coordinate's velocity), and re-query with the reflected
    // facet marked as just-hit.
    VT x1 = traj.position(hit_a.t);
    VT v1 = traj.velocity(hit_a.t);
    unsigned int const hit_coord = traj.facet_i0(hit_a.facet);
    v1(hit_coord) = -v1(hit_coord);
    traj.set_state(x1, v1);

    auto const hit_b = first_hit_modal_dense(traj, NT(3), hit_a.facet);
    REQUIRE(hit_b.facet >= 0);
    CHECK(hit_b.facet != hit_a.facet);
    CHECK(hit_b.t == NT(0));

    // Once the second facet is reflected as well, no immediate hit
    // remains from the corner state.
    v1(traj.facet_i0(hit_b.facet)) = -v1(traj.facet_i0(hit_b.facet));
    traj.set_state(x1, v1);
    auto const hit_c = first_hit_modal_dense(traj, NT(0.3), hit_b.facet);
    CHECK(hit_c.facet == -1);
}


// ---- Part 3: ModalDenseFallback walk -----------------------------------

typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk MatchedPolicy;
typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk SphericalPolicy;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 97> RNGType;
typedef typename MatchedPolicy::template Walk<OP_t, RNGType> MWalk;
typedef typename SphericalPolicy::template Walk<OP_t, RNGType> SWalk;

TEST_CASE("modal_walk_spherical_reduction") {
    // UserSpecified A = 2a I, M = I: the modal walk must reproduce the
    // spherical event-queue walk leg by leg (including the PR-0 exact
    // simultaneous-crossing fixture).
    NT const a = NT(0.5);   // omega = 1

    struct Case { unsigned int n; Poset::RV rels; std::vector<NT> p, v; NT T; };
    std::vector<Case> cases = {
        {2, {}, {NT(0.5), NT(0.5)}, {NT(-0.375), NT(-0.375)},
         NT(1.57079632679489661923)},
        {3, {{0, 1}, {1, 2}}, {NT(0.25), NT(0.375), NT(0.5)},
         {NT(0.34375), NT(0.25), NT(0.875)}, NT(1.57079632679489661923)},
        {3, {{0, 1}, {1, 2}}, {NT(0.3), NT(0.4), NT(0.6)},
         {NT(0.9), NT(-0.7), NT(0.2)}, NT(1.3)},
    };

    for (auto const& tc : cases) {
        Poset::RV rels = tc.rels;
        Poset poset(tc.n, rels);
        OP_t OP(poset);

        Point start(tc.n), v0(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i) {
            start.set_coord(i, tc.p[i]);
            v0.set_coord(i, tc.v[i]);
        }

        typename MatchedPolicy::parameters<NT> params(
            MT(NT(2) * a * MT::Identity(tc.n, tc.n)),
            MT(MT::Identity(tc.n, tc.n)), VT(VT::Zero(tc.n)));
        RNGType rng_m(tc.n);
        MWalk mwalk(OP, start, params, rng_m);
        CHECK(mwalk.backend() == GaussianHMCBackend::ModalDenseFallback);

        RNGType rng_s(tc.n);
        SWalk swalk(OP, start, a, rng_s);

        Point pm = start, vm = v0, ps = start;
        CHECK(mwalk.apply_leg(OP, pm, vm, tc.T));
        CHECK(swalk.apply_leg(OP, ps, v0, tc.T));
        Point vs = swalk.velocity();
        for (unsigned int i = 0; i < tc.n; ++i) {
            CHECK(std::abs(pm[i] - ps[i]) < NT(1e-9));
            CHECK(std::abs(vm[i] - vs[i]) < NT(1e-9));
        }
    }

    // A user-specified mass must never run on a matched fast path.
    {
        Poset::RV rels;
        Poset poset(2, rels);
        OP_t OP(poset);
        Point start(2, {NT(0.5), NT(0.5)});
        typename MatchedPolicy::parameters<NT> bad(
            MT(MT::Identity(2, 2)), MT(MT::Identity(2, 2)), VT(VT::Zero(2)),
            GaussianHMCBackend::MatchedDenseAngleFullScan);
        RNGType rng(2);
        CHECK_THROWS_AS(MWalk(OP, start, bad, rng), std::invalid_argument);
    }
}

TEST_CASE("modal_walk_matched_reduction") {
    // UserSpecified M = A / omega^2 must agree with the matched dense
    // full-scan backend on deterministic legs.
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);
    OP_t OP(poset);

    MT const A = test_spd(3, NT(2), NT(0.5));
    NT const omega = NT(1.4);
    VT center(3);
    center << NT(0.3), NT(0.5), NT(0.65);

    Point start(3, {NT(0.25), NT(0.45), NT(0.7)});

    typename MatchedPolicy::parameters<NT> pmodal(
        A, MT(A / (omega * omega)), center);
    RNGType rng_a(3);
    MWalk wmodal(OP, start, pmodal, rng_a);
    CHECK(wmodal.backend() == GaussianHMCBackend::ModalDenseFallback);

    typename MatchedPolicy::parameters<NT> pdense(
        A, center, omega, GaussianHMCBackend::MatchedDenseAngleFullScan);
    RNGType rng_b(3);
    MWalk wdense(OP, start, pdense, rng_b);

    Point pa = start, pb = start;
    for (unsigned int leg = 0; leg < 3; ++leg) {
        Point va(3), vb(3);
        for (unsigned int i = 0; i < 3; ++i) {
            NT const vi = NT(0.8) - NT(0.5) * NT((i + leg) % 3);
            va.set_coord(i, vi);
            vb.set_coord(i, vi);
        }
        NT const T = NT(0.9) + NT(0.3) * NT(leg);
        CHECK(wmodal.apply_leg(OP, pa, va, T));
        CHECK(wdense.apply_leg(OP, pb, vb, T));
        for (unsigned int i = 0; i < 3; ++i) {
            CHECK(std::abs(pa[i] - pb[i]) < NT(1e-8));
            CHECK(std::abs(va[i] - vb[i]) < NT(1e-8));
        }
    }
}

TEST_CASE("modal_walk_arbitrary_spd") {
    // Arbitrary unrelated SPD A and M: feasibility, finiteness,
    // Hamiltonian conservation, zero hard violations.
    struct Case { unsigned int n; Poset::RV rels; };
    std::vector<Case> cases = {
        {3, {{0, 1}, {1, 2}}},
        {3, {}},
        {4, {{0, 1}, {0, 2}, {1, 3}, {2, 3}}},
        {5, {{0, 2}, {1, 2}, {2, 4}, {3, 4}}},
    };

    for (auto const& tc : cases) {
        Poset::RV rels = tc.rels;
        Poset poset(tc.n, rels);
        OP_t OP(poset);

        MT const A = test_spd(tc.n, NT(3), NT(0.7));
        MT const M = test_spd(tc.n, NT(1.4), NT(-0.35));
        typename MatchedPolicy::parameters<NT> params(
            A, M, VT(VT::Constant(tc.n, NT(0.5))));

        Point p(OP.inner_point());
        RNGType rng(tc.n);
        MWalk walk(OP, p, params, rng);

        // Hamiltonian conservation across a deterministic multi-
        // reflection leg.
        Point pl = p;
        Point vl(tc.n);
        for (unsigned int i = 0; i < tc.n; ++i)
            vl.set_coord(i, NT(0.9) - NT(0.5) * NT(i % 3));
        NT const H0 = walk.hamiltonian(pl, vl);
        CHECK(walk.apply_leg(OP, pl, vl, NT(1.2)));
        NT const H1 = walk.hamiltonian(pl, vl);
        CHECK(std::abs(H1 - H0) < NT(1e-8) * std::max(NT(1), std::abs(H0)));
        CHECK(OP.is_in(pl, NT(1e-8)) == -1);

        unsigned int violations = 0;
        for (unsigned int step = 0; step < 150; ++step) {
            walk.apply(OP, p, 1, rng);
            for (unsigned int i = 0; i < tc.n; ++i)
                CHECK(std::isfinite(p[i]));
            if (OP.is_in(p, NT(1e-7)) != -1) ++violations;
        }
        CHECK(violations == 0);
        CHECK(walk.hard_negative_slack_count() == 0);
    }
}

TEST_CASE("modal_walk_reflection_invariant") {
    // For each facet type with M != A: n^T v flips sign and the kinetic
    // form v^T M v is preserved; the map equals the independent Cholesky
    // formula and is an involution.
    Poset::RV rels{{0, 1}};
    Poset poset(2, rels);
    OP_t OP(poset);   // facets: lb(0), ub(1), cover

    MT const A = test_spd(2, NT(2.5), NT(0.6));
    MT const M = test_spd(2, NT(1.2), NT(0.35));
    typename MatchedPolicy::parameters<NT> params(
        A, M, VT(VT::Constant(2, NT(0.5))));
    Point start(2, {NT(0.3), NT(0.6)});
    RNGType rng(2);
    MWalk walk(OP, start, params, rng);
    REQUIRE(walk.num_event_facets() == 3);

    std::vector<VT> normals;
    VT n0(2), n1(2), n2(2);
    n0 << NT(-1), NT(0);
    n1 << NT(0), NT(1);
    n2 << NT(1), NT(-1);
    normals = {n0, n1, n2};

    for (unsigned int fid = 0; fid < 3; ++fid) {
        VT v(2);
        v << NT(0.9), NT(-0.4);
        NT const nv = normals[fid].dot(v);
        NT const K0 = v.dot(M * v);

        VT v1 = v;
        walk.reflect_velocity(fid, v1);
        CHECK(std::abs(normals[fid].dot(v1) + nv) < NT(1e-12));
        CHECK(std::abs(v1.dot(M * v1) - K0) < NT(1e-12) * K0);

        VT const w = M.llt().solve(normals[fid]);
        VT const v_ref = v - (NT(2) * nv / normals[fid].dot(w)) * w;
        CHECK(std::abs(v1(0) - v_ref(0)) < NT(1e-13));
        CHECK(std::abs(v1(1) - v_ref(1)) < NT(1e-13));

        VT v2 = v1;
        walk.reflect_velocity(fid, v2);
        CHECK(std::abs(v2(0) - v(0)) < NT(1e-12));
        CHECK(std::abs(v2(1) - v(1)) < NT(1e-12));
    }
}

TEST_CASE("modal_walk_distribution") {
    // The stationary law depends on A and c only, never on M: with a
    // diagonal A on a box the coordinate means must match the truncated
    // normal quadrature reference even under an arbitrary dense mass.
    Poset::RV rels;
    Poset poset(2, rels);
    OP_t OP(poset);

    MT A(MT::Zero(2, 2));
    A(0, 0) = NT(6);
    A(1, 1) = NT(3);
    MT Mmass(2, 2);
    Mmass << NT(1.5), NT(0.4), NT(0.4), NT(1);
    VT center(2);
    center << NT(0.3), NT(0.7);

    typename MatchedPolicy::parameters<NT> params(A, Mmass, center);
    Point p(OP.inner_point());
    RNGType rng(2);
    MWalk walk(OP, p, params, rng);

    auto truncated_normal_mean = [](NT prec, NT c) {
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
    };

    unsigned int const N = 20000;
    NT sum0 = NT(0), sum1 = NT(0);
    for (unsigned int step = 0; step < N; ++step) {
        walk.apply(OP, p, 1, rng);
        sum0 += p[0];
        sum1 += p[1];
    }
    CHECK(std::abs(sum0 / NT(N) - truncated_normal_mean(A(0, 0), center(0)))
          < NT(0.02));
    CHECK(std::abs(sum1 / NT(N) - truncated_normal_mean(A(1, 1), center(1)))
          < NT(0.02));
    CHECK(walk.hard_negative_slack_count() == 0);
}
