// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2021- Vaibhav Thakkar

// Contributed by Vaibhav Thakkar, as part of Google Summer of Code 2021 program.
// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"

#include <cmath>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/orderpolytope.h"
#include "generators/order_polytope_generator.h"
#include "misc/poset.h"
#include "random_walks/random_walks.hpp"
#include "sampling/sampling.hpp"


template <typename NT>
void call_test_reflection()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;

    Poset::RV relations{{0, 1}, {0, 2}, {1, 3}};
    OrderPolytope<Point> OP(Poset(4, relations));

    Point ray = Point::all_ones(OP.dimension());
    ray.set_coord(0, NT(1.5));
    Point expected = Point::all_ones(OP.dimension());
    expected.set_coord(1, NT(1.5));

    OP.compute_reflection(ray, Point(), 2 * OP.dimension());
    CHECK(expected == ray);
}


template <typename NT>
void call_test_line_intersect()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;

    Poset::RV relations;
    Poset poset(3, relations);
    OrderPolytope<Point> OP(poset);
    Point start(3, {NT(0.5), NT(0.5), NT(0.5)});
    Point expected = start;
    expected.set_coord(0, NT(1));

    Point direction = expected - start;
    std::pair<NT, NT> interval = OP.line_intersect(start, direction, true);
    CHECK(start + interval.first * direction == expected);
}


template <typename NT>
void call_test_vec_mult()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;

    Poset::RV relations{{0, 3}, {1, 3}, {2, 3}};
    OrderPolytope<Point> OP(Poset(4, relations));
    unsigned int const d = OP.dimension();
    unsigned int const m = OP.num_of_hyperplanes();

    VT x = VT::Ones(d);
    VT expected = -VT::Ones(m);
    expected.segment(d, d).setOnes();
    expected.tail(m - 2 * d).setZero();
    CHECK((expected - OP.vec_mult(x)).norm() == NT(0));

    x = VT::Ones(m);
    expected = VT::Ones(d);
    expected(3) = NT(-3);
    CHECK((expected - OP.vec_mult(x, true)).norm() == NT(0));
}


template <typename NT>
void call_test_basics()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;

    Poset::RV relations{{0, 1}, {0, 2}, {1, 3}};
    Poset poset(4, relations);
    CHECK(poset.num_elem() == 4);
    CHECK(poset.num_relations() == 3);

    OrderPolytope<Point> OP(poset);
    unsigned int const d = OP.dimension();
    unsigned int const m = OP.num_of_hyperplanes();
    CHECK(d == 4);
    CHECK(m == 11);

    VT expected = VT::Zero(m);
    expected.segment(d, d).setOnes();
    VT distances = Eigen::Map<VT>(OP.get_dists(NT(0)).data(), m);
    CHECK((expected - distances).norm() == NT(0));

    CHECK(OP.is_in(Point(4, {NT(0), NT(0.5), NT(1), NT(1)})) == -1);
    CHECK(OP.is_in(Point(4, {NT(1), NT(0.5), NT(1), NT(1)})) == 0);
    CHECK(OP.is_in(Point(4, {NT(0.5), NT(0.5), NT(0), NT(1)})) == 0);
    CHECK(OP.is_in(Point(4, {NT(-0.1), NT(0.5), NT(1), NT(1)})) == 0);
    CHECK(OP.is_in(Point(4, {NT(1), NT(1), NT(1), NT(1.1)})) == 0);

    HPolytope<Point> HP = random_orderpoly<HPolytope<Point>, NT>(10, 30);
    CHECK(HP.dimension() == 10);
    CHECK(HP.num_of_hyperplanes() >= 20);
    CHECK(HP.num_of_hyperplanes() <= 30);
}


template <typename NT>
void call_test_true_box_facets()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;

    Poset::RV relations{{0, 1}, {0, 2}, {2, 3}};
    OrderPolytope<Point> OP(Poset(4, relations));

    CHECK(OP.lower_bound_is_facet(0));
    CHECK_FALSE(OP.lower_bound_is_facet(1));
    CHECK_FALSE(OP.lower_bound_is_facet(2));
    CHECK_FALSE(OP.lower_bound_is_facet(3));
    CHECK_FALSE(OP.upper_bound_is_facet(0));
    CHECK(OP.upper_bound_is_facet(1));
    CHECK_FALSE(OP.upper_bound_is_facet(2));
    CHECK(OP.upper_bound_is_facet(3));
}


template <typename NT>
void call_test_true_relation_facets()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;

    Poset::RV relations{{0, 2}, {1, 2}, {0, 1}, {0, 1}};
    Poset poset(3, relations);
    OrderPolytope<Point> OP(poset);

    CHECK(OP.num_order_relations() == 4);
    CHECK(OP.num_of_hyperplanes() == 10);
    CHECK(OP.num_true_facets() == 4);
    std::vector<unsigned int> const& covers = OP.cover_relation_indices();
    REQUIRE(covers.size() == 2);
    CHECK(covers[0] == 1);
    CHECK(covers[1] == 2);

    Poset const reduced_poset = poset.transitive_reduction();
    OrderPolytope<Point> reduced =
        OrderPolytope<Point>::from_reduced_relations(reduced_poset);
    CHECK(reduced.num_order_relations() == 2);
    CHECK(reduced.num_of_hyperplanes() == 8);
    CHECK(reduced.num_true_facets() == 4);
    CHECK(reduced.cover_relation_indices() ==
          std::vector<unsigned int>{0, 1});

    OrderPolytope<Point> reduced_by_legacy_flag(poset, true);
    CHECK(reduced_by_legacy_flag.num_order_relations() == 2);
    CHECK(reduced_by_legacy_flag.cover_relation_indices() ==
          std::vector<unsigned int>{0, 1});
}


template <typename NT>
void call_test_coord_intersect_normalized_shifted()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef HPolytope<Point> HP_t;
    typedef typename OP_t::VT VT;

    Poset::RV relations{{0, 1}, {1, 2}};
    OP_t OP(Poset(3, relations));
    HP_t HP(OP.dimension(), OP.get_dense_mat(), OP.get_vec());
    OP.normalize();
    HP.normalize();

    VT center(3);
    center << NT(0.2), NT(0.5), NT(0.8);
    OP.shift(center);
    HP.shift(center);

    Point p(3);
    VT op_lambdas = VT::Zero(OP.num_of_hyperplanes());
    VT hp_lambdas = VT::Zero(HP.num_of_hyperplanes());
    auto op_first = OP.line_intersect_coord(p, 0, op_lambdas);
    auto hp_first = HP.line_intersect_coord(p, 0, hp_lambdas);
    CHECK(op_first.first == doctest::Approx(hp_first.first));
    CHECK(op_first.second == doctest::Approx(hp_first.second));

    Point previous = p;
    p.set_coord(0, (op_first.first + op_first.second) / NT(3));
    auto op_second = OP.line_intersect_coord(p, previous, 1, 0, op_lambdas);
    auto hp_second = HP.line_intersect_coord(p, previous, 1, 0, hp_lambdas);
    CHECK(op_second.first == doctest::Approx(hp_second.first));
    CHECK(op_second.second == doctest::Approx(hp_second.second));
}


template <typename NT>
class ScriptedGaussianRNG
{
public:
    ScriptedGaussianRNG(std::vector<NT> normal_values,
                        std::vector<NT> uniform_values)
        : normals(std::move(normal_values)),
          uniforms(std::move(uniform_values))
    {}

    NT sample_ndist()
    {
        REQUIRE(normal_index < normals.size());
        return normals[normal_index++];
    }

    NT sample_urdist()
    {
        REQUIRE(uniform_index < uniforms.size());
        return uniforms[uniform_index++];
    }

private:
    std::vector<NT> normals;
    std::vector<NT> uniforms;
    std::size_t normal_index = 0;
    std::size_t uniform_index = 0;
};


template <typename NT>
struct ScriptedOutcome
{
    explicit ScriptedOutcome(unsigned int n) : point(n) {}
    typename Cartesian<NT>::Point point;
    std::string failure;
};


template <typename NT>
ScriptedOutcome<NT> run_scripted_hmc(
    unsigned int n, Poset::RV const& relations,
    std::vector<NT> const& start, std::vector<NT> normals,
    std::vector<NT> uniforms, unsigned int rho = 100,
    NT a = NT(1), NT length = NT(1), bool normalize = false)
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV stored_relations = relations;
    Poset poset(n, stored_relations);
    OP_t OP(poset);
    if (normalize) OP.normalize();
    Point p(n, start);
    RNG rng(std::move(normals), std::move(uniforms));
    typename OrderPolytopeExactHMCWalk::parameters
        params(length, true, rho, true);
    ScriptedOutcome<NT> outcome(n);
    try {
        Walk walk(OP, p, a, rng, params);
        walk.apply(OP, p, a, 1, rng);
    } catch (std::runtime_error const& error) {
        outcome.failure = error.what();
    }
    outcome.point = p;
    return outcome;
}


template <typename NT>
void test_order_hmc()
{
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 42> RNG;

    Poset::RV relations{{0, 1}};
    Poset poset(2, relations);
    OP_t OP(poset);
    Point start(OP.inner_point());
    RNG rng(OP.dimension());
    std::list<Point> samples;
    gaussian_sampling<OrderPolytopeExactHMCWalk>(
        samples, OP, rng, 1, 40, NT(1), start, 2);

    CHECK(samples.size() == 40);
    for (Point const& sample : samples)
        CHECK(OP.is_in(sample, NT(1e-7)) == -1);
}


template <typename NT>
void test_order_hmc_tied_contacts()
{
    ScriptedOutcome<NT> bounds = run_scripted_hmc<NT>(
        2, {}, {NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(2), NT(2)}, {NT(0), NT(0.75)});
    CHECK(bounds.failure.empty());
    CHECK(bounds.point[0] == doctest::Approx(bounds.point[1]).epsilon(1e-13));
    CHECK(bounds.point[0] ==
          doctest::Approx(NT(0.08838611995487355)).epsilon(1e-12));

    ScriptedOutcome<NT> covers = run_scripted_hmc<NT>(
        4, {{0, 1}, {2, 3}},
        {NT(0.25), NT(0.75), NT(0.25), NT(0.75)},
        {NT(0), NT(0), NT(0), NT(0), NT(1), NT(-1), NT(1), NT(-1)},
        {NT(0), NT(0.75)});
    CHECK(covers.failure.empty());
    CHECK(covers.point[0] == doctest::Approx(covers.point[2]).epsilon(1e-13));
    CHECK(covers.point[1] == doctest::Approx(covers.point[3]).epsilon(1e-13));
    CHECK(covers.point[0] < covers.point[1]);

    NT const theta = std::atan2(NT(0.8), NT(0.6));
    NT const end = NT(1);
    ScriptedOutcome<NT> triple = run_scripted_hmc<NT>(
        3, {}, {NT(0.5), NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(0), NT(0.875), NT(0.875), NT(0.875)},
        {NT(0), NT(0.8)}, 100, NT(0.5), NT(1.25));
    NT const expected = std::cos(end - theta)
                      - NT(0.125) * std::sin(end - theta);
    CHECK(triple.failure.empty());
    for (unsigned int i = 0; i < 3; ++i)
        CHECK(triple.point[i] == doctest::Approx(expected).epsilon(1e-12));

    NT const pi = std::acos(NT(-1));
    ScriptedOutcome<NT> mixed = run_scripted_hmc<NT>(
        3, {{0, 1}, {1, 2}}, {NT(0.25), NT(0.375), NT(0.5)},
        {NT(0), NT(0), NT(0), NT(0.34375), NT(0.25), NT(0.875)},
        {NT(0), pi / NT(4)}, 100, NT(0.5), NT(2));
    CHECK(mixed.failure.empty());
    CHECK(mixed.point[0] == doctest::Approx(NT(0.25)).epsilon(1e-12));
    CHECK(mixed.point[1] == doctest::Approx(NT(0.34375)).epsilon(1e-12));
    CHECK(mixed.point[2] == doctest::Approx(NT(0.725)).epsilon(1e-12));

    ScriptedOutcome<NT> near = run_scripted_hmc<NT>(
        2, {}, {NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(2), NT(1.999999999)},
        {NT(0), NT(0.75)});
    CHECK(near.failure.empty());
    CHECK(std::abs(near.point[0] - near.point[1]) > NT(1e-12));

    NT const alpha = NT(0x1.ffffde7210be9p-1);
    NT const beta = NT(0x1.72ba3ded215c5p-10);
    ScriptedOutcome<NT> shallow = run_scripted_hmc<NT>(
        1, {}, {alpha}, {NT(0), beta}, {NT(0), NT(0.01)},
        100, NT(0.5), NT(1));
    NT const shallow_free = alpha * std::cos(NT(0.01)) +
                            beta * std::sin(NT(0.01));
    CHECK(shallow.failure.empty());
    CHECK(shallow.point[0] < shallow_free - NT(1e-12));
    CHECK(shallow.point[0] >= NT(0));
    CHECK(shallow.point[0] <= NT(1));

    ScriptedOutcome<NT> tangent = run_scripted_hmc<NT>(
        1, {}, {NT(1)}, {NT(0), NT(0)}, {NT(0), NT(0.1)},
        100, NT(0.5), NT(1));
    CHECK(tangent.failure.empty());
    CHECK(tangent.point[0] ==
          doctest::Approx(std::cos(NT(0.1))).epsilon(1e-14));

    NT const endpoint = NT(0.5), delta = NT(1e-14);
    NT const endpoint_velocity =
        (NT(1) - NT(0.5) * std::cos(endpoint + delta)) /
        std::sin(endpoint + delta);
    ScriptedOutcome<NT> after_endpoint = run_scripted_hmc<NT>(
        1, {}, {NT(0.5)}, {NT(0), endpoint_velocity},
        {NT(0), endpoint}, 1, NT(0.5), NT(1));
    NT const endpoint_free = NT(0.5) * std::cos(endpoint) +
                             endpoint_velocity * std::sin(endpoint);
    CHECK(after_endpoint.failure.empty());
    CHECK(after_endpoint.point[0] ==
          doctest::Approx(endpoint_free).epsilon(1e-14));

    NT const endpoint_tie_time = std::atan2(NT(0.8), NT(0.6));
    ScriptedOutcome<NT> endpoint_tie = run_scripted_hmc<NT>(
        2, {}, {NT(0), NT(0.25)},
        {NT(0), NT(0), NT(1.25), NT(1.0625)},
        {NT(0), endpoint_tie_time}, 100, NT(0.5), NT(1));
    CHECK(endpoint_tie.failure.empty());
    CHECK(endpoint_tie.point[0] ==
          doctest::Approx(NT(1)).epsilon(1e-14));
    CHECK(endpoint_tie.point[1] ==
          doctest::Approx(NT(1)).epsilon(1e-14));

    ScriptedOutcome<NT> large_velocity = run_scripted_hmc<NT>(
        1, {}, {NT(0.5)}, {NT(0), NT(1e14)},
        {NT(0), NT(1e-15)}, 100, NT(0.5), NT(1));
    CHECK(large_velocity.failure.empty());
    CHECK(large_velocity.point[0] == doctest::Approx(NT(0.6)).epsilon(1e-14));

    NT const distinct_time = NT(0.5), separation = NT(1e-13);
    NT const a0 = std::sin(distinct_time);
    NT const a1 = separation * std::cos(distinct_time) + NT(2) * a0;
    NT const b0 = -std::cos(distinct_time);
    NT const b1 = separation * std::sin(distinct_time) -
                  NT(2) * std::cos(distinct_time);
    ScriptedOutcome<NT> distinct = run_scripted_hmc<NT>(
        2, {{0, 1}}, {a0, a1}, {NT(0), NT(0), b0, b1},
        {NT(0), NT(0.75)}, 100, NT(0.5), NT(1));
    CHECK(distinct.failure.empty());
    CHECK(distinct.point[0] < distinct.point[1]);

    ScriptedOutcome<NT> rounded_distinct = run_scripted_hmc<NT>(
        2, {{0, 1}},
        {NT(0x1.447e6bc8ce36fp-3), NT(0x1.97b67986bf4b0p-2)},
        {NT(0), NT(0), NT(0x1.cbb9821d75f78p+0),
         NT(0x1.5b64e20408024p+0)}, {NT(0), NT(0.75)},
        100, NT(0.5), NT(1));
    CHECK(rounded_distinct.failure.empty());
    CHECK(rounded_distinct.point[0] < rounded_distinct.point[1]);

    ScriptedOutcome<NT> close_disjoint = run_scripted_hmc<NT>(
        2, {}, {NT(0x1.ccccccccccccdp-1), NT(0x1.9eb851eb851ecp-1)},
        {NT(0), NT(0), NT(0x1.dffa0e8f87f83p-2),
         NT(0x1.801964f98aa01p-1)}, {NT(0), NT(0.7)},
        100, NT(0.5), NT(1));
    CHECK(close_disjoint.failure.empty());
    CHECK(close_disjoint.point[0] ==
          doctest::Approx(NT(0.85910187138570515)).epsilon(1e-13));
    CHECK(close_disjoint.point[1] ==
          doctest::Approx(NT(0.74661202510384406)).epsilon(1e-13));

    NT const q = std::ldexp(NT(1), -35);
    std::vector<NT> const missed_tie_start{
        NT(0.125) - NT(4) * q, NT(0.125)};
    ScriptedOutcome<NT> missed_tie = run_scripted_hmc<NT>(
        2, {{0, 1}}, missed_tie_start,
        {NT(0), NT(0), NT(37) / NT(32) + NT(3) * q, NT(37) / NT(32)},
        {NT(0), NT(1)}, 100, NT(0.5), NT(1.1), true);
    CHECK(missed_tie.failure.find("simultaneous contact") != std::string::npos);
    CHECK(missed_tie.point[0] == missed_tie_start[0]);
    CHECK(missed_tie.point[1] == missed_tie_start[1]);
}


template <typename NT>
void test_order_hmc_shared_contact()
{
    std::vector<NT> const start{NT(0.25), NT(0.5)};
    ScriptedOutcome<NT> outcome = run_scripted_hmc<NT>(
        2, {{0, 1}}, start,
        {NT(0), NT(0), NT(-1), NT(-2)}, {NT(0), NT(0.75)});

    CHECK(outcome.failure.find("simultaneous contact") != std::string::npos);
    CHECK(outcome.point[0] == start[0]);
    CHECK(outcome.point[1] == start[1]);
}


template <typename NT>
void test_order_hmc_reflection_rollback()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV relations;
    Poset poset(1, relations);
    OP_t OP(poset);
    Point p(1, {NT(0.5)});
    RNG rng({NT(0), NT(2), NT(0)}, {NT(0), NT(0.75), NT(0.5)});
    typename OrderPolytopeExactHMCWalk::parameters
        params(NT(1), true, 1, true);
    Walk walk(OP, p, NT(1), rng, params);

    walk.apply(OP, p, NT(1), 1, rng);
    CHECK(p[0] == NT(0.5));

    walk.update_delta(NT(0.2));
    walk.apply(OP, p, NT(1), 1, rng);
    NT const expected = NT(0.5) * std::cos(std::sqrt(NT(2)) * NT(0.1));
    CHECK(p[0] == doctest::Approx(expected).epsilon(1e-12));

    Point external(1, {NT(0.9)});
    walk.apply(OP, external, NT(1), 0, rng);
    CHECK(external[0] == p[0]);

    ScriptedOutcome<NT> long_leg = run_scripted_hmc<NT>(
        1, {}, {NT(0.5)}, {NT(0), NT(2)},
        {NT(0), NT(0.8)}, 100, NT(0.5), NT(5));
    CHECK(long_leg.failure.empty());
    CHECK(std::isfinite(long_leg.point[0]));
    CHECK(long_leg.point[0] >= NT(0));
    CHECK(long_leg.point[0] <= NT(1));
}


template <typename NT>
void test_order_hmc_boundary_refresh()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV relations;
    Poset poset(1, relations);
    OP_t OP(poset);
    Point p(1, {NT(0.5)});
    NT const hit_time = std::atan2(NT(0.8), NT(0.6));
    NT const leg_time = NT(0.2);
    RNG rng({NT(0.875), NT(0.75)}, {hit_time, leg_time});
    typename OrderPolytopeExactHMCWalk::parameters
        params(NT(1), true, 100, true);

    Walk walk(OP, p, NT(0.5), rng, params);
    walk.apply(OP, p, NT(0.5), 1, rng);

    NT const expected = std::cos(leg_time) - NT(0.75) * std::sin(leg_time);
    CHECK(p[0] == doctest::Approx(expected).epsilon(1e-12));
    CHECK(OP.is_in(p, NT(0)) == -1);
}


template <typename NT>
void test_order_hmc_tiny_shifted_crossing()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV relations;
    OP_t OP(Poset(1, relations));
    VT center(1);
    center << NT(1) - NT(1e-13);
    OP.shift(center);
    NT const upper = OP.get_vec()(1);

    Point p(1, {NT(0)});
    NT const pi = std::acos(NT(-1));
    RNG rng({NT(0), NT(2) * upper}, {NT(0), NT(1)});
    typename OrderPolytopeExactHMCWalk::parameters
        params(pi, true, 100, true);
    Walk walk(OP, p, NT(0.5), rng, params);
    walk.apply(OP, p, NT(0.5), 1, rng);

    CHECK(OP.is_in(p, NT(0)) == -1);
    CHECK(p[0] < -upper);
    CHECK(p[0] > -NT(2) * upper);
}


template <typename NT>
void test_order_hmc_last_hit_expires_after_free_flight()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV relations;
    OP_t OP(Poset(1, relations));
    VT center(1);
    center << NT(0.75);
    OP.shift(center);
    NT const upper = OP.get_vec()(1);
    NT const pi = std::acos(NT(-1));
    NT const delta = NT(5e-11);
    NT const velocity = upper * std::tan(pi / NT(4) - delta / NT(2));
    NT const hit_time = NT(1.5) * pi + delta;
    NT const final_time = NT(2) * pi;
    NT const outward_velocity =
        -upper * std::sin(hit_time) - velocity * std::cos(hit_time);
    NT const expected = upper * std::cos(final_time - hit_time) -
                        outward_velocity * std::sin(final_time - hit_time);

    Point p(1, {upper});
    RNG rng({NT(0), velocity}, {NT(0), NT(1)});
    typename OrderPolytopeExactHMCWalk::parameters
        params(final_time, true, 100, true);
    Walk walk(OP, p, NT(0.5), rng, params);
    walk.apply(OP, p, NT(0.5), 1, rng);

    CHECK(OP.is_in(p, NT(0)) == -1);
    CHECK(p[0] == doctest::Approx(expected).epsilon(1e-12));
}


template <typename NT>
void test_order_hmc_shifted_containment()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 31> RNG;

    Poset::RV relations{{0, 1}};
    Poset poset(2, relations);
    OP_t OP(poset);
    OP.normalize();
    VT center(2);
    center << NT(0.2), NT(0.7);
    OP.shift(center);

    Point start(2, {NT(0.15), NT(0.05)});
    REQUIRE(OP.is_in(start) == -1);
    RNG rng(OP.dimension());
    std::list<Point> samples;
    gaussian_sampling<OrderPolytopeExactHMCWalk>(
        samples, OP, rng, 1, 100, NT(1), start, 2);
    for (Point const& sample : samples)
        CHECK(OP.is_in(sample, NT(1e-7)) == -1);
}


template <typename NT>
void test_order_hmc_standard_facets_contract()
{
    typedef typename Cartesian<NT>::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::MT MT;
    typedef typename OP_t::VT VT;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 37> RNG;
    typedef typename OrderPolytopeExactHMCWalk::
        template Walk<OP_t, RNG> Walk;

    Poset::RV relations{{0, 1}};
    Poset const poset(2, relations);
    OP_t body(poset);
    CHECK(body.has_standard_order_facets());
    CHECK(OP_t::from_reduced_relations(poset).has_standard_order_facets());

    OP_t normalized(body);
    unsigned long long const normalized_revision =
        normalized.geometry_revision();
    normalized.normalize();
    CHECK(normalized.has_standard_order_facets());
    CHECK(normalized.geometry_revision() == normalized_revision + 1);
    normalized.normalize();
    CHECK(normalized.geometry_revision() == normalized_revision + 1);

    OP_t shifted(body);
    unsigned long long const shifted_revision = shifted.geometry_revision();
    VT shift(2);
    shift << NT(0.1), NT(0.2);
    shifted.shift(shift);
    CHECK(shifted.has_standard_order_facets());
    CHECK(shifted.geometry_revision() == shifted_revision + 1);

    MT transform = MT::Identity(2, 2);
    transform(0, 0) = NT(2);
    OP_t transformed(body);
    unsigned long long const transformed_revision =
        transformed.geometry_revision();
    transformed.linear_transformIt(transform);
    CHECK_FALSE(transformed.has_standard_order_facets());
    CHECK(transformed.geometry_revision() == transformed_revision + 1);
    CHECK_FALSE(OP_t(transformed).has_standard_order_facets());
    OP_t assigned(body);
    unsigned long long const assigned_revision = assigned.geometry_revision();
    assigned = transformed;
    CHECK_FALSE(assigned.has_standard_order_facets());
    CHECK(assigned.geometry_revision() == assigned_revision + 1);

    Point point(2, {NT(0.25), NT(0.75)});
    RNG rng(2);
    CHECK_THROWS_AS(Walk(transformed, point, NT(0.5), rng),
                    std::invalid_argument);

    OP_t changed_after_construction(body);
    Walk walk(changed_after_construction, point, NT(0.5), rng);
    changed_after_construction.linear_transformIt(transform);
    CHECK_THROWS_AS(
        walk.apply(changed_after_construction, point, NT(0.5), 1, rng),
        std::runtime_error);

    OP_t changed_by_shift(body);
    Point shifted_point(2, {NT(0.25), NT(0.75)});
    RNG shifted_rng(2);
    Walk shifted_walk(changed_by_shift, shifted_point, NT(0.5), shifted_rng);
    changed_by_shift.shift(shift);
    CHECK_THROWS_AS(
        shifted_walk.apply(
            changed_by_shift, shifted_point, NT(0.5), 1, shifted_rng),
        std::runtime_error);

    VT target_shift(2);
    target_shift << NT(0.05), NT(0.1);
    OP_t changed_by_assignment(body);
    changed_by_assignment.shift(target_shift);
    CHECK(changed_by_assignment.geometry_revision() ==
          shifted.geometry_revision());
    Point assigned_point(2, {NT(0.25), NT(0.75)});
    RNG assigned_rng(2);
    Walk assigned_walk(
        changed_by_assignment, assigned_point, NT(0.5), assigned_rng);
    assigned_point = Point(2, {NT(0.3), NT(0.6)});
    Point const point_before_assignment(assigned_point);
    unsigned long long const revision_before_assignment =
        changed_by_assignment.geometry_revision();
    changed_by_assignment = shifted;
    CHECK(changed_by_assignment.geometry_revision() ==
          revision_before_assignment + 1);
    CHECK_THROWS_AS(
        assigned_walk.apply(
            changed_by_assignment, assigned_point, NT(0.5), 0,
            assigned_rng),
        std::runtime_error);
    CHECK((assigned_point.getCoefficients() -
           point_before_assignment.getCoefficients()).norm() ==
          doctest::Approx(NT(0)));
}


TEST_CASE("basics") { call_test_basics<double>(); }
TEST_CASE("line_intersect") { call_test_line_intersect<double>(); }
TEST_CASE("reflection") { call_test_reflection<double>(); }
TEST_CASE("vec_mult") { call_test_vec_mult<double>(); }
TEST_CASE("true_box_facets") { call_test_true_box_facets<double>(); }
TEST_CASE("true_relation_facets") { call_test_true_relation_facets<double>(); }
TEST_CASE("coord_intersect_normalized_shifted") {
    call_test_coord_intersect_normalized_shifted<double>();
}
TEST_CASE("order_polytope_ghmc") {
    test_order_hmc<double>();
}
TEST_CASE("order_hmc_tied_contacts") {
    test_order_hmc_tied_contacts<double>();
}
TEST_CASE("order_hmc_shared_contact") {
    test_order_hmc_shared_contact<double>();
}
TEST_CASE("order_hmc_reflection_rollback") {
    test_order_hmc_reflection_rollback<double>();
}
TEST_CASE("order_hmc_boundary_refresh") {
    test_order_hmc_boundary_refresh<double>();
}
TEST_CASE("order_hmc_tiny_shifted_crossing") {
    test_order_hmc_tiny_shifted_crossing<double>();
}
TEST_CASE("order_hmc_last_hit_expires_after_free_flight") {
    test_order_hmc_last_hit_expires_after_free_flight<double>();
}
TEST_CASE("order_hmc_shifted_containment") {
    test_order_hmc_shifted_containment<double>();
}
TEST_CASE("order_hmc_standard_facets_contract") {
    test_order_hmc_standard_facets_contract<double>();
}
