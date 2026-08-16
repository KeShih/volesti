// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "generators/boost_random_number_generator.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/linear_extensions_volume.hpp"

namespace {

typedef Cartesian<double> Kernel;
typedef Kernel::Point Point;
typedef OrderPolytope<Point> OP;
typedef OP::VT VT;

class ScriptedGaussianRNG {
public:
    ScriptedGaussianRNG(std::vector<double> normal_values,
                        std::vector<double> uniform_values)
        : _normals(std::move(normal_values)),
          _uniforms(std::move(uniform_values))
    {}

    double sample_ndist()
    {
        if (_normal_index >= _normals.size())
            throw std::runtime_error("scripted normal sequence exhausted");
        return _normals[_normal_index++];
    }

    double sample_urdist()
    {
        if (_uniform_index >= _uniforms.size())
            throw std::runtime_error("scripted uniform sequence exhausted");
        return _uniforms[_uniform_index++];
    }

private:
    std::vector<double> _normals, _uniforms;
    std::size_t _normal_index = 0, _uniform_index = 0;
};

Point make_point(std::initializer_list<double> values)
{
    return Point(static_cast<unsigned int>(values.size()),
                 std::vector<double>(values));
}

VT make_vector(std::initializer_list<double> values)
{
    VT result(values.size());
    unsigned int i = 0;
    for (double value : values) result(i++) = value;
    return result;
}

Poset make_poset(unsigned int n, Poset::RV relations)
{
    return Poset(n, relations);
}

double max_point_difference(Point const& lhs, Point const& rhs)
{
    REQUIRE(lhs.dimension() == rhs.dimension());
    double error = 0.0;
    for (int i = 0; i < lhs.dimension(); ++i)
        error = std::max(error, std::abs(lhs[i] - rhs[i]));
    return error;
}

double kinetic_energy(Point const& velocity, VT const& shape)
{
    double energy = 0.0;
    for (int i = 0; i < velocity.dimension(); ++i)
        energy += velocity[i] * velocity[i] / shape(i);
    return energy;
}

double max_original_violation(Point const& centered_point,
                              Point const& center,
                              Poset const& poset)
{
    double violation = 0.0;
    std::vector<double> x(centered_point.dimension());
    for (int i = 0; i < centered_point.dimension(); ++i) {
        x[i] = centered_point[i] + center[i];
        violation = std::max(violation, -x[i]);
        violation = std::max(violation, x[i] - 1.0);
    }
    Poset reduced = poset.transitive_reduction();
    for (unsigned int k = 0; k < reduced.num_relations(); ++k) {
        auto const relation = reduced.get_relation(k);
        violation = std::max(
            violation, x[relation.first] - x[relation.second]);
    }
    return std::max(0.0, violation);
}

OrderPolytopeVolumeReductionOptions forced_residual_options()
{
    OrderPolytopeVolumeReductionOptions options;
    options.enable_chain_antichain = false;
    options.enable_parallel_decomposition = false;
    options.enable_ordinal_decomposition = false;
    options.enable_small_exact_dp = false;
    options.exact_dp_max_n = 0;
    return options;
}

OrderPolytopeVolumeReductionOptions exact_dp_options()
{
    OrderPolytopeVolumeReductionOptions options;
    options.enable_chain_antichain = false;
    options.enable_parallel_decomposition = false;
    options.enable_ordinal_decomposition = false;
    options.enable_small_exact_dp = true;
    options.exact_dp_max_n = 20;
    return options;
}

} // namespace

TEST_CASE("diagonal_rounding_metric_validation")
{
    typedef DiagonalRoundingOrderPolytope<Point> RoundedBody;
    static_assert(!std::is_convertible<RoundedBody*, OP*>::value,
                  "rounded body must not expose mutable OrderPolytope base");
    static_assert(!std::is_copy_constructible<RoundedBody>::value,
                  "rounded body owns walk-bound cached geometry");
    static_assert(!std::is_move_constructible<RoundedBody>::value,
                  "rounded body address must remain stable while walking");
    static_assert(!std::is_copy_assignable<RoundedBody>::value,
                  "rounded body geometry must not be replaced after binding");
    static_assert(!std::is_move_assignable<RoundedBody>::value,
                  "rounded body geometry must not move after binding");
    CHECK_NOTHROW(OrderPolytopeGaussianHamiltonianMonteCarloExactWalk(1.0));
    CHECK_NOTHROW(OrderPolytopeGaussianHamiltonianMonteCarloExactWalk(1.0, 2u));
    CHECK_NOTHROW(OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk(1.0));
    CHECK_NOTHROW(OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk(1.0, 2u));

    Poset chain = make_poset(3, Poset::RV{{0, 1}, {1, 2}});
    Point center = make_point({0.2, 0.5, 0.8});
    VT shape = make_vector({0.25, 4.0, 9.0});
    CHECK_NOTHROW(OrderPolytopeDiagonalRounding<Point>(chain, center, shape));

    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.2, 0.5}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center, make_vector({1.0, 1.0})), std::invalid_argument);

    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const inf = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.2, nan, 0.8}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.2, inf, 0.8}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center, make_vector({0.25, 0.0, 9.0})), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center, make_vector({0.25, -1.0, 9.0})), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center, make_vector({0.25, nan, 9.0})), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center, make_vector({0.25, inf, 9.0})), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center,
        make_vector({std::numeric_limits<double>::denorm_min(), 1.0, 1.0})),
        std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, center,
        make_vector({std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(), 1.0})),
        std::invalid_argument);

    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.0, 0.5, 0.8}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.2, 0.5, 1.0}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.2, 0.2, 0.8}), shape), std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
        chain, make_point({0.6, 0.5, 0.8}), shape), std::invalid_argument);

    // A cached metric from another same-dimensional body must be rebound to
    // the actual true facets before a rounded walk or cooling schedule starts.
    Poset antichain2 = make_poset(2, Poset::RV{});
    Poset chain2 = make_poset(2, Poset::RV{{0, 1}});
    OP antichain_body(antichain2), chain_body(chain2);
    OrderPolytopeDiagonalRounding<Point> boundary_metric(
        antichain_body, make_point({0.5, 0.5}), make_vector({1.0, 1.0}));
    CHECK_THROWS_AS(
        DiagonalRoundingOrderPolytope<Point>(chain_body, boundary_metric),
        std::invalid_argument);

    OrderPolytopeDiagonalRounding<Point> compatible_raw_metric(
        antichain_body, make_point({0.49, 0.51}), make_vector({1.0, 1.0}));
    DiagonalRoundingOrderPolytope<Point> rebound_body(
        chain_body, compatible_raw_metric);
    auto const& rebound_distances =
        rebound_body.rounding_facet_metric_distances();
    REQUIRE(rebound_distances.size() == 3);
    CHECK(rebound_distances[2] ==
          doctest::Approx(0.02 / std::sqrt(2.0)).epsilon(1e-14));

    std::vector<double> a_values;
    std::vector<double> const small_a_distances{9592.4, 9592.4};
    order_polytope_diagonal_cooling_detail::
        get_first_gaussian_from_metric_distances(
            small_a_distances, 0.1, 0.01, a_values);
    REQUIRE(a_values.size() == 1);
    double tail_sum = 0.0;
    double const pi = std::acos(-1.0);
    for (double distance : small_a_distances)
        tail_sum += std::exp(-a_values[0] * distance * distance) /
                    (2.0 * distance * std::sqrt(pi * a_values[0]));
    CHECK(tail_sum <= 0.001);

    a_values.clear();
    CHECK_NOTHROW(order_polytope_diagonal_cooling_detail::
        get_first_gaussian_from_metric_distances(
            std::vector<double>{1e-100}, 0.1, 0.1, a_values));
    REQUIRE(a_values.size() == 1);
    CHECK(std::isfinite(a_values[0]));
    a_values.clear();
    CHECK_THROWS_AS(order_polytope_diagonal_cooling_detail::
        get_first_gaussian_from_metric_distances(
            std::vector<double>{1e-154}, 0.1, 0.1, a_values),
        std::runtime_error);
    CHECK(order_polytope_diagonal_cooling_detail::
              checked_schedule_sample_count(0.1, 0.3) == 556u);
    CHECK_THROWS_AS(order_polytope_diagonal_cooling_detail::
        checked_schedule_sample_count(0.1, 1e-8), std::invalid_argument);
    float const float_error = 150.0f /
        (0.9f * 4294967296.0f);
    CHECK_THROWS_AS(order_polytope_diagonal_cooling_detail::
        checked_schedule_sample_count(0.1f, float_error),
        std::invalid_argument);
}

TEST_CASE("diagonal_rounding_quadratic_distances")
{
    Poset chain = make_poset(3, Poset::RV{{0, 1}, {1, 2}});
    Point center = make_point({0.2, 0.5, 0.8});
    VT shape = make_vector({0.25, 4.0, 9.0});
    OrderPolytopeDiagonalRounding<Point> metric(chain, center, shape);

    CHECK(metric.quadratic_statistic(center) == doctest::Approx(0.0));
    CHECK(metric.quadratic_statistic(make_point({0.7, 0.1, 0.2})) ==
          doctest::Approx(1.08).epsilon(1e-14));
    Point centered = make_point({0.5, -0.4, -0.6});
    double const rounded_weight =
        order_polytope_diagonal_cooling_detail::ratio_weight(
            centered, 2.0, 3.0, metric);
    CHECK(rounded_weight == doctest::Approx(std::exp(1.08)).epsilon(1e-14));
    CHECK(std::abs(rounded_weight - std::exp(0.77)) > 0.5);

    auto const& distances = metric.facet_metric_distances();
    REQUIRE(distances.size() == 4);
    CHECK(distances[0] == doctest::Approx(0.4).epsilon(1e-14));
    CHECK(distances[1] == doctest::Approx(0.2 / 3.0).epsilon(1e-14));
    CHECK(distances[2] == doctest::Approx(0.3 / std::sqrt(4.25)).epsilon(1e-14));
    CHECK(distances[3] == doctest::Approx(0.3 / std::sqrt(13.0)).epsilon(1e-14));

    OP normalized(chain);
    normalized.normalize();
    OrderPolytopeDiagonalRounding<Point> normalized_metric(
        normalized, center, shape);
    auto const& normalized_distances =
        normalized_metric.facet_metric_distances();
    REQUIRE(normalized_distances.size() == distances.size());
    for (unsigned int i = 0; i < distances.size(); ++i)
        CHECK(normalized_distances[i] ==
              doctest::Approx(distances[i]).epsilon(1e-14));
}

TEST_CASE("diagonal_rounding_normalizer")
{
    Poset antichain = make_poset(2, Poset::RV{});
    OrderPolytopeDiagonalRounding<Point> metric(
        antichain, make_point({0.5, 0.5}), make_vector({4.0, 9.0}));
    double const pi = std::acos(-1.0);
    double const log_normalizer = metric.gaussian_log_normalizer(pi);
    CHECK(log_normalizer == doctest::Approx(std::log(6.0)).epsilon(1e-14));
    CHECK(std::exp(log_normalizer) == doctest::Approx(6.0).epsilon(1e-14));
    CHECK(std::abs(log_normalizer - std::log(1.0)) > 1.0);
    CHECK(std::abs(log_normalizer - std::log(36.0)) > 1.0);
    CHECK(std::abs(log_normalizer - std::log(1.0 / 6.0)) > 1.0);
    CHECK(std::isfinite(metric.gaussian_log_normalizer(
        std::numeric_limits<double>::min())));
}

TEST_CASE("diagonal_rounding_reflection")
{
    VT const shape = make_vector({0.25, 4.0, 9.0});
    VT velocity = make_vector({1.0, -1.0, 0.5});
    VT const original = velocity;
    double const energy_before =
        (velocity.array().square() / shape.array()).sum();
    double const normal_before = velocity(0) - velocity(1);

    OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
        velocity, 0, 1, shape);
    CHECK(velocity(0) ==
          doctest::Approx(0.7647058823529411).epsilon(1e-14));
    CHECK(velocity(1) ==
          doctest::Approx(2.7647058823529411).epsilon(1e-14));
    CHECK(velocity(2) == original(2));
    CHECK(velocity(0) - velocity(1) ==
          doctest::Approx(-normal_before).epsilon(1e-14));
    double const energy_after =
        (velocity.array().square() / shape.array()).sum();
    CHECK(energy_after == doctest::Approx(energy_before).epsilon(1e-14));

    OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
        velocity, 0, 1, shape);
    for (int i = 0; i < velocity.size(); ++i)
        CHECK(velocity(i) == doctest::Approx(original(i)).epsilon(1e-14));

    velocity = original;
    CHECK_THROWS_AS(
        OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
            velocity, 0, 1, make_vector({-1.0, 2.0, 9.0})),
        std::invalid_argument);

    // Exercise the production bound-facet reflection with non-identity D.
    Poset antichain = make_poset(1, Poset::RV{});
    OP original_body(antichain);
    OrderPolytopeDiagonalRounding<Point> metric(
        original_body, make_point({0.5}), make_vector({9.0}));
    DiagonalRoundingOrderPolytope<Point> body(original_body, metric);
    typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk Policy;
    typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>,
                         ScriptedGaussianRNG> Walk;
    Point point(1);
    // Constructor v=0,T=0; apply physical v=1,T=.8.
    ScriptedGaussianRNG rng({0.0, 1.0 / 3.0}, {0.0, 0.8});
    Policy::parameters parameters(1.0, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);
    auto const& trace = walk.get_testing_trace();
    REQUIRE(trace.captured_contact);
    CHECK(trace.contact_facets.size() == 1);
    CHECK(trace.post_reflection_velocity[0] ==
          doctest::Approx(-trace.pre_reflection_velocity[0]).epsilon(1e-14));
    CHECK(trace.post_reflection_velocity[0] *
              trace.post_reflection_velocity[0] / 9.0 ==
          doctest::Approx(trace.pre_reflection_velocity[0] *
                          trace.pre_reflection_velocity[0] / 9.0).epsilon(1e-14));
    double twice_reflected = -trace.post_reflection_velocity[0];
    CHECK(twice_reflected ==
          doctest::Approx(trace.pre_reflection_velocity[0]).epsilon(1e-14));
    std::cout << std::setprecision(17)
              << "DIAGONAL_ROUNDING_MAX_REFLECTION_ENERGY_ERROR="
              << std::abs(energy_after - energy_before) << '\n';
}

TEST_CASE("diagonal_rounding_velocity_refresh")
{
    Poset antichain = make_poset(3, Poset::RV{});
    OP original(antichain);
    OrderPolytopeDiagonalRounding<Point> metric(
        original, make_point({0.5, 0.5, 0.5}),
        make_vector({0.25, 4.0, 9.0}));
    DiagonalRoundingOrderPolytope<Point> body(original, metric);
    typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk Policy;
    typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>, ScriptedGaussianRNG> Walk;

    Point point(3);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, 0.0, 1.0, -2.0, 0.5}, {0.0, 0.0});
    Policy::parameters parameters(1.0, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);
    auto const& trace = walk.get_testing_trace();
    CHECK(trace.final_velocity[0] == doctest::Approx(0.5));
    CHECK(trace.final_velocity[1] == doctest::Approx(-4.0));
    CHECK(trace.final_velocity[2] == doctest::Approx(1.5));
    CHECK(max_point_difference(point, Point(3)) == 0.0);

    // Exercise the physical-time/angle conversion away from omega=1.  With
    // a=2, omega=2; the upper wall is hit at theta=pi/6 and the remaining
    // trajectory is propagated with the reflected physical velocity.
    Poset singleton = make_poset(1, Poset::RV{});
    OP singleton_original(singleton);
    OrderPolytopeDiagonalRounding<Point> nonunit_metric(
        singleton_original, make_point({0.5}), make_vector({4.0}));
    DiagonalRoundingOrderPolytope<Point> nonunit_body(
        singleton_original, nonunit_metric);
    typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>,
                         ScriptedGaussianRNG> NonunitWalk;
    Point nonunit_point(1);
    ScriptedGaussianRNG nonunit_rng({0.0, 1.0}, {0.0, 0.35});
    NonunitWalk nonunit_walk(
        nonunit_body, nonunit_point, 2.0, nonunit_rng, parameters);
    nonunit_walk.apply(
        nonunit_body, nonunit_point, 2.0, 1, nonunit_rng);
    auto const& nonunit_trace = nonunit_walk.get_testing_trace();
    REQUIRE(nonunit_trace.captured_contact);
    double const hit_angle = std::acos(-1.0) / 6.0;
    double const remaining_angle = 0.7 - hit_angle;
    double const post_velocity = -std::sqrt(3.0);
    double const expected_position =
        0.5 * std::cos(remaining_angle) +
        (post_velocity / 2.0) * std::sin(remaining_angle);
    double const expected_velocity =
        -std::sin(remaining_angle) +
        post_velocity * std::cos(remaining_angle);
    CHECK(nonunit_trace.hit_angle ==
          doctest::Approx(hit_angle).epsilon(1e-13));
    CHECK(nonunit_trace.collision_position[0] ==
          doctest::Approx(0.5).epsilon(1e-13));
    CHECK(nonunit_trace.post_reflection_velocity[0] ==
          doctest::Approx(post_velocity).epsilon(1e-13));
    CHECK(nonunit_point[0] ==
          doctest::Approx(expected_position).epsilon(1e-13));
    CHECK(nonunit_trace.final_velocity[0] ==
          doctest::Approx(expected_velocity).epsilon(1e-13));
    CHECK(nonunit_body.is_in(nonunit_point, 1e-12) == -1);
}

TEST_CASE("diagonal_rounding_identity_equivalence")
{
    Poset antichain = make_poset(2, Poset::RV{});
    OP original(antichain);
    Point center = make_point({0.5, 0.5});
    VT identity = make_vector({1.0, 1.0});
    OrderPolytopeDiagonalRounding<Point> metric(original, center, identity);

    OP spherical_body(original);
    spherical_body.normalize();
    spherical_body.shift(center.getCoefficients());
    DiagonalRoundingOrderPolytope<Point> diagonal_body(original, metric);

    typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk SphericalPolicy;
    typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk DiagonalPolicy;
    typedef SphericalPolicy::Walk<OP, ScriptedGaussianRNG> SphericalWalk;
    typedef DiagonalPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                 ScriptedGaussianRNG> DiagonalWalk;

    Point spherical_point(2), diagonal_point(2);
    ScriptedGaussianRNG spherical_rng(
        {0.0, 0.0, 1.0, 1.0}, {0.0, 0.8});
    ScriptedGaussianRNG diagonal_rng(
        {0.0, 0.0, 1.0, 1.0}, {0.0, 0.8});
    SphericalPolicy::parameters spherical_parameters(1.0, true, 100, true);
    DiagonalPolicy::parameters diagonal_parameters(1.0, true, 100, true);
    SphericalWalk spherical_walk(
        spherical_body, spherical_point, 0.5, spherical_rng,
        spherical_parameters);
    DiagonalWalk diagonal_walk(
        diagonal_body, diagonal_point, 0.5, diagonal_rng,
        diagonal_parameters);
    spherical_walk.apply(spherical_body, spherical_point, 0.5, 1, spherical_rng);
    diagonal_walk.apply(diagonal_body, diagonal_point, 0.5, 1, diagonal_rng);

    auto const& spherical_trace = spherical_walk.get_testing_trace();
    auto const& diagonal_trace = diagonal_walk.get_testing_trace();
    REQUIRE(spherical_trace.captured_contact);
    REQUIRE(diagonal_trace.captured_contact);
    CHECK(spherical_trace.contact_facets == diagonal_trace.contact_facets);
    CHECK(spherical_trace.contact_facets.size() == 2);
    CHECK(spherical_trace.hit_angle ==
          doctest::Approx(std::acos(-1.0) / 6.0).epsilon(1e-13));

    double max_error = 0.0;
    max_error = std::max(max_error,
        std::abs(spherical_trace.hit_angle - diagonal_trace.hit_angle));
    max_error = std::max(max_error, max_point_difference(
        spherical_trace.collision_position,
        diagonal_trace.collision_position));
    max_error = std::max(max_error, max_point_difference(
        spherical_trace.post_reflection_velocity,
        diagonal_trace.post_reflection_velocity));
    max_error = std::max(max_error,
        max_point_difference(spherical_point, diagonal_point));
    max_error = std::max(max_error, max_point_difference(
        spherical_trace.final_velocity, diagonal_trace.final_velocity));
    CHECK(max_error < 1e-13);
    CHECK(spherical_body.is_in(spherical_point, 1e-12) == -1);
    CHECK(diagonal_body.is_in(diagonal_point, 1e-12) == -1);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
    CHECK(spherical_trace.full_scan_checked);
    CHECK(diagonal_trace.full_scan_checked);
    CHECK(spherical_trace.full_scan_agreed);
    CHECK(diagonal_trace.full_scan_agreed);
#endif

    // Also exercise the compile-time cover-reflection dispatch at D=I; equal
    // diagonal entries must reduce exactly to the established velocity swap.
    {
        Poset chain = make_poset(2, Poset::RV{{0, 1}});
        OP chain_original(chain);
        Point chain_center = make_point({0.25, 0.75});
        OrderPolytopeDiagonalRounding<Point> chain_metric(
            chain_original, chain_center, identity);
        OP chain_spherical_body(chain_original);
        chain_spherical_body.normalize();
        chain_spherical_body.shift(chain_center.getCoefficients());
        DiagonalRoundingOrderPolytope<Point> chain_diagonal_body(
            chain_original, chain_metric);

        Point spherical_cover_point(2), diagonal_cover_point(2);
        ScriptedGaussianRNG spherical_cover_rng(
            {0.0, 0.0, 1.0, -1.0}, {0.0, 0.5});
        ScriptedGaussianRNG diagonal_cover_rng(
            {0.0, 0.0, 1.0, -1.0}, {0.0, 0.5});
        SphericalWalk spherical_cover_walk(
            chain_spherical_body, spherical_cover_point, 0.5,
            spherical_cover_rng, spherical_parameters);
        DiagonalWalk diagonal_cover_walk(
            chain_diagonal_body, diagonal_cover_point, 0.5,
            diagonal_cover_rng, diagonal_parameters);
        spherical_cover_walk.apply(
            chain_spherical_body, spherical_cover_point, 0.5, 1,
            spherical_cover_rng);
        diagonal_cover_walk.apply(
            chain_diagonal_body, diagonal_cover_point, 0.5, 1,
            diagonal_cover_rng);

        auto const& spherical_cover_trace =
            spherical_cover_walk.get_testing_trace();
        auto const& diagonal_cover_trace =
            diagonal_cover_walk.get_testing_trace();
        REQUIRE(spherical_cover_trace.captured_contact);
        REQUIRE(diagonal_cover_trace.captured_contact);
        CHECK(spherical_cover_trace.contact_facets ==
              diagonal_cover_trace.contact_facets);
        CHECK(spherical_cover_trace.contact_facets.size() == 1);
        max_error = std::max(max_error, std::abs(
            spherical_cover_trace.hit_angle - diagonal_cover_trace.hit_angle));
        max_error = std::max(max_error, max_point_difference(
            spherical_cover_trace.post_reflection_velocity,
            diagonal_cover_trace.post_reflection_velocity));
        max_error = std::max(max_error, max_point_difference(
            spherical_cover_point, diagonal_cover_point));
        max_error = std::max(max_error, max_point_difference(
            spherical_cover_trace.final_velocity,
            diagonal_cover_trace.final_velocity));
        CHECK(max_error < 1e-13);
        CHECK(chain_spherical_body.is_in(spherical_cover_point, 1e-12) == -1);
        CHECK(chain_diagonal_body.is_in(diagonal_cover_point, 1e-12) == -1);
    }
    std::cout << std::setprecision(17)
              << "DIAGONAL_ROUNDING_MAX_IDENTITY_TRAJECTORY_ERROR="
              << max_error << '\n';
}

TEST_CASE("diagonal_rounding_event_queue_full_scan")
{
    Poset poset = make_poset(4, Poset::RV{{0, 1}, {2, 3}});
    OP original(poset);
    Point center = make_point({0.25, 0.75, 0.25, 0.75});
    std::vector<VT> shapes{
        make_vector({0.25, 4.0, 1.0, 9.0}),
        make_vector({4.0, 0.25, 9.0, 1.0}),
        make_vector({0.5, 2.0, 3.0, 7.0})};
    double maximum_full_scan_error = 0.0;
    double maximum_reflection_error = 0.0;
    double maximum_collision_error = 0.0;
    double maximum_reflected_state_error = 0.0;
    double maximum_final_state_error = 0.0;

    for (VT const& shape : shapes) {
        OrderPolytopeDiagonalRounding<Point> metric(original, center, shape);
        DiagonalRoundingOrderPolytope<Point> body(original, metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk Policy;
        typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>,
                             ScriptedGaussianRNG> Walk;

        std::vector<double> normals(4, 0.0);
        std::vector<double> physical_velocity{1.0, -1.0, 1.0, -1.0};
        for (unsigned int i = 0; i < 4; ++i)
            normals.push_back(physical_velocity[i] / std::sqrt(shape(i)));
        double const final_angle = 0.3;
        ScriptedGaussianRNG rng(normals, {0.0, final_angle});
        Point point(4);
        Policy::parameters parameters(1.0, true, 100, true);
        Walk walk(body, point, 0.5, rng, parameters);
        walk.apply(body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();

        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets.size() == 2);
        CHECK(trace.contact_facets == std::vector<unsigned int>({4, 5}));
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(body.is_in(point, 1e-10) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);

        double const hit_angle = std::asin(0.25);
        CHECK(trace.hit_angle == doctest::Approx(hit_angle).epsilon(1e-13));
        Point expected_collision = make_point({0.25, -0.25, 0.25, -0.25});
        double const collision_error = max_point_difference(
            trace.collision_position, expected_collision);
        maximum_collision_error = std::max(
            maximum_collision_error, collision_error);
        CHECK(collision_error < 1e-13);
        VT expected_post_velocity(4);
        double const hit_cos = std::cos(hit_angle);
        for (unsigned int i = 0; i < 4; ++i)
            expected_post_velocity(i) = physical_velocity[i] * hit_cos;
        OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
            expected_post_velocity, 0, 1, shape);
        OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
            expected_post_velocity, 2, 3, shape);
        Point expected_post_point(expected_post_velocity);
        double const reflected_state_error = max_point_difference(
            trace.post_reflection_velocity, expected_post_point);
        maximum_reflected_state_error = std::max(
            maximum_reflected_state_error, reflected_state_error);
        CHECK(reflected_state_error < 1e-13);

        double const remaining = final_angle - hit_angle;
        VT expected_final_position(4), expected_final_velocity(4);
        for (unsigned int i = 0; i < 4; ++i) {
            expected_final_position(i) = expected_collision[i] * std::cos(remaining)
                                       + expected_post_velocity(i) * std::sin(remaining);
            expected_final_velocity(i) = -expected_collision[i] * std::sin(remaining)
                                       + expected_post_velocity(i) * std::cos(remaining);
        }
        double const final_position_error = max_point_difference(
            point, Point(expected_final_position));
        double const final_velocity_error = max_point_difference(
            trace.final_velocity, Point(expected_final_velocity));
        maximum_final_state_error = std::max(
            maximum_final_state_error,
            std::max(final_position_error, final_velocity_error));
        CHECK(final_position_error < 1e-12);
        CHECK(final_velocity_error < 1e-12);

        double const energy_before = kinetic_energy(
            trace.pre_reflection_velocity, shape);
        double const energy_after = kinetic_energy(
            trace.post_reflection_velocity, shape);
        maximum_reflection_error = std::max(
            maximum_reflection_error,
            std::abs(energy_after - energy_before));
        CHECK(energy_after == doctest::Approx(energy_before).epsilon(1e-12));
        CHECK(trace.post_reflection_velocity[0] -
              trace.post_reflection_velocity[1] ==
              doctest::Approx(-(trace.pre_reflection_velocity[0] -
                                trace.pre_reflection_velocity[1])).epsilon(1e-12));
        CHECK(trace.post_reflection_velocity[2] -
              trace.post_reflection_velocity[3] ==
              doctest::Approx(-(trace.pre_reflection_velocity[2] -
                                trace.pre_reflection_velocity[3])).epsilon(1e-12));
    }

    // A later event on an untouched component must remain queued after the
    // first batch's incident-only recomputation.
    {
        Poset sequential_poset = make_poset(
            6, Poset::RV{{0, 1}, {2, 3}, {4, 5}});
        OP sequential_original(sequential_poset);
        Point sequential_center =
            make_point({0.25, 0.75, 0.25, 0.75, 0.1, 0.9});
        VT sequential_shape =
            make_vector({0.8, 1.2, 0.9, 1.1, 0.7, 1.3});
        OrderPolytopeDiagonalRounding<Point> sequential_metric(
            sequential_original, sequential_center, sequential_shape);
        DiagonalRoundingOrderPolytope<Point> sequential_body(
            sequential_original, sequential_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SequentialPolicy;
        typedef SequentialPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                       ScriptedGaussianRNG> SequentialWalk;

        std::vector<double> normals(6, 0.0);
        for (unsigned int i = 0; i < 6; ++i) {
            double const physical_velocity = i % 2 == 0 ? 1.0 : -1.0;
            normals.push_back(
                physical_velocity / std::sqrt(sequential_shape(i)));
        }
        double const final_angle = 0.5;
        ScriptedGaussianRNG rng(normals, {0.0, final_angle});
        Point point(6);
        SequentialPolicy::parameters parameters(1.0, true, 100, true);
        SequentialWalk walk(
            sequential_body, point, 0.5, rng, parameters);
        walk.apply(sequential_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();

        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({6, 7}));
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 2);
#endif
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);

        VT expected_final_position(6), expected_final_velocity(6);
        for (unsigned int pair = 0; pair < 3; ++pair) {
            unsigned int const u = 2 * pair, v = u + 1;
            double const hit_sine = pair < 2 ? 0.25 : 0.4;
            double const hit_angle = std::asin(hit_sine);
            VT reflected_velocity(2);
            double const hit_cosine = std::cos(hit_angle);
            reflected_velocity(0) = hit_cosine;
            reflected_velocity(1) = -hit_cosine;
            VT pair_shape(2);
            pair_shape << sequential_shape(u), sequential_shape(v);
            OrderPolytopeDiagonalRounding<Point>::reflect_cover_velocity(
                reflected_velocity, 0, 1, pair_shape);
            double const remaining = final_angle - hit_angle;
            expected_final_position(u) =
                hit_sine * std::cos(remaining) +
                reflected_velocity(0) * std::sin(remaining);
            expected_final_position(v) =
                -hit_sine * std::cos(remaining) +
                reflected_velocity(1) * std::sin(remaining);
            expected_final_velocity(u) =
                -hit_sine * std::sin(remaining) +
                reflected_velocity(0) * std::cos(remaining);
            expected_final_velocity(v) =
                hit_sine * std::sin(remaining) +
                reflected_velocity(1) * std::cos(remaining);
        }
        double const final_position_error = max_point_difference(
            point, Point(expected_final_position));
        double const final_velocity_error = max_point_difference(
            trace.final_velocity, Point(expected_final_velocity));
        maximum_final_state_error = std::max(
            maximum_final_state_error,
            std::max(final_position_error, final_velocity_error));
        CHECK(final_position_error < 1e-12);
        CHECK(final_velocity_error < 1e-12);
        CHECK(sequential_body.is_in(point, 1e-10) == -1);
    }

    // Both members of a simultaneous batch must contribute their incident
    // facets to recomputation.  The upper-wall batch is followed by a
    // lower-wall batch on the same two coordinates within one quarter-period.
    {
        Poset antichain = make_poset(2, Poset::RV{});
        OP original_antichain(antichain);
        Point antichain_center = make_point({0.5, 0.5});
        VT antichain_shape = make_vector({0.25, 4.0});
        OrderPolytopeDiagonalRounding<Point> antichain_metric(
            original_antichain, antichain_center, antichain_shape);
        DiagonalRoundingOrderPolytope<Point> antichain_body(
            original_antichain, antichain_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            AntichainPolicy;
        typedef AntichainPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                      ScriptedGaussianRNG> AntichainWalk;

        double const physical_velocity = 2.0;
        ScriptedGaussianRNG rng(
            {0.0, 0.0,
             physical_velocity / std::sqrt(antichain_shape(0)),
             physical_velocity / std::sqrt(antichain_shape(1))},
            {0.0, 1.0});
        Point point(2);
        AntichainPolicy::parameters parameters(1.0, true, 100, true);
        AntichainWalk walk(
            antichain_body, point, 0.5, rng, parameters);
        walk.apply(antichain_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();

        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({2, 3}));
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 2);
#endif

        double const first_hit_angle = std::asin(0.25);
        double const remaining = 1.0 - 3.0 * first_hit_angle;
        double const wall_velocity = std::sqrt(3.75);
        double const expected_position =
            -0.5 * std::cos(remaining) +
            wall_velocity * std::sin(remaining);
        double const expected_velocity =
            0.5 * std::sin(remaining) +
            wall_velocity * std::cos(remaining);
        CHECK(point[0] == doctest::Approx(expected_position).epsilon(1e-12));
        CHECK(point[1] == doctest::Approx(expected_position).epsilon(1e-12));
        CHECK(trace.final_velocity[0] ==
              doctest::Approx(expected_velocity).epsilon(1e-12));
        CHECK(trace.final_velocity[1] ==
              doctest::Approx(expected_velocity).epsilon(1e-12));
        CHECK(antichain_body.is_in(point, 1e-10) == -1);
        CHECK(max_original_violation(point, antichain_center, antichain) <
              1e-12);
    }

    // Row normalization makes the event oracle invariant to a tiny original
    // coordinate scale.  The rounded lower-wall distance is 0.1, so this is a
    // well-scaled metric problem despite c and D being small in x coordinates.
    {
        Poset tiny_poset = make_poset(1, Poset::RV{});
        OP tiny_original(tiny_poset);
        Point tiny_center = make_point({1e-16});
        VT tiny_shape = make_vector({1e-30});
        OrderPolytopeDiagonalRounding<Point> tiny_metric(
            tiny_original, tiny_center, tiny_shape);
        DiagonalRoundingOrderPolytope<Point> tiny_body(
            tiny_original, tiny_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            TinyPolicy;
        typedef TinyPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                 ScriptedGaussianRNG> TinyWalk;
        Point point(1);
        ScriptedGaussianRNG rng({0.0, -1.0}, {0.0, 0.2});
        TinyPolicy::parameters parameters(1.0, true, 100, true);
        TinyWalk walk(tiny_body, point, 0.5, rng, parameters);
        walk.apply(tiny_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();

        double const hit_angle = std::asin(0.1);
        double const remaining = 0.2 - hit_angle;
        double const post_velocity = std::sqrt(0.99) * 1e-15;
        double const expected_position =
            -1e-16 * std::cos(remaining) +
            post_velocity * std::sin(remaining);
        double const expected_velocity =
            1e-16 * std::sin(remaining) +
            post_velocity * std::cos(remaining);
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({0}));
        CHECK(std::abs(trace.hit_angle - hit_angle) < 1e-13);
        CHECK(std::abs(trace.collision_position[0] + 1e-16) < 1e-28);
        CHECK(std::abs(trace.post_reflection_velocity[0] - post_velocity) < 1e-27);
        CHECK(std::abs(point[0] - expected_position) < 1e-27);
        CHECK(std::abs(trace.final_velocity[0] - expected_velocity) < 1e-27);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(tiny_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // The same scale-invariance check for a cover row.  Subtracting the two
    // coordinates before the 1/slack row scaling avoids catastrophic
    // cancellation when the strictly feasible center is close to the cover.
    {
        Poset tiny_cover_poset = make_poset(2, Poset::RV{{0, 1}});
        OP tiny_cover_original(tiny_cover_poset);
        Point tiny_cover_center = make_point({0.5, std::nextafter(0.5, 1.0)});
        VT tiny_cover_shape = make_vector({1e-30, 1e-30});
        OrderPolytopeDiagonalRounding<Point> tiny_cover_metric(
            tiny_cover_original, tiny_cover_center, tiny_cover_shape);
        DiagonalRoundingOrderPolytope<Point> tiny_cover_body(
            tiny_cover_original, tiny_cover_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            TinyCoverPolicy;
        typedef TinyCoverPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                      ScriptedGaussianRNG> TinyCoverWalk;
        Point point(2);
        ScriptedGaussianRNG rng({0.0, 0.0, 1.0, -1.0}, {0.0, 0.2});
        TinyCoverPolicy::parameters parameters(1.0, true, 100, true);
        TinyCoverWalk walk(
            tiny_cover_body, point, 0.5, rng, parameters);
        walk.apply(tiny_cover_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();

        double const slack = tiny_cover_center[1] - tiny_cover_center[0];
        double const hit_angle = std::asin(slack / 2e-15);
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({2}));
        CHECK(std::abs(trace.hit_angle - hit_angle) < 1e-13);
        CHECK(std::abs(trace.collision_position[0] - slack / 2.0) < 1e-28);
        CHECK(std::abs(trace.collision_position[1] + slack / 2.0) < 1e-28);
        CHECK(trace.post_reflection_velocity[0] < 0.0);
        CHECK(trace.post_reflection_velocity[1] > 0.0);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(tiny_cover_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // Two distinct wall hits can fall inside the finite-precision heap-key
    // tie window.  They are not an exact common root, so the diagonal path
    // must fail before reflection rather than guessing simultaneous or
    // sequential semantics at working precision.
    {
        Poset close_poset = make_poset(2, Poset::RV{});
        OP close_original(close_poset);
        Point close_center = make_point({0.5, 0.5});
        VT close_shape = make_vector({1.0, 1.0});
        OrderPolytopeDiagonalRounding<Point> close_metric(
            close_original, close_center, close_shape);
        DiagonalRoundingOrderPolytope<Point> close_body(
            close_original, close_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            ClosePolicy;
        typedef ClosePolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> CloseWalk;
        double const first_angle = std::acos(-1.0) / 6.0;
        double const second_angle = first_angle + 5e-15;
        double const normal0 = 0.5 / std::sin(second_angle);
        double const normal1 = 0.5 / std::sin(first_angle);
        Point initial(2);
        Point point = initial;
        ScriptedGaussianRNG rng(
            {0.0, 0.0, normal0, normal1}, {0.0, 0.8});
        ClosePolicy::parameters parameters(1.0, true, 100, true);
        CloseWalk walk(close_body, point, 0.5, rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(close_body, point, 0.5, 1, rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(close_body.is_in(point) == -1);
    }

    // The independent full scan must resolve a genuine near-tangent pair,
    // rather than collapsing the two roots with the production value
    // tolerance.  Here the first upper-wall root is just before pi/2.
    {
        Poset tangent_poset = make_poset(1, Poset::RV{});
        OP tangent_original(tangent_poset);
        Point tangent_center = make_point({0.5});
        VT tangent_shape = make_vector({1.0});
        OrderPolytopeDiagonalRounding<Point> tangent_metric(
            tangent_original, tangent_center, tangent_shape);
        DiagonalRoundingOrderPolytope<Point> tangent_body(
            tangent_original, tangent_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            TangentPolicy;
        typedef TangentPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                    ScriptedGaussianRNG> TangentWalk;
        double const B = 1.0 + 1e-13;
        double const end_angle = std::acos(-1.0) / 2.0;
        Point point(1);
        ScriptedGaussianRNG rng({0.0, 0.5 * B}, {0.0, 1.0});
        TangentPolicy::parameters parameters(end_angle, true, 100, true);
        TangentWalk walk(tangent_body, point, 0.5, rng, parameters);
        walk.apply(tangent_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();
        double const expected_hit = std::asin(1.0 / B);
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({1}));
        CHECK(std::abs(trace.hit_angle - expected_hit) < 1e-12);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(tangent_body.is_in(point, 1e-12) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // hypot can round a representable two-root equation to an apparent
    // tangent.  The machine-scale amplitude band leaves that state unresolved;
    // the leg must fail before contact and restore its starting point.
    {
        Poset rounded_tangent_poset = make_poset(1, Poset::RV{});
        OP rounded_tangent_original(rounded_tangent_poset);
        OrderPolytopeDiagonalRounding<Point> rounded_tangent_metric(
            rounded_tangent_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> rounded_tangent_body(
            rounded_tangent_original, rounded_tangent_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            RoundedTangentPolicy;
        typedef RoundedTangentPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                           ScriptedGaussianRNG> RoundedTangentWalk;
        double const A = 0x1.6a09e667f3bb9p-1;
        double const B = 0x1.6a09e667f3be1p-1;
        double const end_angle = std::atan2(B, A) + 3e-8;
        Point initial = make_point({A / 2.0});
        Point point = initial;
        ScriptedGaussianRNG rng({0.0, B / 2.0}, {0.0, 1.0});
        RoundedTangentPolicy::parameters parameters(
            end_angle, true, 100, true);
        RoundedTangentWalk walk(
            rounded_tangent_body, point, 0.5, rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(rounded_tangent_body, point, 0.5, 1, rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(rounded_tangent_body.is_in(point) == -1);
    }

    // The symmetric case where hypot rounds just below the facet radius is
    // equally unresolved: exact binary64 coefficients still have two roots,
    // so it must not be pruned as a definite no-hit trajectory.
    {
        Poset below_tangent_poset = make_poset(1, Poset::RV{});
        OP below_tangent_original(below_tangent_poset);
        OrderPolytopeDiagonalRounding<Point> below_tangent_metric(
            below_tangent_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> below_tangent_body(
            below_tangent_original, below_tangent_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            BelowTangentPolicy;
        typedef BelowTangentPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                         ScriptedGaussianRNG> BelowTangentWalk;
        double const A = std::ldexp(1.0, -26);
        double const B = std::nextafter(1.0, 0.0);
        Point initial = make_point({A / 2.0});
        Point point = initial;
        ScriptedGaussianRNG rng({0.0, B / 2.0}, {0.0, 1.0});
        BelowTangentPolicy::parameters parameters(
            std::acos(-1.0) / 2.0, true, 100, true);
        BelowTangentWalk walk(
            below_tangent_body, point, 0.5, rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(below_tangent_body, point, 0.5, 1, rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(below_tangent_body.is_in(point) == -1);
    }

    // A shallow but well-resolved crossing needs the exact short-expansion
    // radial gap.  Directly subtracting the rounded squares loses enough
    // relative precision here to move the hit by many contact tolerances.
    {
        Poset shallow_poset = make_poset(1, Poset::RV{});
        OP shallow_original(shallow_poset);
        OrderPolytopeDiagonalRounding<Point> shallow_metric(
            shallow_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> shallow_body(
            shallow_original, shallow_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            ShallowPolicy;
        typedef ShallowPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                    ScriptedGaussianRNG> ShallowWalk;
        double const A = 0x1.33332e4000000p-1;
        double const B = 0x1.99999d5000000p-1;
        double const expected_angle = std::atan2(0.8, 0.6);
        Point point = make_point({A / 2.0});
        ScriptedGaussianRNG rng({0.0, B / 2.0}, {0.0, 1.0});
        ShallowPolicy::parameters parameters(1.2, true, 100, true);
        ShallowWalk walk(shallow_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(walk.apply(shallow_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({1}));
        CHECK(std::abs(trace.hit_angle - expected_angle) < 1e-14);
        CHECK(std::abs(trace.collision_position[0] - 0.5) < 1e-14);
        double const pre_hit_velocity =
            (-A * 0.8 + B * 0.6) / 2.0;
        double const remaining = 1.2 - expected_angle;
        double const expected_final_position =
            0.5 * std::cos(remaining) -
            pre_hit_velocity * std::sin(remaining);
        double const expected_final_velocity =
            -0.5 * std::sin(remaining) -
            pre_hit_velocity * std::cos(remaining);
        CHECK(std::abs(point[0] - expected_final_position) < 1e-13);
        CHECK(std::abs(trace.final_velocity[0] - expected_final_velocity) <
              1e-13);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(shallow_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // Scaling A/B/C by their rounded maximum perturbs this shallow root.
    // Exact power-of-two scaling preserves the binary inputs and recovers the
    // Pythagorean hit direction (8191, 33546240) / 33546241.
    {
        Poset scale_root_poset = make_poset(1, Poset::RV{});
        OP scale_root_original(scale_root_poset);
        OrderPolytopeDiagonalRounding<Point> scale_root_metric(
            scale_root_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> scale_root_body(
            scale_root_original, scale_root_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            ScaleRootPolicy;
        typedef ScaleRootPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                      ScriptedGaussianRNG> ScaleRootWalk;
        double const A = 0x1.8028p-14;
        double const B = 0x1.0000001ffb000p+0;
        double const expected_angle = 0x1.920fb4c44026ep+0;
        Point point = make_point({A / 2.0});
        ScriptedGaussianRNG rng({0.0, B / 2.0}, {0.0, 1.0});
        ScaleRootPolicy::parameters parameters(1.5707, true, 100, true);
        ScaleRootWalk walk(
            scale_root_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(scale_root_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({1}));
        CHECK(std::abs(trace.hit_angle - expected_angle) < 1e-14);
        CHECK(std::abs(trace.collision_position[0] - 0.5) < 2e-14);
        CHECK(trace.collision_position[0] <= 0.5);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(scale_root_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // Event-row conditioning must use an exact binary scale.  Dividing this
    // upper-wall equation by its non-power-of-two slack moves the collision
    // to the exterior side even though every input is finite.
    {
        Poset scale_poset = make_poset(1, Poset::RV{});
        OP scale_original(scale_poset);
        double const upper_slack = 0.99518451811329001;
        double const center_value = 1.0 - upper_slack;
        double const modal_velocity = 0.99518451811441144;
        OrderPolytopeDiagonalRounding<Point> scale_metric(
            scale_original, make_point({center_value}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> scale_body(
            scale_original, scale_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            ScalePolicy;
        typedef ScalePolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> ScaleWalk;
        Point point(1);
        ScriptedGaussianRNG rng(
            {0.0, modal_velocity}, {0.0, 1.0});
        ScalePolicy::parameters parameters(1.58, true, 100, true);
        ScaleWalk walk(scale_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(walk.apply(scale_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({1}));
        CHECK(std::abs(trace.hit_angle - 0x1.921f9c147a982p+0) < 1e-14);
        CHECK(std::abs(trace.collision_position[0] - upper_slack) < 2e-14);
        CHECK(trace.collision_position[0] <= upper_slack);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(scale_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // Explicit trajectories longer than a quarter period rebuild the queue at
    // chunk boundaries.  The upper wall is hit again in the third chunk, so a
    // last-hit suppression that leaked across chunks would miss it.
    {
        Poset long_poset = make_poset(1, Poset::RV{});
        OP long_original(long_poset);
        OrderPolytopeDiagonalRounding<Point> long_metric(
            long_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> long_body(
            long_original, long_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            LongPolicy;
        typedef LongPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                 ScriptedGaussianRNG> LongWalk;
        double const initial_velocity = 0.8;
        double const final_angle = 4.4;
        Point point(1);
        ScriptedGaussianRNG rng(
            {0.0, initial_velocity}, {0.0, 1.0});
        LongPolicy::parameters parameters(final_angle, true, 100, true);
        LongWalk walk(long_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(walk.apply(long_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        double const first_hit = std::asin(0.5 / initial_velocity);
        CHECK(std::abs(trace.hit_angle - first_hit) < 1e-13);
        double const third_hit = 5.0 * first_hit;
        double const wall_speed = std::sqrt(
            initial_velocity * initial_velocity - 0.25);
        double const remaining = final_angle - third_hit;
        double const expected_position =
            0.5 * std::cos(remaining) -
            wall_speed * std::sin(remaining);
        double const expected_velocity =
            -0.5 * std::sin(remaining) -
            wall_speed * std::cos(remaining);
        CHECK(std::abs(point[0] - expected_position) < 1e-12);
        CHECK(std::abs(trace.final_velocity[0] - expected_velocity) < 1e-12);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 3);
#endif
        CHECK(long_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // A queue-empty leg is also verified by the reference scan; otherwise a
    // completely missed event could escape the EventQueue cross-check.
    {
        Poset empty_leg_poset = make_poset(1, Poset::RV{});
        OP empty_leg_original(empty_leg_poset);
        OrderPolytopeDiagonalRounding<Point> empty_leg_metric(
            empty_leg_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> empty_leg_body(
            empty_leg_original, empty_leg_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            EmptyLegPolicy;
        typedef EmptyLegPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                     ScriptedGaussianRNG> EmptyLegWalk;
        Point point(1);
        ScriptedGaussianRNG rng({0.0, 0.1}, {0.0, 1.0});
        EmptyLegPolicy::parameters parameters(0.1, true, 100, true);
        EmptyLegWalk walk(
            empty_leg_body, point, 0.5, rng, parameters);
        walk.apply(empty_leg_body, point, 0.5, 1, rng);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 0);
#endif
        CHECK(empty_leg_body.is_in(point) == -1);
    }

    // A root just beyond the requested endpoint is never reflected early.
    // The structured endpoint residual proves that the requested endpoint is
    // still inside, so the leg advances without a contact.
    {
        Poset endpoint_poset = make_poset(1, Poset::RV{});
        OP endpoint_original(endpoint_poset);
        OrderPolytopeDiagonalRounding<Point> endpoint_metric(
            endpoint_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> endpoint_body(
            endpoint_original, endpoint_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            EndpointPolicy;
        typedef EndpointPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                     ScriptedGaussianRNG> EndpointWalk;
        double const end_angle = 0.1;
        double const root_angle = end_angle + 1e-14;
        double const alpha = 0.25;
        double const velocity =
            (0.5 - alpha * std::cos(root_angle)) / std::sin(root_angle);
        Point initial = make_point({alpha});
        Point point = initial;
        ScriptedGaussianRNG rng({0.0, velocity}, {0.0, 1.0});
        EndpointPolicy::parameters parameters(end_angle, true, 100, true);
        EndpointWalk walk(endpoint_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(endpoint_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        double const expected_endpoint =
            alpha * std::cos(end_angle) +
            velocity * std::sin(end_angle);
        CHECK(point[0] == doctest::Approx(expected_endpoint).epsilon(1e-14));
        CHECK(point[0] < 0.5);
        CHECK(endpoint_body.is_in(point) == -1);
    }

    // These two roots are close in key space but their angular directions
    // differ by more than the tight tie window.  They therefore remain
    // sequential singletons rather than being promoted to a batch.
    {
        Poset ulp_close_poset = make_poset(2, Poset::RV{});
        OP ulp_close_original(ulp_close_poset);
        OrderPolytopeDiagonalRounding<Point> ulp_close_metric(
            ulp_close_original, make_point({0.5, 0.5}),
            make_vector({1.0, 1.0}));
        DiagonalRoundingOrderPolytope<Point> ulp_close_body(
            ulp_close_original, ulp_close_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            UlpClosePolicy;
        typedef UlpClosePolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                     ScriptedGaussianRNG> UlpCloseWalk;
        double const first_angle = std::acos(-1.0) / 6.0;
        double const second_angle = first_angle + 2e-14;
        double const velocity0 = 0.5 / std::sin(first_angle);
        double const velocity1 = 0.5 / std::sin(second_angle);
        Point initial(2), point(2);
        ScriptedGaussianRNG rng(
            {0.0, 0.0, velocity0, velocity1}, {0.0, 0.8});
        UlpClosePolicy::parameters parameters(1.0, true, 100, true);
        UlpCloseWalk walk(
            ulp_close_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(ulp_close_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.captured_post_reflection);
        CHECK(trace.contact_facets.size() == 1);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(ulp_close_body.is_in(point) == -1);
    }

    // Even with large cancelling row terms, a direction separation outside
    // the tight tie window is processed sequentially and cross-checked by the
    // independent full scan.
    {
        Poset cancellation_poset = make_poset(2, Poset::RV{});
        OP cancellation_original(cancellation_poset);
        double const upper_slack0 = 1e-12;
        Point cancellation_center =
            make_point({1.0 - upper_slack0, 0.5});
        OrderPolytopeDiagonalRounding<Point> cancellation_metric(
            cancellation_original, cancellation_center,
            make_vector({1.0, 1.0}));
        DiagonalRoundingOrderPolytope<Point> cancellation_body(
            cancellation_original, cancellation_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            CancellationPolicy;
        typedef CancellationPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                          ScriptedGaussianRNG> CancellationWalk;
        Point initial = make_point({0.1 - cancellation_center[0], 0.0});
        Point point = initial;
        double const later = 0.5;
        double const earlier = later - 2e-14;
        double const alpha0 = point[0] / upper_slack0;
        double const beta0 =
            (1.0 - alpha0 * std::cos(later)) / std::sin(later);
        double const velocity0 = beta0 * upper_slack0;
        double const velocity1 = 0.5 / std::sin(earlier);
        ScriptedGaussianRNG rng(
            {0.0, 0.0, velocity0, velocity1}, {0.0, 1.0});
        CancellationPolicy::parameters parameters(0.51, true, 100, true);
        CancellationWalk walk(
            cancellation_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(cancellation_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.captured_post_reflection);
        CHECK(trace.contact_facets.size() == 1);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(cancellation_body.is_in(point) == -1);
    }

    // Near-tangent roots that are still outside the tight direction window
    // remain sequential even when a rounded residual alone would look tied.
    {
        Poset tangent_pair_poset = make_poset(2, Poset::RV{});
        OP tangent_pair_original(tangent_pair_poset);
        OrderPolytopeDiagonalRounding<Point> tangent_pair_metric(
            tangent_pair_original, make_point({0.5, 0.5}),
            make_vector({1.0, 1.0}));
        DiagonalRoundingOrderPolytope<Point> tangent_pair_body(
            tangent_pair_original, tangent_pair_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            TangentPairPolicy;
        typedef TangentPairPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                        ScriptedGaussianRNG> TangentPairWalk;
        double const radius = 1.0 + 1e-12;
        double const phi0 = 1e-5;
        double const phi1 = phi0 + 2e-14;
        double const A0 = radius * std::cos(phi0);
        double const A1 = radius * std::cos(phi1);
        double const B0 = radius * std::sin(phi0);
        double const B1 = radius * std::sin(phi1);
        Point initial = make_point({A0 / 2.0, A1 / 2.0});
        Point point = initial;
        ScriptedGaussianRNG rng(
            {0.0, 0.0, B0 / 2.0, B1 / 2.0}, {0.0, 1.0});
        TangentPairPolicy::parameters parameters(2e-5, true, 100, true);
        TangentPairWalk walk(
            tangent_pair_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(tangent_pair_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.captured_post_reflection);
        CHECK(trace.contact_facets.size() == 1);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
#endif
        CHECK(tangent_pair_body.is_in(point) == -1);
    }

    // Validate the popped facet at its own stored direction as well.  With
    // huge cancelling row terms, the algebraic root can otherwise share a
    // rounded key with another wall while missing its own wall materially.
    {
        Poset rounded_root_poset = make_poset(2, Poset::RV{});
        OP rounded_root_original(rounded_root_poset);
        double const center0 = std::nextafter(1.0 - 1e-13, 1.0);
        double const slack = 1.0 - center0;
        OrderPolytopeDiagonalRounding<Point> rounded_root_metric(
            rounded_root_original, make_point({center0, 0.5}),
            make_vector({1e16, 1e16}));
        DiagonalRoundingOrderPolytope<Point> rounded_root_body(
            rounded_root_original, rounded_root_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            RoundedRootPolicy;
        typedef RoundedRootPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                        ScriptedGaussianRNG> RoundedRootWalk;
        double const target_A = -1e13;
        double const target_B = 1.0000000000000997e21;
        double const alpha0 = target_A * slack;
        double const velocity0 = target_B * slack;
        double const velocity1 = 0.5 / 1e-8;
        Point initial = make_point({alpha0, 0.0});
        Point point = initial;
        ScriptedGaussianRNG rng(
            {0.0, 0.0, velocity0 / 1e8, velocity1 / 1e8},
            {0.0, 2e-8});
        RoundedRootPolicy::parameters parameters(1.0, true, 100, true);
        RoundedRootWalk walk(
            rounded_root_body, point, 0.5, rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(rounded_root_body, point, 0.5, 1, rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(rounded_root_body.is_in(point) == -1);
    }

    // Two non-proportional dyadic wall equations share the exact outward
    // direction (3/5,4/5).  The cold common-root predicate must recognize the
    // represented equality and preserve a disjoint pre-reflection batch.
    {
        Poset exact_tie_poset = make_poset(2, Poset::RV{});
        OP exact_tie_original(exact_tie_poset);
        double const center0 = 0.25;
        double const center1 = 0.5;
        double const A0 = std::ldexp(42221196506598.0, -48);
        double const A1 = std::ldexp(42221156506598.0, -47);
        double const B0 = (5.0 * center0 - 3.0 * A0) / 4.0;
        double const B1 = (5.0 * center1 - 3.0 * A1) / 4.0;
        OrderPolytopeDiagonalRounding<Point> exact_tie_metric(
            exact_tie_original, make_point({center0, center1}),
            make_vector({1.0, 1.0}));
        DiagonalRoundingOrderPolytope<Point> exact_tie_body(
            exact_tie_original, exact_tie_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            ExactTiePolicy;
        typedef ExactTiePolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                     ScriptedGaussianRNG> ExactTieWalk;
        Point point = make_point({-A0, -A1});
        ScriptedGaussianRNG rng(
            {0.0, 0.0, -B0, -B1}, {0.0, 1.0});
        ExactTiePolicy::parameters parameters(1.0, true, 100, true);
        ExactTieWalk walk(
            exact_tie_body, point, 0.5, rng, parameters);
        CHECK_NOTHROW(
            walk.apply(exact_tie_body, point, 0.5, 1, rng));
        auto const& trace = walk.get_testing_trace();
        REQUIRE(trace.captured_contact);
        CHECK(trace.captured_post_reflection);
        CHECK(trace.contact_facets == std::vector<unsigned int>({0, 1}));
        CHECK(std::abs(trace.hit_angle - std::atan2(4.0, 3.0)) < 1e-13);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 1);
#endif
        CHECK(exact_tie_body.is_in(point) == -1);
        maximum_full_scan_error = std::max(
            maximum_full_scan_error,
            trace.max_full_scan_direction_error);
    }

    // The existing policy for a simultaneous contact sharing a coordinate is
    // deliberately fail-closed.  Diagonal rounding must not choose an order.
    {
        Poset shared_poset = make_poset(2, Poset::RV{{0, 1}});
        OP shared_original(shared_poset);
        Point shared_center = make_point({0.25, 0.5});
        VT shared_shape = make_vector({0.25, 4.0});
        OrderPolytopeDiagonalRounding<Point> shared_metric(
            shared_original, shared_center, shared_shape);
        DiagonalRoundingOrderPolytope<Point> shared_body(
            shared_original, shared_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk Policy;
        typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>,
                             ScriptedGaussianRNG> Walk;
        // Physical v=(-1,-2) reaches x0=0 and x0=x1 at sin(theta)=0.25.
        ScriptedGaussianRNG rng(
            {0.0, 0.0, -2.0, -1.0}, {0.0, 0.8});
        Point point(2);
        Policy::parameters parameters(1.0, true, 100, true);
        Walk walk(shared_body, point, 0.5, rng, parameters);
        bool failed_closed = false;
        try {
            walk.apply(shared_body, point, 0.5, 1, rng);
        } catch (std::runtime_error const& error) {
            failed_closed = std::string(error.what()).find("shared-coordinate") !=
                            std::string::npos;
        }
        CHECK(failed_closed);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 1);
#endif
        CHECK(max_point_difference(point, Point(2)) == 0.0);
        CHECK(shared_body.is_in(point) == -1);
    }

    // Reconstruct diagonal event rows directly from the structured center.
    // Separately normalizing dense rows shifts the cover root in this finite
    // near-tangent case and would silently turn an exact shared contact into
    // several sequential reflections.
    {
        Poset shared_poset = make_poset(2, Poset::RV{{0, 1}});
        OP shared_original(shared_poset);
        double const center0 = 0.49205227461936385;
        double const center1 = 2.0 * center0;
        double const beta0 = -center0 * (1.0 + 1e-10);
        double const beta1 = 2.0 * beta0;
        OrderPolytopeDiagonalRounding<Point> shared_metric(
            shared_original, make_point({center0, center1}),
            make_vector({1.0, 1.0}));
        DiagonalRoundingOrderPolytope<Point> shared_body(
            shared_original, shared_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SharedPolicy;
        typedef SharedPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                   ScriptedGaussianRNG> SharedWalk;
        Point point(2);
        ScriptedGaussianRNG rng(
            {0.0, 0.0, beta0, beta1}, {0.0, 1.0});
        SharedPolicy::parameters parameters(
            std::acos(-1.0) / 2.0, true, 100, true);
        SharedWalk walk(shared_body, point, 0.5, rng, parameters);
        bool failed_closed = false;
        try {
            walk.apply(shared_body, point, 0.5, 1, rng);
        } catch (std::runtime_error const& error) {
            failed_closed = std::string(error.what()).find(
                "shared-coordinate") != std::string::npos;
        }
        CHECK(failed_closed);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        CHECK(trace.full_scan_checked);
        CHECK(trace.full_scan_agreed);
        CHECK(trace.full_scan_contact_count == 1);
#endif
        CHECK(max_point_difference(point, Point(2)) == 0.0);
        CHECK(shared_body.is_in(point) == -1);
    }

    CHECK(maximum_full_scan_error < 1e-12);
    CHECK(maximum_reflection_error < 1e-11);
    std::cout << std::setprecision(17)
              << "DIAGONAL_ROUNDING_MAX_FULL_SCAN_DIRECTION_ERROR="
              << maximum_full_scan_error << '\n'
              << "DIAGONAL_ROUNDING_MAX_COLLISION_STATE_ERROR="
              << maximum_collision_error << '\n'
              << "DIAGONAL_ROUNDING_MAX_REFLECTED_STATE_ERROR="
              << maximum_reflected_state_error << '\n'
              << "DIAGONAL_ROUNDING_MAX_FINAL_STATE_ERROR="
              << maximum_final_state_error << '\n'
              << "DIAGONAL_ROUNDING_MAX_HMC_REFLECTION_ENERGY_ERROR="
              << maximum_reflection_error << '\n';
}

TEST_CASE("diagonal_rounding_feasibility")
{
    Poset poset = make_poset(
        5, Poset::RV{{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}});
    OP original(poset);
    Point center = make_point({0.1, 0.3, 0.45, 0.7, 0.9});
    OrderPolytopeDiagonalRounding<Point> metric(
        original, center, make_vector({0.2, 0.7, 1.5, 3.0, 8.0}));
    DiagonalRoundingOrderPolytope<Point> body(original, metric);
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 2468> RNG;
    typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk Policy;
    typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>, RNG> Walk;
    RNG rng(5);
    Point point(5);
    Policy::parameters parameters(0.75, true, 1000, true);
    Walk walk(body, point, 1.0, rng, parameters);

    double maximum_violation = max_original_violation(point, center, poset);
    for (unsigned int sample = 0; sample < 200; ++sample) {
        walk.apply(body, point, 1.0, 1, rng);
        maximum_violation = std::max(
            maximum_violation,
            max_original_violation(point, center, poset));
        CHECK(body.is_in(point, 1e-10) == -1);
    }
    CHECK(maximum_violation <= 1e-10);

    // A non-finite refreshed velocity is rejected even for a zero-time leg;
    // the last feasible point is restored and no sample is emitted silently.
    {
        Poset antichain = make_poset(2, Poset::RV{});
        OP small_original(antichain);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5, 0.5}), make_vector({1.0, 4.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        double const nan = std::numeric_limits<double>::quiet_NaN();
        ScriptedGaussianRNG scripted_rng(
            {0.0, 0.0, nan, 0.0}, {0.0, 0.0});
        Point small_point(2);
        SmallPolicy::parameters small_parameters(1.0, true, 100, true);
        SmallWalk small_walk(
            small_body, small_point, 0.5, scripted_rng, small_parameters);
        CHECK_THROWS_AS(
            small_walk.apply(small_body, small_point, 0.5, 1, scripted_rng),
            std::runtime_error);
        CHECK(max_point_difference(small_point, Point(2)) == 0.0);
        CHECK(small_body.is_in(small_point) == -1);
    }

    // The scaled diagonal root formula handles a huge but finite modal
    // coefficient without forming R^2.  It must reflect at the wall and
    // return a feasible finite sample instead of overflowing or skipping it.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point small_point(1);
        ScriptedGaussianRNG scripted_rng(
            {0.0, 1e200}, {0.0, 1e-200});
        SmallPolicy::parameters small_parameters(1.0, true, 100, true);
        SmallWalk small_walk(
            small_body, small_point, 0.5, scripted_rng, small_parameters);
        CHECK_NOTHROW(
            small_walk.apply(small_body, small_point, 0.5, 1, scripted_rng));
        auto const& trace = small_walk.get_testing_trace();
        CHECK(trace.captured_contact);
        CHECK(trace.contact_facets == std::vector<unsigned int>({1}));
        CHECK(std::isfinite(small_point[0]));
        CHECK(small_body.is_in(small_point) == -1);
    }

    // If power-of-two root scaling would underflow a nonzero coefficient,
    // the equation is no longer preserved exactly.  Reject and roll back
    // rather than solve the silently altered equation.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point initial = make_point({std::numeric_limits<double>::denorm_min()});
        Point point = initial;
        ScriptedGaussianRNG scripted_rng(
            {0.0, 1e308}, {0.0, 1.0});
        SmallPolicy::parameters parameters(1e-308, true, 100, true);
        SmallWalk walk(
            small_body, point, 0.5, scripted_rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(small_body, point, 0.5, 1, scripted_rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(small_body.is_in(point) == -1);
    }

    // A strictly forward root may be smaller than the least representable
    // positive direction cross and collapse onto the current direction.  It
    // must be rejected by the event oracle before reflection, not skipped
    // until an infeasible endpoint is discovered.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        double const A = 0x1.fe391491bc4a2p-1;
        double const B = 0x1.024de1b9fedfap+1016;
        double const C = 0x1.fe391491bc4a3p-1;
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({1.0 - C}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point initial = make_point({A});
        Point point = initial;
        ScriptedGaussianRNG scripted_rng(
            {0.0, B}, {0.0, 1.0});
        SmallPolicy::parameters parameters(1.0, true, 100, true);
        SmallWalk walk(
            small_body, point, 0.5, scripted_rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(small_body, point, 0.5, 1, scripted_rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(small_body.is_in(point) == -1);
    }

    // An explicitly infeasible start is rejected even when its violation is
    // smaller than the former endpoint tolerance.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point infeasible_point = make_point({0.5000000000001});
        ScriptedGaussianRNG scripted_rng({0.0}, {0.0});
        SmallPolicy::parameters parameters(1.0, true, 100, true);
        CHECK_THROWS_AS(
            (SmallWalk(small_body, infeasible_point, 0.5,
                       scripted_rng, parameters)),
            std::runtime_error);
    }

    // Event-row normalization itself may round an outside product to exactly
    // one.  Check feasibility in original coordinates, where this lower-bound
    // violation remains representable even for a zero-time leg.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        double const center_value = 1.0 / 3.0;
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({center_value}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point infeasible_point = make_point(
            {-std::nextafter(center_value,
                             std::numeric_limits<double>::infinity())});
        CHECK(center_value + infeasible_point[0] < 0.0);
        ScriptedGaussianRNG scripted_rng({0.0}, {0.0});
        SmallPolicy::parameters parameters(0.0, true, 100, true);
        CHECK_THROWS_AS(
            (SmallWalk(small_body, infeasible_point, 0.5,
                       scripted_rng, parameters)),
            std::runtime_error);
        CHECK(center_value + infeasible_point[0] < 0.0);
    }

    // The dual upper-bound case can also round center + coordinate to exactly
    // one.  The exact short-expansion sign must still reject the shifted
    // coordinate that lies one ulp beyond its stored slack.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        double const center_value = 2.0 / 3.0;
        double const upper_slack = 1.0 - center_value;
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({center_value}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point infeasible_point = make_point(
            {std::nextafter(upper_slack,
                            std::numeric_limits<double>::infinity())});
        CHECK(center_value + infeasible_point[0] == 1.0);
        CHECK(small_body.is_in(infeasible_point, 0.0) == 0);
        ScriptedGaussianRNG scripted_rng({0.0}, {0.0});
        SmallPolicy::parameters parameters(0.0, true, 100, true);
        CHECK_THROWS_AS(
            (SmallWalk(small_body, infeasible_point, 0.5,
                       scripted_rng, parameters)),
            std::runtime_error);
    }

    // A boundary start with outward velocity has unresolved time-zero
    // reflection semantics.  Detect it before queue construction, roll back,
    // and never follow an exterior branch to a later re-entry.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point initial = make_point({0.5});
        Point point = initial;
        double const reentry = 2.0 * std::atan(0.2);
        ScriptedGaussianRNG scripted_rng(
            {-0.1, 0.1}, {0.0, 1.0});
        SmallPolicy::parameters parameters(
            2.0 * reentry, true, 100, true);
        SmallWalk walk(
            small_body, point, 0.5, scripted_rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(small_body, point, 0.5, 1, scripted_rng),
            std::runtime_error);
        auto const& trace = walk.get_testing_trace();
        CHECK_FALSE(trace.captured_contact);
        CHECK_FALSE(trace.captured_post_reflection);
        CHECK(max_point_difference(point, initial) == 0.0);
        CHECK(small_body.is_in(point) == -1);
    }

    // Finite time and frequency inputs whose product overflows are rejected
    // before the quarter-period loop can stall.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP small_original(singleton);
        OrderPolytopeDiagonalRounding<Point> small_metric(
            small_original, make_point({0.5}), make_vector({1.0}));
        DiagonalRoundingOrderPolytope<Point> small_body(
            small_original, small_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            SmallPolicy;
        typedef SmallPolicy::Walk<DiagonalRoundingOrderPolytope<Point>,
                                  ScriptedGaussianRNG> SmallWalk;
        Point small_point(1);
        ScriptedGaussianRNG scripted_rng({0.0, 0.0}, {0.0, 1.0});
        SmallPolicy::parameters parameters(
            std::numeric_limits<double>::max(), true, 100, true);
        SmallWalk small_walk(
            small_body, small_point, 1.0, scripted_rng, parameters);
        CHECK_THROWS_AS(
            small_walk.apply(small_body, small_point, 1.0, 1, scripted_rng),
            std::runtime_error);
        CHECK(max_point_difference(small_point, Point(1)) == 0.0);
    }

    // A walk owns pointers into its construction-time rounded body.  Applying
    // it to another same-typed instance must fail instead of mixing geometry
    // and velocity scales.
    {
        Poset singleton = make_poset(1, Poset::RV{});
        OP original(singleton);
        OrderPolytopeDiagonalRounding<Point> first_metric(
            original, make_point({0.5}), make_vector({1.0}));
        OrderPolytopeDiagonalRounding<Point> second_metric(
            original, make_point({0.5}), make_vector({4.0}));
        DiagonalRoundingOrderPolytope<Point> first_body(
            original, first_metric);
        DiagonalRoundingOrderPolytope<Point> second_body(
            original, second_metric);
        typedef OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
            Policy;
        typedef Policy::Walk<DiagonalRoundingOrderPolytope<Point>,
                             ScriptedGaussianRNG> Walk;
        Point point(1);
        ScriptedGaussianRNG rng({0.0, 0.0}, {0.0, 0.0});
        Policy::parameters parameters(1.0, true, 100, true);
        Walk walk(first_body, point, 0.5, rng, parameters);
        CHECK_THROWS_AS(
            walk.apply(second_body, point, 0.5, 1, rng),
            std::runtime_error);
        CHECK(max_point_difference(point, Point(1)) == 0.0);
    }
    std::cout << std::setprecision(17)
              << "DIAGONAL_ROUNDING_MAX_FEASIBILITY_VIOLATION="
              << maximum_violation << '\n';
}

TEST_CASE("diagonal_rounding_counting_exact")
{
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 97531> RNG;
    std::vector<Poset> posets{
        make_poset(4, Poset::RV{{0, 1}, {0, 2}, {1, 3}, {2, 3}}),
        make_poset(5, Poset::RV{{0, 2}, {0, 4}, {1, 3}, {2, 4}}),
        make_poset(5, Poset::RV{{0, 1}, {0, 3}, {1, 4}, {2, 4}})};
    std::vector<double> expected_log_counts{
        std::log(2.0), std::log(10.0), std::log(11.0)};

    for (unsigned int case_index = 0; case_index < posets.size(); ++case_index) {
        Poset const& poset = posets[case_index];
        unsigned int const n = poset.num_elem();
        VT center_coeffs(n), identity = VT::Ones(n), shape(n);
        for (unsigned int i = 0; i < n; ++i) {
            center_coeffs(i) = 0.1 + 0.8 * double(i + 1) / double(n + 1);
            shape(i) = 0.5 + double((i + 1) * (i + 1));
        }
        Point center(center_coeffs);
        OrderPolytopeDiagonalRounding<Point> identity_metric(
            poset, center, identity);
        OrderPolytopeDiagonalRounding<Point> nonuniform_metric(
            poset, center, shape);
        if (case_index == 0) {
            RNG invalid_rng(n);
            CHECK_THROWS_AS(
                (estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
                    poset, invalid_rng, identity_metric, 0.0, 1,
                    exact_dp_options())),
                std::invalid_argument);
            CHECK_THROWS_AS(
                (estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
                    poset, invalid_rng, identity_metric, 0.3, 0,
                    exact_dp_options())),
                std::invalid_argument);
            auto const float_error_result =
                estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
                    poset, invalid_rng, identity_metric, 0.3f, 1,
                    exact_dp_options());
            CHECK(float_error_result.log_extensions ==
                  doctest::Approx(std::log(2.0)).epsilon(1e-14));
        }
        RNG identity_rng(n), nonuniform_rng(n);
        OrderPolytopeDiagonalRoundingDiagnostics<double> identity_diagnostics;
        OrderPolytopeDiagonalRoundingDiagnostics<double> nonuniform_diagnostics;
        auto const options = exact_dp_options();

        auto const identity_result =
            estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
                poset, identity_rng, identity_metric, 0.3, 1, options,
                &identity_diagnostics);
        auto const nonuniform_result =
            estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
                poset, nonuniform_rng, nonuniform_metric, 0.3, 1, options,
                &nonuniform_diagnostics);
        CHECK(identity_result.reduction_exact);
        CHECK(nonuniform_result.reduction_exact);
        CHECK(identity_result.residual_count == 0);
        CHECK(nonuniform_result.residual_count == 0);
        CHECK(identity_diagnostics.total_trajectories() == 0);
        CHECK(nonuniform_diagnostics.total_trajectories() == 0);
        CHECK(identity_result.log_extensions ==
              doctest::Approx(nonuniform_result.log_extensions).epsilon(1e-14));

        CHECK(identity_result.log_extensions ==
              doctest::Approx(expected_log_counts[case_index]).epsilon(1e-14));
    }

    Poset empty = make_poset(0, Poset::RV{});
    Point empty_center(0);
    VT empty_shape(0);
    OrderPolytopeDiagonalRounding<Point> empty_metric(
        empty, empty_center, empty_shape);
    ScriptedGaussianRNG empty_rng({}, {});
    auto const empty_result =
        estimate_log_linear_extensions_diagonal_order_hmc<
            ScriptedGaussianRNG, Point>(
                empty, empty_rng, empty_metric, 0.3, 1,
                exact_dp_options());
    CHECK(empty_result.reduction_exact);
    CHECK(empty_result.residual_count == 0);
    CHECK(empty_result.log_extensions == doctest::Approx(0.0));
    CHECK(empty_result.volume_estimate == doctest::Approx(1.0));

    OP empty_body(empty);
    ScriptedGaussianRNG direct_empty_rng({}, {});
    OrderPolytopeDiagonalCoolingDiagnostics<double> empty_diagnostics;
    auto const direct_empty_result =
        volume_cooling_gaussians_order_polytope_diagonal_result(
            empty_body, direct_empty_rng, empty_metric, 0.3, 1,
            &empty_diagnostics);
    CHECK(direct_empty_result.log_volume == doctest::Approx(0.0));
    CHECK(direct_empty_result.volume_estimate == doctest::Approx(1.0));
    CHECK(empty_diagnostics.cooling_phase_count == 0);
    CHECK(empty_diagnostics.total_trajectories() == 0);
}

TEST_CASE("diagonal_rounding_residual_vertex_maps")
{
    Poset poset = make_poset(
        8, Poset::RV{{0, 2}, {0, 6}, {4, 6},
                     {1, 3}, {1, 7}, {5, 7}});
    OrderPolytopeVolumeReductionOptions options;
    options.enable_chain_antichain = false;
    options.enable_parallel_decomposition = true;
    options.enable_ordinal_decomposition = false;
    options.enable_small_exact_dp = false;
    options.exact_dp_max_n = 0;
    auto const reduced =
        reduce_order_polytope_volume_problem_with_vertex_maps(poset, options);
    REQUIRE(reduced.problem.residual_posets.size() == 2);
    REQUIRE(reduced.residual_vertices.size() == 2);
    CHECK(reduced.residual_vertices[0] ==
          std::vector<unsigned int>({0, 2, 4, 6}));
    CHECK(reduced.residual_vertices[1] ==
          std::vector<unsigned int>({1, 3, 5, 7}));

    Point center = make_point({0.2, 0.5, 0.3, 0.7,
                               0.1, 0.4, 0.25, 0.8});
    VT shape = make_vector({1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0});
    OrderPolytopeDiagonalRounding<Point> metric(poset, center, shape);
    for (unsigned int residual_index = 0; residual_index < 2; ++residual_index) {
        auto restricted = metric.restrict_to(
            reduced.problem.residual_posets[residual_index],
            reduced.residual_vertices[residual_index]);
        REQUIRE(restricted.dimension() == 4);
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int const original =
                reduced.residual_vertices[residual_index][i];
            CHECK(restricted.center()[i] == center[original]);
            CHECK(restricted.shape_diag()(i) == shape(original));
        }
    }
}

TEST_CASE("diagonal_rounding_counting_residual")
{
    typedef BoostRandomNumberGenerator<boost::mt19937, double> RNG;
    Poset poset = make_poset(4, Poset::RV{{0, 1}, {0, 3}, {2, 3}});
    Point center = make_point({0.2, 0.5, 0.3, 0.7});
    VT identity = VT::Ones(4);
    VT nonuniform = make_vector({4.0, 9.0, 16.0, 25.0});
    OrderPolytopeDiagonalRounding<Point> identity_metric(
        poset, center, identity);
    OrderPolytopeDiagonalRounding<Point> nonuniform_metric(
        poset, center, nonuniform);
    RNG identity_rng(4), nonuniform_rng(4);
    identity_rng.set_seed(424242);
    nonuniform_rng.set_seed(424242);
    OrderPolytopeDiagonalRoundingDiagnostics<double> identity_diagnostics;
    OrderPolytopeDiagonalRoundingDiagnostics<double> nonuniform_diagnostics;
    auto const options = forced_residual_options();

    auto const identity_result =
        estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
            poset, identity_rng, identity_metric, 0.3, 1, options,
            &identity_diagnostics);
    auto const nonuniform_result =
        estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
            poset, nonuniform_rng, nonuniform_metric, 0.3, 1, options,
            &nonuniform_diagnostics);
    // Exercise the public diagnostics-null branch, which instantiates the
    // compile-time uninstrumented diagonal walk used by normal callers.
    RNG uninstrumented_rng(4);
    uninstrumented_rng.set_seed(424242);
    auto const uninstrumented_result =
        estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
            poset, uninstrumented_rng, nonuniform_metric, 0.3, 1, options);
    double const truth = std::log(5.0);
    double const identity_error =
        std::abs(identity_result.log_extensions - truth);
    double const nonuniform_error =
        std::abs(nonuniform_result.log_extensions - truth);

    CHECK(!identity_result.reduction_exact);
    CHECK(!nonuniform_result.reduction_exact);
    CHECK(identity_result.residual_count == 1);
    CHECK(nonuniform_result.residual_count == 1);
    CHECK(uninstrumented_result.residual_count == 1);
    CHECK(identity_diagnostics.schedule_trajectories() > 0);
    CHECK(identity_diagnostics.ratio_trajectories() > 0);
    CHECK(nonuniform_diagnostics.schedule_trajectories() > 0);
    CHECK(nonuniform_diagnostics.ratio_trajectories() > 0);
    CHECK(identity_diagnostics.total_trajectories() ==
          identity_diagnostics.schedule_trajectories() +
          identity_diagnostics.ratio_trajectories());
    CHECK(nonuniform_diagnostics.total_trajectories() ==
          nonuniform_diagnostics.schedule_trajectories() +
          nonuniform_diagnostics.ratio_trajectories());
    CHECK(identity_diagnostics.reflection_count() > 0);
    CHECK(nonuniform_diagnostics.reflection_count() > 0);
    CHECK(identity_diagnostics.event_recomputation_count() > 0);
    CHECK(nonuniform_diagnostics.event_recomputation_count() > 0);
    REQUIRE(nonuniform_diagnostics.residual_cooling.size() == 1);
    gaussian_annealing_parameters<double> annealing_parameters(4);
    std::vector<double> expected_nonuniform_a0;
    order_polytope_diagonal_cooling_detail::
        get_first_gaussian_from_metric_distances(
            nonuniform_metric.facet_metric_distances(),
            annealing_parameters.frac, 0.3, expected_nonuniform_a0);
    REQUIRE(expected_nonuniform_a0.size() == 1);
    CHECK(identity_diagnostics.residual_cooling.front().initial_a0 > 0.0);
    CHECK(nonuniform_diagnostics.residual_cooling.front().initial_a0 > 0.0);
    CHECK(nonuniform_diagnostics.residual_cooling.front().initial_a0 ==
          doctest::Approx(expected_nonuniform_a0.front()).epsilon(1e-14));
    CHECK(identity_diagnostics.residual_cooling.front().cooling_phase_count > 0);
    CHECK(nonuniform_diagnostics.residual_cooling.front().cooling_phase_count > 0);
    CHECK(identity_diagnostics.metric_setup_microseconds >= 0.0);
    CHECK(nonuniform_diagnostics.metric_setup_microseconds >= 0.0);
    CHECK(identity_diagnostics.cooling_microseconds() > 0.0);
    CHECK(nonuniform_diagnostics.cooling_microseconds() > 0.0);
    CHECK(identity_diagnostics.microseconds_per_trajectory() > 0.0);
    CHECK(nonuniform_diagnostics.microseconds_per_trajectory() > 0.0);
    CHECK(identity_diagnostics.total_estimator_microseconds >=
          identity_diagnostics.cooling_microseconds());
    CHECK(nonuniform_diagnostics.total_estimator_microseconds >=
          nonuniform_diagnostics.cooling_microseconds());
    CHECK(identity_error < 0.5);
    CHECK(nonuniform_error < 0.5);
    CHECK(std::abs(uninstrumented_result.log_extensions - truth) < 0.5);
    CHECK(std::abs(identity_result.log_extensions -
                   nonuniform_result.log_extensions) < 0.5);
    // sqrt(det D)=120.  Missing or duplicated determinant would shift the
    // nonuniform estimate by log(120), well outside these matched tolerances.
    CHECK(identity_error < 0.5 * std::log(120.0));
    CHECK(nonuniform_error < 0.5 * std::log(120.0));

    std::cout << std::setprecision(17)
              << "DIAGONAL_ROUNDING_RESIDUAL_IDENTITY_LOG_ERROR="
              << identity_error << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_NONUNIFORM_LOG_ERROR="
              << nonuniform_error << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_PAIRED_LOG_DIFFERENCE="
              << std::abs(identity_result.log_extensions -
                          nonuniform_result.log_extensions) << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_IDENTITY_LOG_ESTIMATE="
              << identity_result.log_extensions << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_NONUNIFORM_LOG_ESTIMATE="
              << nonuniform_result.log_extensions << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_IDENTITY_A0="
              << identity_diagnostics.residual_cooling.front().initial_a0 << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_NONUNIFORM_A0="
              << nonuniform_diagnostics.residual_cooling.front().initial_a0 << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_IDENTITY_PHASES="
              << identity_diagnostics.residual_cooling.front().cooling_phase_count << '\n'
              << "DIAGONAL_ROUNDING_RESIDUAL_NONUNIFORM_PHASES="
              << nonuniform_diagnostics.residual_cooling.front().cooling_phase_count << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_TOTAL_TRAJECTORIES="
              << identity_diagnostics.total_trajectories() << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_TOTAL_TRAJECTORIES="
              << nonuniform_diagnostics.total_trajectories() << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_REFLECTIONS="
              << identity_diagnostics.reflection_count() << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_REFLECTIONS="
              << nonuniform_diagnostics.reflection_count() << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_EVENT_RECOMPUTATIONS="
              << identity_diagnostics.event_recomputation_count() << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_EVENT_RECOMPUTATIONS="
              << nonuniform_diagnostics.event_recomputation_count() << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_METRIC_SETUP_US="
              << identity_diagnostics.metric_setup_microseconds << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_METRIC_SETUP_US="
              << nonuniform_diagnostics.metric_setup_microseconds << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_COOLING_US="
              << identity_diagnostics.cooling_microseconds() << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_COOLING_US="
              << nonuniform_diagnostics.cooling_microseconds() << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_US_PER_TRAJECTORY="
              << identity_diagnostics.microseconds_per_trajectory() << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_US_PER_TRAJECTORY="
              << nonuniform_diagnostics.microseconds_per_trajectory() << '\n'
              << "DIAGONAL_ROUNDING_IDENTITY_ESTIMATOR_US="
              << identity_diagnostics.total_estimator_microseconds << '\n'
              << "DIAGONAL_ROUNDING_NONUNIFORM_ESTIMATOR_US="
              << nonuniform_diagnostics.total_estimator_microseconds << '\n';
}
