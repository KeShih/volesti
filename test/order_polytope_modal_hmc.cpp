// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "generators/boost_random_number_generator.hpp"
#include "misc/poset.h"
#include "random_walks/order_polytope_general_gaussian_hmc_walk.hpp"
#include "volume/order_polytope_general_mass_cooling.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef OrderPolytope<Point> OrderBody;
typedef typename OrderBody::MT MT;
typedef typename OrderBody::VT VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 97> RNG;
typedef OrderPolytopeGeneralGaussianHMCWalk GeneralWalk;

namespace GeneralDetail = order_gaussian_hmc_detail;

class ScriptedRNG
{
public:
    ScriptedRNG(std::vector<NT> normals, std::vector<NT> uniforms)
        : _normals(normals), _uniforms(uniforms)
    {}

    NT sample_ndist() { return _normals.at(_normal_index++); }
    NT sample_urdist() { return _uniforms.at(_uniform_index++); }

private:
    std::vector<NT> _normals, _uniforms;
    std::size_t _normal_index = 0, _uniform_index = 0;
};

static MT test_spd(unsigned int n, NT diagonal, NT coupling)
{
    MT matrix(n, n);
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            matrix(i, j) = i == j
                ? diagonal + NT(0.3) * NT(i)
                : coupling / NT(1 + std::abs(int(i) - int(j)));
    return NT(0.5) * (matrix + MT(matrix.transpose()));
}

TEST_CASE("general_mass_oracle_certified_bounds")
{
    NT const tolerance = NT(1e-3);
    CHECK(GeneralDetail::certified_nonpositive(-tolerance, tolerance));
    CHECK_FALSE(GeneralDetail::certified_nonpositive(
        -NT(0.5) * tolerance, tolerance));
    CHECK(GeneralDetail::certified_positive(
        NT(2) * tolerance, tolerance));
    CHECK_FALSE(GeneralDetail::certified_positive(
        NT(0.5) * tolerance, tolerance));

    NT const midpoint_value = NT(-0.56);
    NT const midpoint_derivative = NT(0.1);
    NT const derivative_tolerance = NT(0.5);
    NT const old_quadratic_upper =
        midpoint_value + NT(0.01) + std::abs(midpoint_derivative);
    NT const quadratic_upper = GeneralDetail::quadratic_value_upper(
        midpoint_value, NT(0.01), midpoint_derivative,
        derivative_tolerance, NT(0), NT(1));
    CHECK(old_quadratic_upper < NT(0));
    CHECK(quadratic_upper > NT(0));

    Poset::RV relations;
    Poset poset(1, relations);
    OrderBody body(poset);
    GeneralDetail::ModalTrajectory<OrderBody> trajectory(
        body, MT::Identity(1, 1), MT::Identity(1, 1),
        VT::Constant(1, NT(0.5)));
    trajectory.set_state(
        VT::Constant(1, NT(1e-7)), VT::Constant(1, NT(-0.1)));

    NT lower = NT(0), upper = NT(2e-6);
    for (unsigned int step = 0; step < 80; ++step) {
        NT const midpoint = NT(0.5) * (lower + upper);
        if (trajectory.facet_value(0, midpoint) > NT(-5e-13))
            upper = midpoint;
        else
            lower = midpoint;
    }
    NT const horizon = NT(0.5) * (lower + upper);
    CHECK(trajectory.facet_value(0, horizon) < NT(0));
    CHECK(std::abs(trajectory.facet_value(0, horizon)) < NT(1e-12));

    GeneralDetail::OracleOptions<NT> options;
    options.max_interval_steps = 1;
    CHECK_THROWS_WITH_AS(
        GeneralDetail::first_hit(trajectory, horizon, -1, options),
        doctest::Contains("interval step limit"), std::runtime_error);
}

TEST_CASE("general_mass_modal_trajectory")
{
    Poset::RV relations{{0, 1}, {1, 2}};
    Poset poset(3, relations);
    OrderBody body(poset);
    MT const precision = test_spd(3, NT(2.5), NT(0.6));
    MT const mass = test_spd(3, NT(1.2), NT(-0.3));
    VT const center = VT::Constant(3, NT(0.45));
    GeneralDetail::ModalTrajectory<OrderBody> trajectory(
        body, precision, mass, center);

    VT position(3), velocity(3);
    position << NT(0.2), NT(0.5), NT(0.8);
    velocity << NT(-0.7), NT(0.4), NT(0.9);
    trajectory.set_state(position, velocity);

    auto energy = [&](VT const& x, VT const& v) {
        VT const offset = x - center;
        return NT(0.5) *
            (offset.dot(precision * offset) + v.dot(mass * v));
    };
    NT const initial_energy = energy(position, velocity);
    NT const h = NT(1e-6);
    for (NT time : {NT(0.15), NT(0.7), NT(1.9)}) {
        VT const x = trajectory.position(time);
        VT const v = trajectory.velocity(time);
        CHECK(energy(x, v) == doctest::Approx(initial_energy).epsilon(1e-10));
        VT const derivative =
            (trajectory.position(time + h) - trajectory.position(time - h)) /
            (NT(2) * h);
        CHECK((derivative - v).norm() < NT(2e-6));
        for (unsigned int facet = 0; facet < trajectory.num_facets(); ++facet) {
            NT const finite_difference =
                (trajectory.facet_value(facet, time + h) -
                 trajectory.facet_value(facet, time - h)) /
                (NT(2) * h);
            CHECK(std::abs(finite_difference -
                           trajectory.facet_derivative(facet, time)) <
                  NT(2e-6));
        }
    }

    VT incident_velocity(3);
    incident_velocity << NT(0.9), NT(-0.4), NT(0.3);
    NT const kinetic_energy = incident_velocity.dot(mass * incident_velocity);
    for (unsigned int facet = 0; facet < trajectory.num_facets(); ++facet) {
        VT const inverse_mass_normal =
            trajectory.inverse_mass_normal(facet);
        NT const denominator =
            trajectory.normal_dot(facet, inverse_mass_normal);
        auto reflect = [&](VT const& value) {
            return value -
                (NT(2) * trajectory.normal_dot(facet, value) / denominator) *
                    inverse_mass_normal;
        };
        VT const reflected = reflect(incident_velocity);
        CHECK(trajectory.normal_dot(facet, reflected) ==
              doctest::Approx(-trajectory.normal_dot(
                  facet, incident_velocity)).epsilon(1e-12));
        CHECK(reflected.dot(mass * reflected) ==
              doctest::Approx(kinetic_energy).epsilon(1e-12));
        CHECK((reflect(reflected) - incident_velocity).norm() < NT(1e-12));
    }

    Poset::RV shifted_relations{{0, 1}};
    Poset shifted_poset(2, shifted_relations);
    OrderBody shifted_body(shifted_poset);
    shifted_body.normalize();
    VT shift(2);
    shift << NT(0.25), NT(-0.1);
    shifted_body.shift(shift);
    GeneralDetail::ModalTrajectory<OrderBody> shifted_trajectory(
        shifted_body, MT::Identity(2, 2), MT::Identity(2, 2), VT::Zero(2));
    VT original(2);
    original << NT(0.3), NT(0.7);
    shifted_trajectory.set_state(original - shift, VT::Zero(2));
    REQUIRE(shifted_trajectory.num_facets() == 3);
    CHECK(shifted_trajectory.facet_value(0, NT(0)) ==
          doctest::Approx(-original(0)).epsilon(1e-14));
    CHECK(shifted_trajectory.facet_value(1, NT(0)) ==
          doctest::Approx(original(1) - NT(1)).epsilon(1e-14));
    CHECK(shifted_trajectory.facet_value(2, NT(0)) ==
          doctest::Approx((original(0) - original(1)) / std::sqrt(NT(2)))
              .epsilon(1e-14));
}

TEST_CASE("general_mass_oracle_subinterval_exit")
{
    Poset::RV relations;
    Poset poset(2, relations);
    OrderBody body(poset);

    MT precision(2, 2);
    precision << NT(0.57103969698233803), NT(-0.42896030301766175),
                 NT(-0.42896030301766175), NT(0.57103969698233803);
    VT center(2), position(2), velocity(2);
    center << NT(1.0062225359328096), NT(0.522644208);
    position << NT(0.99999160993280967), NT(0.5);
    velocity << NT(0.00040684607255183302), NT(0.069256254072551843);

    GeneralDetail::ModalTrajectory<OrderBody> trajectory(
        body, precision, MT::Identity(2, 2), center);
    trajectory.set_state(position, velocity);
    NT const horizon = NT(0.39269908169872414);
    unsigned int const upper_first_coordinate = 2;
    CHECK(trajectory.facet_value(upper_first_coordinate, NT(0)) < NT(0));
    CHECK(trajectory.facet_value(
              upper_first_coordinate, NT(0.08250372087041176)) > NT(0));
    CHECK(trajectory.facet_value(upper_first_coordinate, horizon) < NT(0));

    NT lower = NT(0.025), upper = NT(0.026);
    for (unsigned int step = 0; step < 80; ++step) {
        NT const middle = NT(0.5) * (lower + upper);
        if (trajectory.facet_value(upper_first_coordinate, middle) > NT(0))
            upper = middle;
        else
            lower = middle;
    }
    NT const expected = NT(0.5) * (lower + upper);
    auto const hit = GeneralDetail::first_hit(trajectory, horizon, -1);
    CHECK(hit.facet == static_cast<int>(upper_first_coordinate));
    CHECK(hit.time == doctest::Approx(expected).epsilon(1e-11));
    CHECK(trajectory.facet_value(
              upper_first_coordinate, hit.time) <= NT(0));
    CHECK(trajectory.facet_derivative(
              upper_first_coordinate, hit.time) > NT(0));

    GeneralDetail::OracleOptions<NT> bounded;
    bounded.max_interval_steps = 1;
    CHECK_THROWS_WITH_AS(
        GeneralDetail::first_hit(trajectory, horizon, -1, bounded),
        doctest::Contains("interval step limit"), std::runtime_error);

    Poset::RV no_relations;
    Poset fast_poset(1, no_relations);
    OrderBody fast_body(fast_poset);
    NT const frequency = NT(1e12);
    GeneralDetail::ModalTrajectory<OrderBody> fast_trajectory(
        fast_body,
        frequency * frequency * MT::Identity(1, 1),
        MT::Identity(1, 1), VT::Zero(1));
    fast_trajectory.set_state(VT::Zero(1), VT::Constant(1, NT(10)));
    auto const recollision =
        GeneralDetail::first_hit(fast_trajectory, NT(4e-12), 0);
    REQUIRE(recollision.facet == 0);
    CHECK(recollision.time ==
          doctest::Approx(NT(3.14159265358979323846) / frequency)
              .epsilon(1e-10));
    CHECK(recollision.time < GeneralDetail::OracleOptions<NT>().time_tolerance);

    Poset tied_poset(3, no_relations);
    OrderBody tied_body(tied_poset);
    GeneralDetail::ModalTrajectory<OrderBody> tied_trajectory(
        tied_body, MT::Identity(3, 3), MT::Identity(3, 3), VT::Zero(3));
    tied_trajectory.set_state(
        VT::Constant(3, NT(0.5)), VT::Constant(3, NT(-0.375)));
    auto const tied = GeneralDetail::first_hit(tied_trajectory, NT(3), -1);
    REQUIRE(tied.tied_facets.size() == 3);
    for (unsigned int facet = 0; facet < 3; ++facet)
        CHECK(tied.tied_facets[facet] == static_cast<int>(facet));
}

TEST_CASE("general_mass_walk_feasibility")
{
    Poset::RV relations{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    Poset poset(4, relations);
    OrderBody body(poset);
    MT const precision = test_spd(4, NT(3), NT(0.7));
    MT const mass = test_spd(4, NT(1.4), NT(-0.35));
    VT const center = VT::Constant(4, NT(0.5));

    Point point(body.inner_point());
    VT const initial = point.getCoefficients();
    RNG rng(4);
    GeneralWalk::parameters<NT> parameters(precision, mass, center);
    GeneralWalk::Walk<OrderBody, RNG> walk(body, point, parameters, rng);
    walk.update_delta(NT(1));

    NT maximum_displacement = NT(0);
    for (unsigned int step = 0; step < 100; ++step) {
        walk.apply(body, point, 1, rng);
        CHECK(point.getCoefficients().allFinite());
        CHECK(body.is_in(point, NT(1e-8)) == -1);
        maximum_displacement = std::max(
            maximum_displacement,
            (point.getCoefficients() - initial).norm());
    }
    CHECK(maximum_displacement > NT(1e-6));
}

TEST_CASE("general_mass_interior_start_is_not_a_contact")
{
    Poset::RV relations;
    Poset poset(1, relations);
    OrderBody body(poset);
    MT precision(1, 1);
    precision(0, 0) = NT(6.25e-24);
    MT const mass = MT::Identity(1, 1);
    VT const center = VT::Constant(1, NT(0.7));
    Point point(center);
    ScriptedRNG rng({NT(1)}, {NT(1)});
    GeneralWalk::parameters<NT> parameters(precision, mass, center);
    GeneralWalk::Walk<OrderBody, ScriptedRNG> walk(
        body, point, parameters, rng);
    walk.update_delta(NT(0.1));

    walk.apply(body, point, 1, rng);
    CHECK(point[0] == doctest::Approx(NT(0.8)).epsilon(1e-14));
}

TEST_CASE("general_mass_walk_propagates_trajectory_failure")
{
    Poset::RV relations;
    Poset poset(2, relations);
    OrderBody body(poset);
    MT matrix(2, 2);
    matrix << NT(1), NT(1), NT(1), NT(2);
    VT const center = VT::Constant(2, NT(0.5));
    Point point(center);
    Point const initial(point);
    ScriptedRNG rng({NT(-2), NT(-1)}, {NT(0.75)});
    GeneralWalk::parameters<NT> parameters(matrix, matrix, center);
    GeneralWalk::Walk<OrderBody, ScriptedRNG> walk(
        body, point, parameters, rng);

    CHECK_THROWS_WITH_AS(
        walk.apply(body, point, 1, rng),
        doctest::Contains("failed to complete"), std::runtime_error);
    CHECK((point.getCoefficients() - initial.getCoefficients()).norm() ==
          doctest::Approx(NT(0)));
}

TEST_CASE("general_mass_standard_facets_contract")
{
    Poset::RV relations{{0, 1}};
    Poset const poset(2, relations);
    OrderBody body(poset);
    MT const precision = MT::Identity(2, 2);
    MT const mass = MT::Identity(2, 2);
    VT const center = VT::Constant(2, NT(0.5));
    Point point(body.inner_point());
    RNG rng(2);
    GeneralWalk::parameters<NT> const parameters(precision, mass, center);
    typedef GeneralWalk::Walk<OrderBody, RNG> Walk;

    MT transform = MT::Identity(2, 2);
    transform(0, 0) = NT(2);
    OrderBody transformed(body);
    transformed.linear_transformIt(transform);
    CHECK_THROWS_AS(
        GeneralDetail::ModalTrajectory<OrderBody>(
            transformed, precision, mass, center),
        std::invalid_argument);
    CHECK_THROWS_AS(Walk(transformed, point, parameters, rng),
                    std::invalid_argument);

    OrderBody changed_after_construction(body);
    Walk walk(changed_after_construction, point, parameters, rng);
    changed_after_construction.linear_transformIt(transform);
    CHECK_THROWS_AS(walk.apply(changed_after_construction, point, 1, rng),
                    std::invalid_argument);

    OrderBody changed_by_shift(body);
    Point shifted_point(2, {NT(0.25), NT(0.5)});
    RNG shifted_rng(2);
    Walk shifted_walk(changed_by_shift, shifted_point, parameters, shifted_rng);
    VT shift(2);
    shift << NT(0.1), NT(0.2);
    changed_by_shift.shift(shift);
    CHECK(changed_by_shift.is_in(shifted_point, NT(0)) == -1);
    CHECK_THROWS_WITH_AS(
        shifted_walk.apply(changed_by_shift, shifted_point, 0, shifted_rng),
        doctest::Contains("body or point mismatch"), std::invalid_argument);
}

TEST_CASE("general_mass_volume_integration")
{
    Poset::RV relations;
    Poset poset(1, relations);
    OrderBody body(poset);
    MT mass(1, 1);
    mass(0, 0) = NT(2);
    RNG rng(1);

    auto const estimate =
        volume_cooling_gaussians_log_mass(
            body, rng, mass, 0.8, 1, 500000ULL);
    REQUIRE(estimate.success);
    CHECK(std::isfinite(estimate.log_volume));
    CHECK(std::abs(estimate.log_volume) < NT(1));
}
