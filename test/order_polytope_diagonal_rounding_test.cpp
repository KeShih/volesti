// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "convex_bodies/order_polytope_spd_rounding.hpp"
#include "generators/boost_random_number_generator.hpp"
#include "preprocess/order_polytope_diagonal_mvie.hpp"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "random_walks/order_polytope_spd_gaussian_hmc_exact_walk.hpp"
#include "volume/linear_extensions_volume.hpp"

namespace {

using Kernel = Cartesian<double>;
using Point = Kernel::Point;
using OP = OrderPolytope<Point>;
using VT = OP::VT;
using MT = OP::MT;
using DiagonalBody = order_polytope_rounding_detail::DiagonalBody<Point>;
using SPDBody = order_polytope_rounding_detail::SPDBody<Point>;

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

class ScriptedGaussianRNG {
public:
    ScriptedGaussianRNG(std::vector<double> normals,
                        std::vector<double> uniforms)
        : _normals(std::move(normals)), _uniforms(std::move(uniforms))
    {}

    double sample_ndist()
    {
        if (_normal_index == _normals.size())
            throw std::runtime_error("scripted normal sequence exhausted");
        return _normals[_normal_index++];
    }

    double sample_urdist()
    {
        if (_uniform_index == _uniforms.size())
            throw std::runtime_error("scripted uniform sequence exhausted");
        return _uniforms[_uniform_index++];
    }

private:
    std::vector<double> _normals;
    std::vector<double> _uniforms;
    std::size_t _normal_index = 0;
    std::size_t _uniform_index = 0;
};

} // namespace

TEST_CASE("diagonal_mvie_analytic")
{
    SUBCASE("antichain") {
        Poset const poset = make_poset(4, {});
        OP const body = OP::from_reduced_relations(poset);
        auto const ellipsoid = max_inscribed_diagonal_ellipsoid(body);
        for (unsigned int i = 0; i < 4; ++i) {
            CHECK(ellipsoid.center()[i] ==
                  doctest::Approx(0.5).epsilon(2e-6));
            CHECK(ellipsoid.shape_diag()(i) ==
                  doctest::Approx(0.25).epsilon(2e-6));
        }
    }

    SUBCASE("singleton") {
        Poset const poset = make_poset(1, {});
        OP const body = OP::from_reduced_relations(poset);
        auto const ellipsoid = max_inscribed_diagonal_ellipsoid(body);
        CHECK(ellipsoid.center()[0] ==
              doctest::Approx(0.5).epsilon(2e-6));
        CHECK(ellipsoid.shape_diag()(0) ==
              doctest::Approx(0.25).epsilon(2e-6));
    }

    SUBCASE("two element chain") {
        Poset const poset = make_poset(2, {{0, 1}});
        OP const body = OP::from_reduced_relations(poset);
        auto const ellipsoid = max_inscribed_diagonal_ellipsoid(body);
        double const axis = 1.0 / (2.0 + std::sqrt(2.0));
        CHECK(ellipsoid.center()[0] ==
              doctest::Approx(axis).epsilon(2e-5));
        CHECK(ellipsoid.center()[1] ==
              doctest::Approx(1.0 - axis).epsilon(2e-5));
        CHECK(ellipsoid.shape_diag()(0) ==
              doctest::Approx(axis * axis).epsilon(2e-5));
        CHECK(ellipsoid.shape_diag()(1) ==
              doctest::Approx(axis * axis).epsilon(2e-5));

        auto const rounding = diagonal_mvie_rounding(body);
        CHECK(rounding.shape_diag().prod() ==
              doctest::Approx(1.0).epsilon(1e-12));
    }

    SUBCASE("transitive edges and shifted input") {
        Poset const reduced = make_poset(3, {{0, 1}, {1, 2}});
        Poset const redundant = make_poset(
            3, {{0, 1}, {1, 2}, {0, 2}});
        OP const reduced_body = OP::from_reduced_relations(reduced);
        OP redundant_body(redundant);
        auto const expected = max_inscribed_diagonal_ellipsoid(reduced_body);
        auto const actual = max_inscribed_diagonal_ellipsoid(redundant_body);
        for (unsigned int i = 0; i < 3; ++i) {
            CHECK(actual.center()[i] ==
                  doctest::Approx(expected.center()[i]).epsilon(1e-12));
            CHECK(actual.shape_diag()(i) ==
                  doctest::Approx(expected.shape_diag()(i)).epsilon(1e-12));
        }

        OP normalized(reduced);
        normalized.normalize();
        CHECK(normalized.has_standard_order_facets());
        auto const normalized_result =
            max_inscribed_diagonal_ellipsoid(normalized);
        CHECK(normalized_result.shape_diag()(1) ==
              doctest::Approx(expected.shape_diag()(1)).epsilon(1e-12));

        VT shift = make_vector({0.1, 0.2, 0.3});
        redundant_body.shift(shift);
        CHECK(redundant_body.has_standard_order_facets());
        CHECK_THROWS_AS(max_inscribed_diagonal_ellipsoid(redundant_body),
                        std::invalid_argument);

        OP transformed(reduced);
        MT transform = MT::Identity(3, 3);
        transform(0, 0) = 2.0;
        transformed.linear_transformIt(transform);
        CHECK_FALSE(transformed.has_standard_order_facets());
        CHECK_THROWS_AS(max_inscribed_diagonal_ellipsoid(transformed),
                        std::invalid_argument);

        Point const center = make_point({0.2, 0.5, 0.8});
        VT const diagonal = VT::Ones(3);
        MT const dense = MT::Identity(3, 3);
        OrderPolytopeDiagonalRounding<Point> const diagonal_metric(
            reduced, center, diagonal);
        OrderPolytopeSPDRounding<Point> const spd_metric(
            reduced, center, dense);
        CHECK_THROWS_AS(
            OrderPolytopeDiagonalRounding<Point>(
                transformed, center, diagonal), std::invalid_argument);
        CHECK_THROWS_AS(
            OrderPolytopeSPDRounding<Point>(transformed, center, dense),
            std::invalid_argument);
        CHECK_THROWS_AS(DiagonalBody(transformed, diagonal_metric),
                        std::invalid_argument);
        CHECK_THROWS_AS(SPDBody(transformed, spd_metric),
                        std::invalid_argument);
        CHECK_THROWS_AS(spd_metric.metric_distances(transformed),
                        std::invalid_argument);
    }
}

TEST_CASE("diagonal_rounding_metric")
{
    Poset const chain = make_poset(3, {{0, 1}, {1, 2}});
    Point const center = make_point({0.2, 0.5, 0.8});
    VT const shape = make_vector({0.25, 4.0, 9.0});
    OrderPolytopeDiagonalRounding<Point> const metric(chain, center, shape);

    CHECK(metric.dimension() == 3);
    CHECK(metric.centered_quadratic_statistic(make_point({0.5, -0.4, -0.6})) ==
          doctest::Approx(1.08).epsilon(1e-14));

    auto const& distances = metric.facet_metric_distances();
    REQUIRE(distances.size() == 4);
    CHECK(distances[0] == doctest::Approx(0.4).epsilon(1e-14));
    CHECK(distances[1] == doctest::Approx(0.2 / 3.0).epsilon(1e-14));
    CHECK(distances[2] ==
          doctest::Approx(0.3 / std::sqrt(4.25)).epsilon(1e-14));
    CHECK(distances[3] ==
          doctest::Approx(0.3 / std::sqrt(13.0)).epsilon(1e-14));

    OP normalized(chain);
    normalized.normalize();
    OrderPolytopeDiagonalRounding<Point> const normalized_metric(
        normalized, center, shape);
    REQUIRE(normalized_metric.facet_metric_distances().size() ==
            distances.size());
    for (unsigned int i = 0; i < distances.size(); ++i)
        CHECK(normalized_metric.facet_metric_distances()[i] ==
              doctest::Approx(distances[i]).epsilon(1e-14));
}

TEST_CASE("diagonal_rounding_validation")
{
    Poset const chain = make_poset(3, {{0, 1}, {1, 2}});
    Point const center = make_point({0.2, 0.5, 0.8});
    VT const shape = make_vector({0.25, 4.0, 9.0});

    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
                        chain, make_point({0.2, 0.5}), shape),
                    std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
                        chain, center, make_vector({0.25, 0.0, 9.0})),
                    std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
                        chain, make_point({0.2, 0.2, 0.8}), shape),
                    std::invalid_argument);
    CHECK_THROWS_AS(OrderPolytopeDiagonalRounding<Point>(
                        chain,
                        make_point({0.2,
                                    std::numeric_limits<double>::quiet_NaN(),
                                    0.8}),
                        shape),
                    std::invalid_argument);
}

TEST_CASE("diagonal_rounding_normalizer_and_restriction")
{
    Poset const antichain = make_poset(3, {});
    OrderPolytopeDiagonalRounding<Point> const metric(
        antichain, make_point({0.2, 0.5, 0.8}),
        make_vector({4.0, 9.0, 16.0}));

    double const pi = std::acos(-1.0);
    CHECK(metric.gaussian_log_normalizer(pi) ==
          doctest::Approx(std::log(24.0)).epsilon(1e-14));

    Poset const residual = make_poset(2, {{0, 1}});
    auto const restricted = metric.restrict_to(residual, {0, 2});
    OP const residual_body = OP::from_reduced_relations(residual);
    auto const reused = metric.restrict_to(residual_body, {0, 2});
    CHECK(restricted.center()[0] == doctest::Approx(0.2));
    CHECK(restricted.center()[1] == doctest::Approx(0.8));
    CHECK(restricted.shape_diag()(0) == doctest::Approx(4.0));
    CHECK(restricted.shape_diag()(1) == doctest::Approx(16.0));
    CHECK(reused.center()[0] == doctest::Approx(restricted.center()[0]));
    CHECK(reused.center()[1] == doctest::Approx(restricted.center()[1]));
    CHECK(reused.shape_diag()(0) ==
          doctest::Approx(restricted.shape_diag()(0)));
    CHECK(reused.shape_diag()(1) ==
          doctest::Approx(restricted.shape_diag()(1)));
    CHECK_THROWS_AS(metric.restrict_to(residual, {0, 3}),
                    std::invalid_argument);
}

TEST_CASE("diagonal_rounding_shifted_walk")
{
    Poset const singleton = make_poset(1, {});
    OP original(singleton);
    OrderPolytopeDiagonalRounding<Point> const metric(
        original, make_point({0.25}), make_vector({1.0}));
    DiagonalBody body(original, metric);

    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, ScriptedGaussianRNG>;
    Point point(1);
    ScriptedGaussianRNG rng({0.0, 1.0}, {0.0, 0.8});
    Policy::parameters parameters(1.25, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(body.is_in(point, 1e-12) == -1);
    CHECK(point[0] == doctest::Approx(0.6412484483966816).epsilon(1e-12));
}

TEST_CASE("diagonal_rounding_metric_reflection")
{
    Poset const chain = make_poset(2, {{0, 1}});
    OP original(chain);
    OrderPolytopeDiagonalRounding<Point> const metric(
        original, make_point({0.25, 0.75}), make_vector({1.0, 4.0}));
    DiagonalBody body(original, metric);
    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, ScriptedGaussianRNG>;
    Point point(2);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, 1.0, -0.5}, {0.0, 0.7});
    Policy::parameters parameters(0.5, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(point[0] ==
          doctest::Approx(0.26763319550084186).epsilon(1e-12));
    CHECK(point[1] ==
          doctest::Approx(-0.04183935963701341).epsilon(1e-12));
    CHECK(body.is_in(point, 0.0) == -1);
}

TEST_CASE("diagonal_rounding_near_tangent")
{
    Poset const singleton = make_poset(1, {});
    OP original(singleton);
    OrderPolytopeDiagonalRounding<Point> const metric(
        original, make_point({0.5}), make_vector({1.0}));
    DiagonalBody body(original, metric);
    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, ScriptedGaussianRNG>;
    double const amplitude = 1.0 + 1e-10;
    Point point(1);
    ScriptedGaussianRNG rng(
        {0.0, 0.5 * amplitude}, {0.0, std::acos(-1.0) / 4.0});
    Policy::parameters parameters(2.0, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(point[0] == doctest::Approx(0.49999999985).epsilon(1e-12));
    CHECK(body.is_in(point, 0.0) == -1);
}

TEST_CASE("diagonal_rounding_exact_shared_contact")
{
    Poset const chain = make_poset(2, {{0, 1}});
    OP original(chain);
    OrderPolytopeDiagonalRounding<Point> const metric(
        original,
        make_point({0x1p-3, 0x1.0000000000001p-2}),
        make_vector({1.0, 1.0}));
    DiagonalBody body(original, metric);
    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, ScriptedGaussianRNG>;
    Point point = make_point({
        0x1.f1d3ed527e540p-3, 0x1.abead4f5903c2p-3});
    Point const initial(point);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, 0x1.ae5de15ca6c9ep-1, 0x1.70cc35ce18264p-1},
        {0.0, 1.0});
    double const length = std::atan2(24.0, 7.0) + 0.01;
    Policy::parameters parameters(length, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);

    CHECK_THROWS_WITH_AS(
        walk.apply(body, point, 0.5, 1, rng),
        doctest::Contains("simultaneous contact"), std::runtime_error);
    CHECK(point[0] == initial[0]);
    CHECK(point[1] == initial[1]);
}

TEST_CASE("diagonal_rounding_exact_shifted_feasibility")
{
    Poset const singleton = make_poset(1, {});
    OP original(singleton);
    double const center = 0x1.999999999999ap-4;
    double const centered_point = 0x1.ccccccccccccdp-1;
    OrderPolytopeDiagonalRounding<Point> const metric(
        original, make_point({center}), make_vector({1.0}));
    DiagonalBody body(original, metric);
    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, ScriptedGaussianRNG>;
    Point point(1, {centered_point});
    ScriptedGaussianRNG rng({0.0}, {0.0});
    Policy::parameters parameters(0.0, true, 100, true);

    CHECK(body.is_in(point, 0.0) == -1);
    CHECK_THROWS_AS(Walk(body, point, 0.5, rng, parameters),
                    std::runtime_error);
}

TEST_CASE("diagonal_rounding_walk_containment")
{
    Poset const poset = make_poset(
        5, {{0, 1}, {0, 3}, {2, 3}, {2, 4}});
    OP original(poset);
    OrderPolytopeDiagonalRounding<Point> const metric(
        original, make_point({0.15, 0.45, 0.25, 0.7, 0.8}),
        make_vector({0.5, 2.0, 1.0, 3.0, 4.0}));
    DiagonalBody body(original, metric);

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 90817>;
    using Policy = OrderPolytopeDiagonalExactHMCWalk;
    using Walk = Policy::Walk<DiagonalBody, RNG>;
    RNG rng(poset.num_elem());
    Point point(poset.num_elem());
    Walk walk(body, point, 0.5, rng);

    for (unsigned int i = 0; i < 200; ++i) {
        walk.apply(body, point, 0.5, 1, rng);
        CHECK(body.is_in(point, 1e-10) == -1);
    }
}

TEST_CASE("diagonal_rounding_counting_exact")
{
    Poset const poset = make_poset(4, {{0, 1}, {0, 3}, {2, 3}});
    OrderPolytopeDiagonalRounding<Point> const metric(
        poset, make_point({0.2, 0.5, 0.3, 0.7}),
        make_vector({1.0, 1.0, 1.0, 1.0}));
    LinearExtensionOptions options;
    options.reduction.exact_dp_max_n = 20;

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 97531>;
    RNG rng(poset.num_elem());
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(
            poset, rng, metric, options);
    CHECK(result.reduction_exact);
    CHECK(result.residual_count == 0);
    CHECK(result.log_extensions ==
          doctest::Approx(std::log(5.0)).epsilon(1e-14));
}

TEST_CASE("diagonal_rounding_counting_residual")
{
    Poset const poset = make_poset(4, {{0, 1}, {0, 3}, {2, 3}});
    OrderPolytopeDiagonalRounding<Point> const metric(
        poset, make_point({0.2, 0.5, 0.3, 0.7}),
        make_vector({4.0, 9.0, 16.0, 25.0}));
    LinearExtensionOptions options;
    options.error = 0.5;
    options.reduction.exact_dp_max_n = 0;

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 424242>;
    RNG rng(poset.num_elem());
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(
            poset, rng, metric, options);
    CHECK_FALSE(result.reduction_exact);
    CHECK(result.residual_count == 1);
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::abs(result.log_extensions - std::log(5.0)) < 1.0);
}

TEST_CASE("spd_rounding_metric_validation_and_restriction")
{
    Poset const chain = make_poset(2, {{0, 1}});
    MT shape(2, 2);
    shape << 2.0, 0.5,
             0.5, 1.0;
    OrderPolytopeSPDRounding<Point> const metric(
        chain, make_point({0.2, 0.7}), shape);

    CHECK(metric.centered_quadratic_statistic(make_point({1.0, 2.0})) ==
          doctest::Approx(4.0).epsilon(1e-14));
    CHECK(metric.gaussian_log_normalizer(std::acos(-1.0)) ==
          doctest::Approx(0.5 * std::log(1.75)).epsilon(1e-14));
    auto const& distances = metric.facet_metric_distances();
    REQUIRE(distances.size() == 3);
    CHECK(distances.back() ==
          doctest::Approx(0.5 / std::sqrt(2.0)).epsilon(1e-14));

    Poset const singleton = make_poset(1, {});
    auto const restricted = metric.restrict_to(singleton, {1});
    CHECK(restricted.center()[0] == doctest::Approx(0.7));
    CHECK(restricted.shape()(0, 0) == doctest::Approx(1.0));

    MT asymmetric = shape;
    asymmetric(0, 1) = 0.25;
    CHECK_THROWS_AS(OrderPolytopeSPDRounding<Point>(
                        chain, make_point({0.2, 0.7}), asymmetric),
                    std::invalid_argument);
    MT indefinite(2, 2);
    indefinite << 1.0, 2.0,
                  2.0, 1.0;
    CHECK_THROWS_AS(OrderPolytopeSPDRounding<Point>(
                        chain, make_point({0.2, 0.7}), indefinite),
                    std::invalid_argument);
}

TEST_CASE("spd_rounding_near_singular_cover_metric")
{
    Poset const chain = make_poset(2, {{0, 1}});
    MT shape(2, 2);
    shape << 1.0, 1.0,
             1.0, std::nextafter(1.0, 2.0);
    OrderPolytopeSPDRounding<Point> const metric(
        chain, make_point({0.25, 0.75}), shape);

    double const cover_distance = metric.facet_metric_distances().back();
    CHECK(std::isfinite(cover_distance));
    CHECK(cover_distance > 0.0);
}

TEST_CASE("spd_rounding_dense_wall_reflection")
{
    Poset const antichain = make_poset(2, {});
    OP original(antichain);
    MT shape(2, 2);
    shape << 1.0, 0.5,
             0.5, 1.0;
    OrderPolytopeSPDRounding<Point> const metric(
        original, make_point({0.5, 0.5}), shape);
    SPDBody body(original, metric);

    using Policy = OrderPolytopeSPDExactHMCWalk;
    using Walk = Policy::Walk<SPDBody, ScriptedGaussianRNG>;
    Point point(2);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, -0.75, 0.4330127018922193}, {0.0, 1.0});
    Policy::parameters parameters(1.0, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(point[0] ==
          doctest::Approx(-0.33259497937302324).epsilon(1e-11));
    CHECK(point[1] ==
          doctest::Approx(0.14925412961644957).epsilon(1e-11));
    CHECK(body.is_in(point, 0.0) == -1);
}

TEST_CASE("spd_rounding_uses_shifted_cover_offset")
{
    Poset const chain = make_poset(2, {{0, 1}});
    OP original(chain);
    MT shape(2, 2);
    shape << 1.0, 0.25,
             0.25, 2.0;
    OrderPolytopeSPDRounding<Point> const metric(
        original, make_point({0.2, 0.7}), shape);
    SPDBody body(original, metric);

    using Policy = OrderPolytopeSPDExactHMCWalk;
    using Walk = Policy::Walk<SPDBody, ScriptedGaussianRNG>;
    Point point = make_point({-0.05, -0.15});
    ScriptedGaussianRNG rng(
        {0.0, 0.0, 0.5, -0.3053290134455173}, {0.0, 1.0});
    Policy::parameters parameters(1.2, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(point[0] ==
          doctest::Approx(0.21662434917006002).epsilon(1e-11));
    CHECK(point[1] ==
          doctest::Approx(0.2056816585810099).epsilon(1e-11));
    CHECK(body.is_in(point, 0.0) == -1);
}

TEST_CASE("spd_identity_matches_diagonal_walk")
{
    Poset const antichain = make_poset(2, {});
    OP original(antichain);
    Point const center = make_point({0.5, 0.5});
    MT shape = MT::Identity(2, 2);
    OrderPolytopeSPDRounding<Point> const spd_metric(
        original, center, shape);
    OrderPolytopeDiagonalRounding<Point> const diagonal_metric(
        original, center, make_vector({1.0, 1.0}));
    SPDBody spd_body(original, spd_metric);
    DiagonalBody diagonal_body(
        original, diagonal_metric);

    using SPDPolicy = OrderPolytopeSPDExactHMCWalk;
    using DiagonalPolicy = OrderPolytopeDiagonalExactHMCWalk;
    using SPDWalk = SPDPolicy::Walk<SPDBody, ScriptedGaussianRNG>;
    using DiagonalWalk = DiagonalPolicy::Walk<
        DiagonalBody, ScriptedGaussianRNG>;
    ScriptedGaussianRNG spd_rng(
        {0.0, 0.0, -0.75, -0.2}, {0.0, 1.0});
    ScriptedGaussianRNG diagonal_rng(
        {0.0, 0.0, -0.75, -0.2}, {0.0, 1.0});
    Point spd_point(2), diagonal_point(2);
    SPDPolicy::parameters spd_parameters(1.0, true, 100, true);
    DiagonalPolicy::parameters diagonal_parameters(1.0, true, 100, true);
    SPDWalk spd_walk(spd_body, spd_point, 0.5, spd_rng, spd_parameters);
    DiagonalWalk diagonal_walk(
        diagonal_body, diagonal_point, 0.5, diagonal_rng,
        diagonal_parameters);
    spd_walk.apply(spd_body, spd_point, 0.5, 1, spd_rng);
    diagonal_walk.apply(
        diagonal_body, diagonal_point, 0.5, 1, diagonal_rng);

    CHECK(spd_point[0] ==
          doctest::Approx(diagonal_point[0]).epsilon(1e-13));
    CHECK(spd_point[1] ==
          doctest::Approx(diagonal_point[1]).epsilon(1e-13));
}

TEST_CASE("spd_nonorthogonal_contact_rollback")
{
    Poset const antichain = make_poset(2, {});
    OP original(antichain);
    MT shape(2, 2);
    shape << 1.0, 0.5,
             0.5, 1.0;
    OrderPolytopeSPDRounding<Point> const metric(
        original, make_point({0.5, 0.5}), shape);
    SPDBody body(original, metric);

    using Policy = OrderPolytopeSPDExactHMCWalk;
    using Walk = Policy::Walk<SPDBody, ScriptedGaussianRNG>;
    Point point(2);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, -0.75, -0.4330127018922193}, {0.0, 1.0});
    Policy::parameters parameters(1.0, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);

    CHECK_THROWS_AS(walk.apply(body, point, 0.5, 1, rng),
                    std::runtime_error);
    CHECK(point[0] == 0.0);
    CHECK(point[1] == 0.0);
}

TEST_CASE("spd_rounding_dense_reflection_rebuilds_all_events")
{
    Poset const antichain = make_poset(2, {});
    OP original(antichain);
    MT shape(2, 2);
    shape << 1.0, 0.9,
             0.9, 1.0;
    OrderPolytopeSPDRounding<Point> const metric(
        original, make_point({0.5, 0.5}), shape);
    SPDBody body(original, metric);

    using Policy = OrderPolytopeSPDExactHMCWalk;
    using Walk = Policy::Walk<SPDBody, ScriptedGaussianRNG>;
    Point point(2);
    ScriptedGaussianRNG rng(
        {0.0, 0.0, -0.75, 0.675 / std::sqrt(0.19)}, {0.0, 1.0});
    Policy::parameters parameters(1.4, true, 100, true);
    Walk walk(body, point, 0.5, rng, parameters);
    walk.apply(body, point, 0.5, 1, rng);

    CHECK(body.is_in(point, 0.0) == -1);
    CHECK(point[1] < 0.5);
}

TEST_CASE("spd_rounding_walk_containment")
{
    Poset const poset = make_poset(
        5, {{0, 1}, {0, 3}, {2, 3}, {2, 4}});
    OP original(poset);
    MT factor(5, 5);
    factor << 1.0, 0.0, 0.0, 0.0, 0.0,
              0.2, 1.1, 0.0, 0.0, 0.0,
             -0.1, 0.1, 0.9, 0.0, 0.0,
              0.3, 0.0, 0.2, 1.2, 0.0,
              0.0, 0.2, 0.1, 0.1, 1.0;
    MT const shape = factor * factor.transpose();
    OrderPolytopeSPDRounding<Point> const metric(
        original, make_point({0.15, 0.45, 0.25, 0.7, 0.8}), shape);
    SPDBody body(original, metric);

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 61091>;
    using Policy = OrderPolytopeSPDExactHMCWalk;
    using Walk = Policy::Walk<SPDBody, RNG>;
    RNG rng(poset.num_elem());
    Point point(poset.num_elem());
    Walk walk(body, point, 0.5, rng);

    for (unsigned int i = 0; i < 500; ++i) {
        walk.apply(body, point, 0.5, 1, rng);
        CHECK(body.is_in(point, 1e-10) == -1);
    }
}

TEST_CASE("spd_rounding_counting_exact")
{
    Poset const poset = make_poset(4, {{0, 1}, {0, 3}, {2, 3}});
    MT shape(4, 4);
    shape << 2.0, 0.1, 0.0, 0.0,
             0.1, 2.0, 0.1, 0.0,
             0.0, 0.1, 2.0, 0.1,
             0.0, 0.0, 0.1, 2.0;
    OrderPolytopeSPDRounding<Point> const metric(
        poset, make_point({0.2, 0.5, 0.3, 0.7}), shape);
    LinearExtensionOptions options;
    options.reduction.exact_dp_max_n = 20;

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 86420>;
    RNG rng(poset.num_elem());
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(
            poset, rng, metric, options);
    CHECK(result.reduction_exact);
    CHECK(result.log_extensions ==
          doctest::Approx(std::log(5.0)).epsilon(1e-14));
}

TEST_CASE("spd_rounding_counting_residual")
{
    Poset const poset = make_poset(4, {{0, 1}, {0, 3}, {2, 3}});
    MT shape(4, 4);
    shape << 2.0, 0.1, 0.0, 0.0,
             0.1, 2.0, 0.1, 0.0,
             0.0, 0.1, 2.0, 0.1,
             0.0, 0.0, 0.1, 2.0;
    OrderPolytopeSPDRounding<Point> const metric(
        poset, make_point({0.2, 0.5, 0.3, 0.7}), shape);
    LinearExtensionOptions options;
    options.error = 0.5;
    options.reduction.exact_dp_max_n = 0;

    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 75319>;
    RNG rng(poset.num_elem());
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(
            poset, rng, metric, options);
    CHECK_FALSE(result.reduction_exact);
    CHECK(result.residual_count == 1);
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::abs(result.log_extensions - std::log(5.0)) < 1.0);
}
