// VolEsti (volume computation and sampling library)

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "generators/boost_random_number_generator.hpp"
#include "misc/order_polytope_volume_reduction.h"
#include "misc/poset.h"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "volume/linear_extensions_volume.hpp"
#include "volume/volume_cooling_gaussians_log.hpp"

namespace {

Poset make_poset(unsigned int n, Poset::RV relations)
{
    return Poset(n, relations);
}

Poset chain(unsigned int n)
{
    Poset::RV relations;
    for (unsigned int i = 0; i + 1 < n; ++i)
        relations.push_back(Poset::RT(i, i + 1));
    return Poset(n, relations);
}

Poset antichain(unsigned int n)
{
    return make_poset(n, Poset::RV());
}

Poset n_poset()
{
    return make_poset(4, Poset::RV{{0, 1}, {0, 3}, {2, 3}});
}

struct UnrelatedDeltaWalk
{
    bool updated = false;
    void update_delta(double) { updated = true; }
};

} // namespace

TEST_CASE("le_transitive_reduction")
{
    Poset poset = make_poset(4, Poset::RV{
        {0, 1}, {1, 2}, {2, 3}, {0, 2}, {0, 3}, {0, 1}});
    Poset reduced = poset.transitive_reduction();

    CHECK(reduced.num_relations() == 3);
    CHECK(reduced.get_relation(0) == Poset::RT(0, 1));
    CHECK(reduced.get_relation(1) == Poset::RT(1, 2));
    CHECK(reduced.get_relation(2) == Poset::RT(2, 3));
}

TEST_CASE("le_structural_reductions")
{
    auto const reduced_chain = reduce_order_polytope_volume_problem(chain(64));
    CHECK(reduced_chain.is_exact());
    CHECK(reduced_chain.residual_vertices.empty());
    CHECK(reduced_chain.log_volume_offset ==
          doctest::Approx(-std::lgamma(65.0)).epsilon(1e-12));

    auto const reduced_antichain =
        reduce_order_polytope_volume_problem(antichain(64));
    CHECK(reduced_antichain.is_exact());
    CHECK(reduced_antichain.log_volume_offset == doctest::Approx(0.0));

    // chain(3) in parallel with antichain(2): volume = 1/3!.
    Poset parallel = make_poset(5, Poset::RV{{0, 1}, {1, 2}});
    auto const reduced_parallel =
        reduce_order_polytope_volume_problem(parallel);
    CHECK(reduced_parallel.is_exact());
    CHECK(reduced_parallel.log_volume_offset ==
          doctest::Approx(-std::lgamma(4.0)).epsilon(1e-12));

    // antichain(2) ordinal-sum antichain(3): volume = 2! 3! / 5!.
    Poset::RV ordinal_relations;
    for (unsigned int u : {0u, 1u})
        for (unsigned int v : {2u, 3u, 4u})
            ordinal_relations.push_back(Poset::RT(u, v));
    auto const reduced_ordinal = reduce_order_polytope_volume_problem(
        make_poset(5, ordinal_relations));
    CHECK(reduced_ordinal.is_exact());
    CHECK(reduced_ordinal.log_volume_offset == doctest::Approx(
        std::lgamma(3.0) + std::lgamma(4.0) - std::lgamma(6.0))
        .epsilon(1e-12));
}

TEST_CASE("le_exact_dp_and_residual_maps")
{
    Poset const core = n_poset();

    OrderPolytopeVolumeReductionOptions exact_options;
    exact_options.exact_dp_max_n = 4;
    auto const exact =
        reduce_order_polytope_volume_problem(core, exact_options);
    CHECK(exact.is_exact());
    CHECK(exact.log_volume_offset ==
          doctest::Approx(std::log(5.0) - std::lgamma(5.0)).epsilon(1e-12));

    OrderPolytopeVolumeReductionOptions residual_options;
    residual_options.exact_dp_max_n = 0;
    auto const residual =
        reduce_order_polytope_volume_problem(core, residual_options);
    REQUIRE(residual.residual_posets.size() == 1);
    REQUIRE(residual.residual_vertices.size() == 1);
    CHECK(residual.log_volume_offset == doctest::Approx(0.0));
    CHECK(residual.residual_vertices[0] ==
          std::vector<unsigned int>{0, 1, 2, 3});

    // A disjoint chain on vertices {1,4} is removed exactly, leaving the
    // non-series-parallel core on the non-contiguous coordinates {0,2,3,5}.
    Poset mixed = make_poset(6, Poset::RV{
        {0, 2}, {0, 5}, {3, 5}, {1, 4}});
    auto const mapped =
        reduce_order_polytope_volume_problem(mixed, residual_options);
    REQUIRE(mapped.residual_posets.size() == 1);
    REQUIRE(mapped.residual_vertices.size() == 1);
    CHECK(mapped.residual_vertices[0] ==
          std::vector<unsigned int>{0, 2, 3, 5});
    CHECK(mapped.log_volume_offset ==
          doctest::Approx(-std::lgamma(3.0)).epsilon(1e-12));
}

TEST_CASE("le_default_spherical_exact")
{
    using Point = Cartesian<double>::Point;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 101>;

    LinearExtensionOptions options;
    options.repetitions = 3;

    RNG chain_rng(8);
    auto const chain_result =
        estimate_log_linear_extensions<RNG, Point>(chain(8), chain_rng, options);
    CHECK(chain_result.reduction_exact);
    CHECK(chain_result.residual_count == 0);
    CHECK(chain_result.repetitions == 3);
    CHECK(chain_result.log_volume ==
          doctest::Approx(-std::lgamma(9.0)).epsilon(1e-12));
    CHECK(chain_result.log_extensions == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(chain_result.se_log_volume == doctest::Approx(0.0));

    RNG antichain_rng(8);
    auto const antichain_result =
        estimate_log_linear_extensions<RNG, Point>(antichain(8), antichain_rng);
    CHECK(antichain_result.reduction_exact);
    CHECK(antichain_result.log_volume == doctest::Approx(0.0));
    CHECK(antichain_result.log_extensions ==
          doctest::Approx(std::lgamma(9.0)).epsilon(1e-12));

    RNG invalid_rng(8);
    options.error = 0.0;
    CHECK_THROWS_AS(
        (estimate_log_linear_extensions<RNG, Point>(
            chain(8), invalid_rng, options)),
        std::invalid_argument);
    options.error = 0.1;
    options.walk_length = 0;
    CHECK_THROWS_AS(
        (estimate_log_linear_extensions<RNG, Point>(
            chain(8), invalid_rng, options)),
        std::invalid_argument);
    options.walk_length = 1;
    options.repetitions = 0;
    CHECK_THROWS_AS(
        (estimate_log_linear_extensions<RNG, Point>(
            chain(8), invalid_rng, options)),
        std::invalid_argument);
}

TEST_CASE("le_default_spherical_sampling")
{
    using Point = Cartesian<double>::Point;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 103>;

    LinearExtensionOptions options;
    options.error = 0.3;
    options.repetitions = 2;
    options.reduction.exact_dp_max_n = 0;

    RNG rng(4);
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(n_poset(), rng, options);

    CHECK_FALSE(result.reduction_exact);
    CHECK(result.residual_count == 1);
    CHECK(result.repetitions == 2);
    CHECK(std::isfinite(result.log_volume));
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::isfinite(result.se_log_volume));
    CHECK(result.se_log_volume >= 0.0);
    CHECK(std::abs(result.log_extensions - std::log(5.0)) < 0.6);
}

TEST_CASE("le_diagonal_mvie_reduced_residual")
{
    using Point = Cartesian<double>::Point;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 105>;

    // Two interleaved N-posets give two non-contiguous residuals and
    // C(8,4) * 5 * 5 = 1750 linear extensions.
    Poset const poset = make_poset(8, Poset::RV{
        {0, 2}, {0, 6}, {4, 6},
        {1, 3}, {1, 7}, {5, 7}});
    LinearExtensionOptions options;
    options.error = 0.5;
    options.repetitions = 2;
    options.reduction.exact_dp_max_n = 0;

    RNG rng(8);
    auto const result =
        estimate_log_linear_extensions_diagonal_mvie<RNG, Point>(
            poset, rng, options);

    CHECK_FALSE(result.reduction_exact);
    CHECK(result.residual_count == 2);
    CHECK(result.repetitions == 2);
    CHECK(std::isfinite(result.log_volume));
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::abs(result.log_extensions - std::log(1750.0)) < 1.0);
}

TEST_CASE("le_log_domain_underflow")
{
    using Point = Cartesian<double>::Point;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 107>;

    // The 200-chain is reduced exactly; only the disjoint N core is sampled.
    // Its volume is 5 / (24 * 200!), below the range of double, while the log
    // count remains finite.
    Poset::RV relations;
    for (unsigned int i = 0; i + 1 < 200; ++i)
        relations.push_back(Poset::RT(i, i + 1));
    relations.push_back(Poset::RT(200, 201));
    relations.push_back(Poset::RT(200, 203));
    relations.push_back(Poset::RT(202, 203));
    Poset poset(204, relations);

    LinearExtensionOptions options;
    options.error = 0.3;
    options.reduction.exact_dp_max_n = 0;

    RNG rng(204);
    auto const result =
        estimate_log_linear_extensions<RNG, Point>(poset, rng, options);
    double const expected_log_volume =
        std::log(5.0 / 24.0) - std::lgamma(201.0);
    double const expected_log_count =
        expected_log_volume + std::lgamma(205.0);

    CHECK(result.residual_count == 1);
    CHECK(std::isfinite(result.log_volume));
    CHECK(result.volume_estimate == 0.0);
    CHECK(std::abs(result.log_volume - expected_log_volume) < 0.6);
    CHECK(std::abs(result.log_extensions - expected_log_count) < 0.6);
}

TEST_CASE("le_log_cooling_budget")
{
    using Point = Cartesian<double>::Point;
    using Polytope = OrderPolytope<Point>;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 109>;

    Polytope body(n_poset());
    RNG rng(4);
    auto const exhausted = volume_cooling_gaussians_log<
        OrderPolytopeExactHMCWalk>(
            body, rng, 0.3, 1, 1);
    CHECK_FALSE(exhausted.success);
    CHECK(std::isnan(exhausted.log_volume));

    CHECK_THROWS_AS(
        (volume_cooling_gaussians_log<
            OrderPolytopeExactHMCWalk>(
                body, rng, 0.3, 1, 0)),
        std::invalid_argument);
}

TEST_CASE("le_log_cooling_parameter_helpers")
{
    UnrelatedDeltaWalk walk;
    volume_cooling_gaussians_log_detail::set_walk_delta(walk, 1.0);
    CHECK_FALSE(walk.updated);

    unsigned int sample_count = 0;
    CHECK(volume_cooling_gaussians_log_detail::schedule_sample_count(
        0.1f, 0.3f, sample_count));
    CHECK(sample_count > 0);
    CHECK_FALSE(volume_cooling_gaussians_log_detail::schedule_sample_count(
        0.1f, 1.0e-12f, sample_count));
}

TEST_CASE("le_general_mass_counting")
{
    using Point = Cartesian<double>::Point;
    using Polytope = OrderPolytope<Point>;
    using Matrix = Polytope::MT;
    using RNG = BoostRandomNumberGenerator<boost::mt19937, double, 113>;

    Poset mixed = make_poset(6, Poset::RV{
        {0, 2}, {0, 5}, {3, 5}, {1, 4}});
    Matrix mass = Matrix::Identity(6, 6);
    for (Eigen::Index i = 0; i < mass.rows(); ++i)
        mass(i, i) = 1.5 + 0.2 * double(i);
    for (Eigen::Index i = 0; i < mass.rows(); ++i) {
        for (Eigen::Index j = i + 1; j < mass.cols(); ++j)
            mass(i, j) = mass(j, i) = 0.01 * double(i + j + 1);
    }

    LinearExtensionOptions options;
    options.error = 0.3;
    options.reduction.exact_dp_max_n = 0;

    RNG rng(6);
    auto const result = estimate_log_linear_extensions_mass<RNG, Point>(
        mixed, rng, mass, options);
    CHECK_FALSE(result.reduction_exact);
    CHECK(result.residual_count == 1);
    CHECK(std::isfinite(result.log_volume));
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::abs(result.log_extensions - std::log(75.0)) < 0.8);

    Matrix wrong_dimension = Matrix::Identity(5, 5);
    CHECK_THROWS_AS(
        (estimate_log_linear_extensions_mass<RNG, Point>(
            mixed, rng, wrong_dimension, options)),
        std::invalid_argument);

    Matrix indefinite = Matrix::Identity(6, 6);
    indefinite(0, 0) = -1.0;
    CHECK_THROWS_AS(
        (estimate_log_linear_extensions_mass<RNG, Point>(
            mixed, rng, indefinite, options)),
        std::invalid_argument);
}
