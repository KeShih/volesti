// VolEsti (volume computation and sampling library)

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include <cmath>

#include <boost/random.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "misc/poset.h"
#include "generators/boost_random_number_generator.hpp"

#include "misc/order_polytope_volume_reduction.h"
#include "volume/linear_extensions_volume.hpp"

static Poset make_test_chain(unsigned int n)
{
    Poset::RV rels;
    for (unsigned int i = 0; i + 1 < n; ++i) {
        rels.push_back(Poset::RT(i, i + 1));
    }
    return Poset(n, rels);
}

static Poset make_test_antichain(unsigned int n)
{
    Poset::RV rels;
    return Poset(n, rels);
}

static bool has_reduction_step(ReducedOrderPolytopeVolumeProblem const& reduced,
                               OrderPolytopeReductionStepKind kind)
{
    for (auto const& step : reduced.steps) {
        if (step.kind == kind) return true;
    }
    return false;
}

static double exact_dp_log_volume(Poset const& poset)
{
    OrderPolytopeVolumeReductionOptions opts;
    opts.enable_chain_antichain = false;
    opts.enable_parallel_decomposition = false;
    opts.enable_ordinal_decomposition = false;
    opts.enable_small_exact_dp = true;
    opts.exact_dp_max_n = 20;
    auto reduced = reduce_order_polytope_volume_problem(poset, opts);
    CHECK(reduced.is_exact());
    CHECK(has_reduction_step(reduced, OrderPolytopeReductionStepKind::ExactDP));
    return reduced.log_volume_offset;
}


TEST_CASE("transitive_reduction") {
    typedef Poset::RV RV;

    // Chain 0<1<2 with redundant edge 0<2
    {
        RV rels{{0, 1}, {1, 2}, {0, 2}};
        Poset poset(3, rels);
        CHECK(poset.num_relations() == 3);

        Poset reduced = poset.transitive_reduction();
        CHECK(reduced.num_relations() == 2);
    }

    // Diamond with redundant edges
    {
        RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}, {0, 3}};
        Poset poset(4, rels);
        CHECK(poset.num_relations() == 5);

        Poset reduced = poset.transitive_reduction();
        CHECK(reduced.num_relations() == 4);
    }

    // Already minimal
    {
        RV rels{{0, 1}, {1, 2}};
        Poset poset(3, rels);
        Poset reduced = poset.transitive_reduction();
        CHECK(reduced.num_relations() == 2);
    }

    // Antichain
    {
        RV rels;
        Poset poset(3, rels);
        Poset reduced = poset.transitive_reduction();
        CHECK(reduced.num_relations() == 0);
    }

    // Duplicate edges collapse to one
    {
        RV rels{{0, 1}, {0, 1}, {1, 2}};
        Poset poset(3, rels);
        Poset reduced = poset.transitive_reduction();
        CHECK(reduced.num_relations() == 2);
    }
}


TEST_CASE("log_factorial_identity") {
    // Verify lgamma(n+1) matches log(n!) for small n
    double fact = 1.0;
    for (unsigned int n = 1; n <= 12; ++n) {
        fact *= n;
        double log_fact = std::lgamma(n + 1);
        CHECK(std::abs(log_fact - std::log(fact)) < 1e-10);
    }
}


TEST_CASE("linear_extensions_volume_chain") {
    typedef Cartesian<double> Kernel;
    typedef Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 123> RNGType;
    typedef Poset::RV RV;

    // Chain 0<1<2: volume = 1/6, e(P) = 1, log_extensions = 0
    RV rels{{0, 1}, {1, 2}};
    Poset poset(3, rels);

    RNGType rng(3);
    auto result = estimate_log_linear_extensions_spherical_order_hmc<RNGType, Point>(
        poset, rng, 0.3, 1);

    CHECK(result.n == 3);
    CHECK(result.reduced_relations == 2);
    CHECK(result.reduction_exact);
    CHECK(result.residual_count == 0);
    CHECK(std::isfinite(result.log_extensions));
    CHECK(std::isfinite(result.volume_estimate));
    CHECK(result.volume_estimate > 0.0);
    CHECK(std::abs(result.log_volume + std::lgamma(4.0)) < 1e-12);
    CHECK(std::abs(result.log_extensions) < 1e-12);
}


TEST_CASE("linear_extensions_volume_antichain") {
    typedef Cartesian<double> Kernel;
    typedef Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 456> RNGType;
    typedef Poset::RV RV;

    // Antichain n=3: volume = 1, e(P) = 6, log_extensions = log(6)
    RV rels;
    Poset poset(3, rels);

    RNGType rng(3);
    auto result = estimate_log_linear_extensions_spherical_order_hmc<RNGType, Point>(
        poset, rng, 0.3, 1);

    CHECK(result.n == 3);
    CHECK(result.input_relations == 0);
    CHECK(result.reduced_relations == 0);
    CHECK(result.reduction_exact);
    CHECK(result.residual_count == 0);
    CHECK(std::isfinite(result.log_extensions));
    CHECK(result.volume_estimate > 0.0);
    double exact_log = std::log(6.0);
    CHECK(std::abs(result.log_volume) < 1e-12);
    CHECK(std::abs(result.log_extensions - exact_log) < 1e-12);
}


TEST_CASE("linear_extensions_volume_generic_walk") {
    typedef Cartesian<double> Kernel;
    typedef Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 789> RNGType;
    typedef Poset::RV RV;

    // Use GaussianCDHRWalk (the default walk) on a nontrivial order polytope.
    RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    Poset poset(4, rels);

    RNGType rng(4);
    OrderPolytopeVolumeReductionOptions opts;
    opts.enable_chain_antichain = false;
    opts.enable_parallel_decomposition = false;
    opts.enable_ordinal_decomposition = false;
    opts.enable_small_exact_dp = false;
    opts.exact_dp_max_n = 0;
    auto result = estimate_log_linear_extensions<GaussianCDHRWalk, RNGType, Point>(
        poset, rng, 0.3, 1, opts);

    CHECK(!result.reduction_exact);
    CHECK(result.residual_count == 1);
    CHECK(std::isfinite(result.log_extensions));
    CHECK(result.volume_estimate > 0.0);
    double exact_log = std::log(2.0);
    CHECK(std::abs(result.log_extensions - exact_log) < 2.0);
}


TEST_CASE("order_polytope_volume_reduction_chain_exact") {
    typedef Cartesian<double> Kernel;
    typedef Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 321> RNGType;

    for (unsigned int n : {1u, 2u, 3u, 8u, 32u, 64u}) {
        Poset poset = make_test_chain(n);
        auto reduced = reduce_order_polytope_volume_problem(poset);
        double exact_log_volume = -std::lgamma(double(n) + 1.0);

        CHECK(reduced.is_exact());
        CHECK(reduced.residual_posets.empty());
        CHECK(std::abs(reduced.log_volume_offset - exact_log_volume) < 1e-12);

        RNGType rng(n);
        auto result = estimate_log_linear_extensions_spherical_order_hmc
            <RNGType, Point>(poset, rng, 0.3, 1);
        CHECK(result.reduction_exact);
        CHECK(result.residual_count == 0);
        CHECK(std::abs(result.log_volume - exact_log_volume) < 1e-12);
        CHECK(std::abs(result.log_extensions) < 1e-12);
    }
}


TEST_CASE("order_polytope_volume_reduction_antichain_exact") {
    typedef Cartesian<double> Kernel;
    typedef Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, double, 654> RNGType;

    for (unsigned int n : {1u, 2u, 8u, 64u}) {
        Poset poset = make_test_antichain(n);
        auto reduced = reduce_order_polytope_volume_problem(poset);

        CHECK(reduced.is_exact());
        CHECK(reduced.residual_posets.empty());
        CHECK(std::abs(reduced.log_volume_offset) < 1e-12);

        RNGType rng(n);
        auto result = estimate_log_linear_extensions_spherical_order_hmc
            <RNGType, Point>(poset, rng, 0.3, 1);
        CHECK(result.reduction_exact);
        CHECK(result.residual_count == 0);
        CHECK(std::abs(result.log_volume) < 1e-12);
        CHECK(std::abs(result.log_extensions
                       - std::lgamma(double(n) + 1.0)) < 1e-12);
    }
}


TEST_CASE("order_polytope_volume_reduction_parallel_decomposition") {
    // P = chain(3) || antichain(2).
    Poset::RV rels{{0, 1}, {1, 2}};
    Poset poset(5, rels);

    auto reduced = reduce_order_polytope_volume_problem(poset);
    double expected_log_volume = -std::lgamma(4.0);
    double expected_log_extensions = std::log(20.0);

    CHECK(reduced.is_exact());
    CHECK(has_reduction_step(reduced,
                             OrderPolytopeReductionStepKind::ParallelDecomposition));
    CHECK(std::abs(reduced.log_volume_offset - expected_log_volume) < 1e-12);
    CHECK(std::abs(reduced.log_volume_offset + std::lgamma(6.0)
                   - expected_log_extensions) < 1e-12);
}


TEST_CASE("order_polytope_volume_reduction_ordinal_sum_decomposition") {
    // P = antichain(2) + antichain(3), with every first-block element
    // below every second-block element.
    Poset::RV rels;
    for (unsigned int i : {0u, 1u}) {
        for (unsigned int j : {2u, 3u, 4u}) {
            rels.push_back(Poset::RT(i, j));
        }
    }
    Poset poset(5, rels);

    auto reduced = reduce_order_polytope_volume_problem(poset);
    double expected_log_volume = std::lgamma(3.0) + std::lgamma(4.0)
                               - std::lgamma(6.0);
    double expected_log_extensions = std::log(12.0);

    CHECK(reduced.is_exact());
    CHECK(has_reduction_step(reduced,
                             OrderPolytopeReductionStepKind::OrdinalSumDecomposition));
    CHECK(std::abs(reduced.log_volume_offset - expected_log_volume) < 1e-12);
    CHECK(std::abs(reduced.log_volume_offset + std::lgamma(6.0)
                   - expected_log_extensions) < 1e-12);
}


TEST_CASE("order_polytope_volume_reduction_mixed_recursive_decomposition") {
    // P = (chain(3) || singleton) + antichain(2).
    Poset::RV rels{{0, 1}, {1, 2}};
    for (unsigned int i : {0u, 1u, 2u, 3u}) {
        rels.push_back(Poset::RT(i, 4));
        rels.push_back(Poset::RT(i, 5));
    }
    Poset poset(6, rels);

    auto reduced = reduce_order_polytope_volume_problem(poset);
    double dp_log_volume = exact_dp_log_volume(poset);

    CHECK(reduced.is_exact());
    CHECK(has_reduction_step(reduced,
                             OrderPolytopeReductionStepKind::OrdinalSumDecomposition));
    CHECK(has_reduction_step(reduced,
                             OrderPolytopeReductionStepKind::ParallelDecomposition));
    CHECK(std::abs(reduced.log_volume_offset - dp_log_volume) < 1e-12);
}


TEST_CASE("order_polytope_volume_reduction_small_exact_dp") {
    // Diamond: 0 < {1,2} < 3 has exactly two linear extensions.
    Poset::RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    Poset poset(4, rels);

    OrderPolytopeVolumeReductionOptions opts;
    opts.enable_ordinal_decomposition = false;
    auto reduced = reduce_order_polytope_volume_problem(poset, opts);

    double expected_log_volume = std::log(2.0) - std::lgamma(5.0);
    CHECK(reduced.is_exact());
    CHECK(has_reduction_step(reduced, OrderPolytopeReductionStepKind::ExactDP));
    CHECK(std::abs(reduced.log_volume_offset - expected_log_volume) < 1e-12);
}


TEST_CASE("order_polytope_volume_reduction_residual_fallback") {
    // The N poset is connected in both comparability and incomparability
    // graphs, so with exact DP disabled it should remain as one residual core.
    Poset::RV rels{{0, 1}, {0, 3}, {2, 3}};
    Poset poset(4, rels);

    OrderPolytopeVolumeReductionOptions opts;
    opts.enable_small_exact_dp = false;
    opts.exact_dp_max_n = 0;
    auto reduced = reduce_order_polytope_volume_problem(poset, opts);

    CHECK(!reduced.is_exact());
    CHECK(reduced.residual_posets.size() == 1);
    CHECK(reduced.residual_posets[0].num_elem() == 4);
    CHECK(reduced.residual_posets[0].num_relations() == 3);
    CHECK(std::abs(reduced.log_volume_offset) < 1e-12);
    CHECK(has_reduction_step(reduced, OrderPolytopeReductionStepKind::ResidualCore));
}


TEST_CASE("order_polytope_volume_reduction_options") {
    Poset chain64 = make_test_chain(64);

    OrderPolytopeVolumeReductionOptions residual_opts;
    residual_opts.enable_chain_antichain = false;
    residual_opts.enable_ordinal_decomposition = false;
    residual_opts.enable_small_exact_dp = false;
    residual_opts.exact_dp_max_n = 0;
    auto residual = reduce_order_polytope_volume_problem(chain64, residual_opts);

    CHECK(!residual.is_exact());
    CHECK(residual.residual_posets.size() == 1);
    CHECK(residual.residual_posets[0].num_elem() == 64);

    OrderPolytopeVolumeReductionOptions ordinal_opts = residual_opts;
    ordinal_opts.enable_ordinal_decomposition = true;
    auto ordinal = reduce_order_polytope_volume_problem(chain64, ordinal_opts);
    CHECK(ordinal.is_exact());
    CHECK(std::abs(ordinal.log_volume_offset + std::lgamma(65.0)) < 1e-12);

    Poset::RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    Poset diamond(4, rels);
    OrderPolytopeVolumeReductionOptions no_dp_opts;
    no_dp_opts.enable_ordinal_decomposition = false;
    no_dp_opts.enable_small_exact_dp = false;
    no_dp_opts.exact_dp_max_n = 0;
    auto no_dp = reduce_order_polytope_volume_problem(diamond, no_dp_opts);
    CHECK(!no_dp.is_exact());
    CHECK(no_dp.residual_posets.size() == 1);
}
