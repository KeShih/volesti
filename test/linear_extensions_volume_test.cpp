// VolEsti (volume computation and sampling library)

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include <cmath>
#include <limits>

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


// ---- Backend-selectable log-count wrapper (general Gaussian e2e) -------

TEST_CASE("le_log_mean_exp") {
    typedef double NT;

    // log(mean(exp(log_w))), never mean(log_w).
    std::vector<NT> lw = {std::log(1.0), std::log(2.0), std::log(4.0)};
    NT const lme = log_mean_exp(lw);
    CHECK(std::abs(lme - std::log(7.0 / 3.0)) < 1e-14);
    NT mean_log = (lw[0] + lw[1] + lw[2]) / 3.0;
    CHECK(std::abs(lme - mean_log) > 0.1);   // the wrong estimator differs

    // Stable under large negative offsets (raw weights underflow).
    std::vector<NT> tiny = {-1000.0, -1000.0 + std::log(3.0)};
    NT const lme2 = log_mean_exp(tiny);
    CHECK(std::isfinite(lme2));
    CHECK(std::abs(lme2 - (-1000.0 + std::log(2.0))) < 1e-12);

    // Degenerate inputs.
    std::vector<NT> one = {0.5};
    CHECK(std::abs(log_mean_exp(one) - 0.5) < 1e-15);
    std::vector<NT> none;
    CHECK(log_mean_exp(none) == -std::numeric_limits<NT>::infinity());
}

TEST_CASE("le_options_exact_posets") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 101> RNGType;

    // Chain: e(P) = 1, so log_count = 0; antichain: e(P) = n!.  Both are
    // reduced exactly, so the wrapper must return the identities to
    // machine precision, for every backend selection (no sampling runs).
    for (auto backend : {GaussianHMCBackend::Auto,
                         GaussianHMCBackend::MatchedDiagonalEventQueue,
                         GaussianHMCBackend::MatchedDenseAngleFullScan,
                         GaussianHMCBackend::ModalDenseFallback}) {
        LinearExtensionOptions options;
        options.backend = backend;

        RNGType rng1(5);
        Poset chain = make_test_chain(5);
        auto const rc = estimate_log_linear_extensions<RNGType, Point>(
            chain, rng1, options);
        CHECK(std::abs(rc.log_extensions) < 1e-9);
        CHECK(rc.backend == backend);

        RNGType rng2(5);
        Poset anti = make_test_antichain(5);
        auto const ra = estimate_log_linear_extensions<RNGType, Point>(
            anti, rng2, options);
        CHECK(std::abs(ra.log_extensions - std::lgamma(6.0)) < 1e-9);
    }
}

TEST_CASE("le_options_backend_smoke") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 103> RNGType;

    // The N poset on 4 elements is not series-parallel; with the exact
    // small-poset DP disabled its volume is GENUINELY sampled through
    // Gaussian cooling (asserted below via the residual count), driving
    // each cooling adapter end to end; e(N) = 5.
    Poset::RV rels{{0, 1}, {0, 3}, {2, 3}};
    Poset nposet(4, rels);
    NT const log5 = std::log(5.0);

    for (auto backend : {GaussianHMCBackend::SphericalEventQueue,
                         GaussianHMCBackend::MatchedDiagonalEventQueue,
                         GaussianHMCBackend::MatchedDenseAngleFullScan,
                         GaussianHMCBackend::ModalDenseFallback}) {
        LinearExtensionOptions options;
        options.backend = backend;
        options.error = 0.3;
        options.repetitions = 2;
        options.reduction.enable_small_exact_dp = false;

        RNGType rng(4);
        auto const r = estimate_log_linear_extensions<RNGType, Point>(
            nposet, rng, options);
        CHECK(r.residual_count >= 1);            // sampling really ran
        CHECK(std::isfinite(r.log_extensions));
        CHECK(r.backend == backend);
        CHECK(r.repetitions == 2);
        CHECK(std::abs(r.log_extensions - log5) < 0.6);
        CHECK(r.se_log_volume >= 0.0);
        CHECK(std::isfinite(r.se_log_volume));
    }
}

TEST_CASE("le_options_log_domain_no_underflow") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 107> RNGType;

    // chain-200: the raw volume 1/200! underflows double entirely, but
    // the log-domain pipeline stays exact and finite.
    RNGType rng(200);
    Poset chain = make_test_chain(200);
    LinearExtensionOptions options;
    auto const r = estimate_log_linear_extensions<RNGType, Point>(
        chain, rng, options);
    CHECK(std::isfinite(r.log_volume));
    CHECK(std::abs(r.log_volume + std::lgamma(201.0)) < 1e-6);
    CHECK(std::abs(r.log_extensions) < 1e-6);
    CHECK(std::exp(r.log_volume) == 0.0);       // raw volume underflows
}


TEST_CASE("le_options_generic_cdhr") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 109> RNGType;

    // Backend consistency: the classic generic CDHR sampler and the HMC
    // backends must agree on the genuinely sampled N poset (e = 5).
    Poset::RV rels{{0, 1}, {0, 3}, {2, 3}};
    Poset nposet(4, rels);

    LinearExtensionOptions options;
    options.use_generic_cdhr = true;
    options.error = 0.3;
    options.repetitions = 2;
    options.reduction.enable_small_exact_dp = false;

    RNGType rng(4);
    auto const r = estimate_log_linear_extensions<RNGType, Point>(
        nposet, rng, options);
    CHECK(r.residual_count >= 1);
    CHECK(r.used_generic_cdhr);
    CHECK(std::isfinite(r.log_extensions));
    CHECK(std::abs(r.log_extensions - std::log(5.0)) < 0.6);
}

TEST_CASE("le_options_dp_reference_posets") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 113> RNGType;

    // Sampled log-volume vs the exact small-poset DP reference on the
    // diamond and a sparse 5-element poset.
    std::vector<std::pair<unsigned int, Poset::RV>> posets = {
        {4, {{0, 1}, {0, 2}, {1, 3}, {2, 3}}},
        {5, {{0, 2}, {1, 2}, {2, 4}, {3, 4}}},
    };
    for (auto& pr : posets) {
        Poset poset(pr.first, pr.second);
        NT const dp_ref = exact_dp_log_volume(poset);

        LinearExtensionOptions options;
        options.error = 0.25;
        options.repetitions = 2;
        // Disable every reduction: these posets are series-parallel, so
        // anything less leaves nothing for the sampler to do.
        options.reduction.enable_small_exact_dp = false;
        options.reduction.enable_chain_antichain = false;
        options.reduction.enable_parallel_decomposition = false;
        options.reduction.enable_ordinal_decomposition = false;

        RNGType rng(pr.first);
        auto const r = estimate_log_linear_extensions<RNGType, Point>(
            poset, rng, options);
        CHECK(r.residual_count >= 1);
        CHECK(std::isfinite(r.log_volume));
        CHECK(std::abs(r.log_volume - dp_ref) < 0.35);
    }
}

TEST_CASE("le_log_cooling_stage_diagnostics") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 127> RNGType;

    // Direct log-domain cooling on O(N): the bookkeeping identity
    // log_volume = log_first_gaussian + sum(stage log-ratios) holds, the
    // per-stage diagnostics are sane, no sample leaves the polytope, and
    // the estimate matches vol(O(N)) = 5/4! = 5/24.
    Poset::RV rels{{0, 1}, {0, 3}, {2, 3}};
    Poset nposet(4, rels);
    OP_t OP(nposet);

    RNGType rng(4);
    auto const r = volume_cooling_gaussians_log
        <OrderPolytopeGaussianHamiltonianMonteCarloExactWalk>(
            OP, rng, 0.2, 1, true);

    REQUIRE(r.success);
    REQUIRE(r.stages.size() >= 1);
    NT sum = r.log_first_gaussian;
    unsigned long long samples = 0;
    for (auto const& s : r.stages) {
        CHECK(std::isfinite(s.log_ratio));
        CHECK(s.samples > 0);
        CHECK(s.cv2 > -1e-9);
        CHECK(s.a_next < s.a_curr);
        sum += s.log_ratio;
        samples += s.samples;
    }
    CHECK(std::abs(sum - r.log_volume) < 1e-12);
    CHECK(samples == r.total_samples);
    CHECK(r.violations == 0);
    CHECK(r.sampling_seconds >= 0.0);
    CHECK(std::abs(r.log_volume - std::log(5.0 / 24.0)) < 0.3);

    // Same algorithm, both domains: the linear-domain routine agrees
    // within loose Monte Carlo tolerance.
    RNGType rng2(4);
    double const vol_linear = volume_cooling_gaussians
        <OrderPolytopeGaussianHamiltonianMonteCarloExactWalk>(
            OP, rng2, 0.2, 1);
    CHECK(std::abs(std::log(vol_linear) - r.log_volume) < 0.4);
}

TEST_CASE("le_log_cooling_large_error_terminates") {
    typedef double NT;
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 131> RNGType;

    // A pathologically large error (curr_eps >= 2) must saturate the
    // window criterion instead of feeding a negative argument to log:
    // the linear routine terminates trivially here, so the log routine
    // must too (regression for the window_tol NaN guard).
    Poset::RV rels{{0, 1}, {0, 3}, {2, 3}};
    Poset nposet(4, rels);
    OP_t OP(nposet);

    RNGType rng(4);
    auto const r = volume_cooling_gaussians_log
        <OrderPolytopeGaussianHamiltonianMonteCarloExactWalk>(
            OP, rng, 3.0, 1);
    REQUIRE(r.success);
    CHECK(std::isfinite(r.log_volume));
    for (auto const& s : r.stages)
        CHECK(s.samples >= 1);   // finite, bounded stages
}
