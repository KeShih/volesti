// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

// Opt-in benchmark isolating the diagonal backend's fixed cost from the
// benefit of a precomputed diagonal metric.  All three arms use the same
// center, cooling implementation, random seeds, and trajectory-time law:
//   S  = spherical event/reflection backend with D=I target statistics;
//   D0 = diagonal backend with D=I;
//   D1 = diagonal backend with a determinant-one precomputed diagonal D.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <boost/random.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "generators/boost_random_number_generator.hpp"
#include "misc/order_polytope_volume_reduction.h"
#include "volume/order_polytope_diagonal_cooling.hpp"

namespace {

typedef Cartesian<double> Kernel;
typedef Kernel::Point Point;
typedef OrderPolytope<Point> OP;
typedef OP::VT VT;
typedef BoostRandomNumberGenerator<boost::mt19937, double> RNG;

struct BenchmarkSphericalDynamics
    : order_polytope_exact_hmc_detail::SphericalDynamics {
    static constexpr bool report_failure_diagnostics = true;

    template <typename Polytope>
    struct State
        : order_polytope_exact_hmc_detail::SphericalDynamics::State<Polytope> {
        typedef order_polytope_exact_hmc_detail::SphericalDynamics::State<Polytope>
            Base;
        typedef typename Polytope::PointType PointType;
        typedef typename PointType::FT NT;

        explicit State(Polytope const& polytope)
            : Base(polytope), diagnostics(polytope.rounding_diagnostics())
        {}

        void count_trajectory() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled) {
                if (diagnostics->active_stage ==
                    OrderPolytopeDiagonalCoolingStage::Schedule)
                    ++diagnostics->schedule_trajectories;
                else
                    ++diagnostics->ratio_trajectories;
            }
        }

        void count_reflection() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++diagnostics->reflection_count;
        }

        void count_event_recomputation() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++diagnostics->event_recomputation_count;
        }

        void count_reflection_limit_rejection() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++diagnostics->rejected_reflection_limit;
        }

        void count_ambiguous_tie_failure() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++diagnostics->ambiguous_tie_failures;
        }

        void count_shared_contact_failure() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++diagnostics->shared_contact_failures;
        }

        void observe_endpoint_violations(PointType const&) const {}

        OrderPolytopeDiagonalCoolingDiagnostics<NT>* diagnostics;
    };
};

typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy<
    BenchmarkSphericalDynamics> BenchmarkSphericalPolicy;

struct MetricTruth {
    Point center;
    VT shape;
    double log_extensions;
};

struct BenchmarkCase {
    std::string name;
    Poset poset;
    MetricTruth truth;
};

struct Sample {
    std::string case_name;
    std::string arm;
    unsigned int repetition = 0;
    unsigned int seed = 0;
    unsigned int dimension = 0;
    unsigned int relations = 0;
    bool success = false;
    double log_extensions = std::numeric_limits<double>::quiet_NaN();
    double log_error = std::numeric_limits<double>::quiet_NaN();
    double elapsed_microseconds = std::numeric_limits<double>::quiet_NaN();
    double shape_ratio = std::numeric_limits<double>::quiet_NaN();
    OrderPolytopeDiagonalCoolingDiagnostics<double> diagnostics;
};

Poset make_chain(unsigned int n)
{
    Poset::RV relations;
    for (unsigned int i = 0; i + 1 < n; ++i)
        relations.push_back({i, i + 1});
    return Poset(n, relations);
}

Poset make_funnel(unsigned int n)
{
    Poset::RV relations;
    for (unsigned int i = 1; i + 1 < n; ++i) {
        relations.push_back({0, i});
        relations.push_back({i, n - 1});
    }
    return Poset(n, relations);
}

Poset make_layered21()
{
    Poset::RV relations;
    unsigned int const width = 3, layers = 7;
    for (unsigned int layer = 0; layer + 1 < layers; ++layer) {
        for (unsigned int i = 0; i < width; ++i) {
            unsigned int const u = layer * width + i;
            relations.push_back({u, (layer + 1) * width + i});
            relations.push_back(
                {u, (layer + 1) * width + (i + 1) % width});
        }
    }
    return Poset(width * layers, relations);
}

void normalize_determinant(VT& shape)
{
    double log_geometric_mean = 0.0;
    for (int i = 0; i < shape.size(); ++i)
        log_geometric_mean += std::log(shape(i));
    log_geometric_mean /= static_cast<double>(shape.size());
    shape /= std::exp(log_geometric_mean);
}

MetricTruth chain_metric(unsigned int n)
{
    std::vector<double> center_values(n);
    VT shape(n);
    double const denominator =
        double(n + 1) * double(n + 1) * double(n + 2);
    for (unsigned int i = 0; i < n; ++i) {
        double const rank = double(i + 1);
        center_values[i] = rank / double(n + 1);
        shape(i) = rank * double(n + 1 - (i + 1)) / denominator;
    }
    normalize_determinant(shape);
    return {Point(n, center_values), shape, 0.0};
}

MetricTruth exact_rank_metric(Poset const& poset)
{
    using namespace order_polytope_volume_reduction_detail;
    typedef boost::multiprecision::uint128_t Count;
    unsigned int const n = poset.num_elem();
    if (n == 0 || n > 22)
        throw std::invalid_argument(
            "exact-rank benchmark metric requires 1 <= n <= 22");

    ReachabilityMatrix const reach = transitive_closure(poset);
    std::vector<unsigned long long> predecessor_mask(n, 0);
    for (unsigned int v = 0; v < n; ++v)
        for (unsigned int u = 0; u < n; ++u)
            if (reach[u][v]) predecessor_mask[v] |= 1ULL << u;

    std::size_t const state_count = std::size_t(1) << n;
    std::vector<Count> forward(state_count, 0);
    std::vector<Count> suffix(state_count, 0);
    forward[0] = 1;
    for (std::size_t mask = 0; mask < state_count; ++mask) {
        unsigned long long const mask64 =
            static_cast<unsigned long long>(mask);
        for (unsigned int v = 0; v < n; ++v) {
            unsigned long long const bit = 1ULL << v;
            if ((mask64 & bit) == 0 &&
                (predecessor_mask[v] & ~mask64) == 0)
                forward[mask | bit] += forward[mask];
        }
    }

    suffix[state_count - 1] = 1;
    for (std::size_t mask = state_count - 1; mask-- > 0;) {
        unsigned long long const mask64 =
            static_cast<unsigned long long>(mask);
        for (unsigned int v = 0; v < n; ++v) {
            unsigned long long const bit = 1ULL << v;
            if ((mask64 & bit) == 0 &&
                (predecessor_mask[v] & ~mask64) == 0)
                suffix[mask] += suffix[mask | bit];
        }
    }

    Count const extension_count = forward.back();
    if (extension_count == 0)
        throw std::runtime_error("benchmark poset has no linear extension");

    std::vector<std::vector<Count>> rank_counts(
        n, std::vector<Count>(n + 1, 0));
    for (std::size_t mask = 0; mask < state_count; ++mask) {
        Count const prefix_count = forward[mask];
        if (prefix_count == 0) continue;
        unsigned long long const mask64 =
            static_cast<unsigned long long>(mask);
        unsigned int const rank =
            static_cast<unsigned int>(__builtin_popcountll(mask64)) + 1;
        for (unsigned int v = 0; v < n; ++v) {
            unsigned long long const bit = 1ULL << v;
            if ((mask64 & bit) == 0 &&
                (predecessor_mask[v] & ~mask64) == 0)
                rank_counts[v][rank] +=
                    prefix_count * suffix[mask | bit];
        }
    }

    std::vector<double> center_values(n);
    VT shape(n);
    double const first_denominator = double(n + 1);
    double const second_denominator = double(n + 1) * double(n + 2);
    for (unsigned int v = 0; v < n; ++v) {
        long double first_rank_moment = 0.0L;
        long double second_rank_moment = 0.0L;
        for (unsigned int rank = 1; rank <= n; ++rank) {
            long double const probability =
                rank_counts[v][rank].convert_to<long double>() /
                extension_count.convert_to<long double>();
            first_rank_moment += probability * rank;
            second_rank_moment += probability * rank * (rank + 1);
        }
        double const mean =
            static_cast<double>(first_rank_moment / first_denominator);
        double const second =
            static_cast<double>(second_rank_moment / second_denominator);
        center_values[v] = mean;
        shape(v) = second - mean * mean;
        if (!std::isfinite(shape(v)) || shape(v) <= 0.0)
            throw std::runtime_error(
                "benchmark exact-rank metric produced invalid variance");
    }
    normalize_determinant(shape);
    return {Point(n, center_values), shape,
            std::log(extension_count.convert_to<double>())};
}

template <typename WalkPolicy>
Sample run_arm(BenchmarkCase const& benchmark_case,
               std::string const& arm,
               OrderPolytopeDiagonalRounding<Point> const& metric,
               unsigned int repetition,
               unsigned int seed,
               double error,
               unsigned int walk_length)
{
    Sample sample;
    sample.case_name = benchmark_case.name;
    sample.arm = arm;
    sample.repetition = repetition;
    sample.seed = seed;
    sample.dimension = benchmark_case.poset.num_elem();
    sample.relations = benchmark_case.poset.num_relations();
    sample.shape_ratio =
        metric.shape_diag().maxCoeff() / metric.shape_diag().minCoeff();
    OP polytope(benchmark_case.poset);
    RNG rng(sample.dimension);
    rng.set_seed(seed);

    auto const start = std::chrono::steady_clock::now();
    try {
        auto const result =
            volume_cooling_gaussians_order_polytope_diagonal_implementation<
                true, WalkPolicy>(
                polytope, rng, metric, error, walk_length,
                &sample.diagnostics);
        sample.log_extensions = result.log_volume +
                                std::lgamma(double(sample.dimension) + 1.0);
        sample.log_error = sample.log_extensions -
                           benchmark_case.truth.log_extensions;
        sample.success = std::isfinite(sample.log_extensions);
    } catch (std::exception const& error_message) {
        std::cerr << "FAIL case=" << sample.case_name
                  << " arm=" << arm
                  << " rep=" << repetition
                  << " error=" << error_message.what() << '\n';
    }
    sample.elapsed_microseconds =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    return sample;
}

double median(std::vector<double> values)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    std::size_t const middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

void print_raw_header()
{
    std::cout
        << "case,arm,repetition,seed,n,relations,success,log_extensions,"
           "log_error,elapsed_us,metric_setup_us,a0,phases,trajectories,"
           "us_per_trajectory,reflections,event_recomputations,"
           "rejected_reflection_limit,ambiguous_tie_failures,"
           "shared_contact_failures,max_bound_violation,"
           "max_cover_violation,shape_ratio\n";
}

void print_raw(Sample const& sample)
{
    auto const& diagnostic = sample.diagnostics;
    std::cout << sample.case_name << ',' << sample.arm << ','
              << sample.repetition << ',' << sample.seed << ','
              << sample.dimension << ',' << sample.relations << ','
              << (sample.success ? 1 : 0) << ','
              << sample.log_extensions << ',' << sample.log_error << ','
              << sample.elapsed_microseconds << ','
              << diagnostic.metric_setup_microseconds << ','
              << diagnostic.initial_a0 << ','
              << diagnostic.cooling_phase_count << ','
              << diagnostic.total_trajectories() << ','
              << diagnostic.microseconds_per_trajectory() << ','
              << diagnostic.reflection_count << ','
              << diagnostic.event_recomputation_count << ','
              << diagnostic.rejected_reflection_limit << ','
              << diagnostic.ambiguous_tie_failures << ','
              << diagnostic.shared_contact_failures << ','
              << diagnostic.max_observed_bound_violation << ','
              << diagnostic.max_observed_cover_violation << ','
              << sample.shape_ratio << '\n';
}

void print_summary(std::string const& case_name,
                   std::string const& arm,
                   std::vector<Sample> const& samples)
{
    std::vector<double> times, trajectories, errors;
    std::vector<double> initial_a0, phases, time_per_trajectory;
    unsigned int successes = 0, attempts = 0;
    for (Sample const& sample : samples) {
        if (sample.case_name != case_name || sample.arm != arm)
            continue;
        ++attempts;
        if (!sample.success) continue;
        ++successes;
        times.push_back(sample.elapsed_microseconds);
        trajectories.push_back(
            static_cast<double>(sample.diagnostics.total_trajectories()));
        initial_a0.push_back(sample.diagnostics.initial_a0);
        phases.push_back(sample.diagnostics.cooling_phase_count);
        time_per_trajectory.push_back(
            sample.diagnostics.microseconds_per_trajectory());
        errors.push_back(sample.log_error);
    }
    double squared_error_sum = std::inner_product(
        errors.begin(), errors.end(), errors.begin(), 0.0);
    double const rms = errors.empty()
                           ? std::numeric_limits<double>::quiet_NaN()
                           : std::sqrt(squared_error_sum / errors.size());
    std::cerr << std::setprecision(17)
              << "SUMMARY case=" << case_name
              << " arm=" << arm
              << " success=" << successes << '/' << attempts
              << " median_total_us=" << median(times)
              << " median_a0=" << median(initial_a0)
              << " median_phases=" << median(phases)
              << " median_trajectories=" << median(trajectories)
              << " median_us_per_trajectory="
              << median(time_per_trajectory)
              << " rms_log_error=" << rms << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    unsigned int const repetitions =
        argc > 1 ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10))
                 : 3u;
    double const error = argc > 2 ? std::strtod(argv[2], nullptr) : 0.3;
    unsigned int const walk_length =
        argc > 3 ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10))
                 : 1u;
    if (repetitions == 0 || !std::isfinite(error) || error <= 0.0 ||
        walk_length == 0) {
        std::cerr << "usage: " << argv[0]
                  << " [positive repetitions] [positive error] "
                     "[positive walk_length]\n";
        return 2;
    }

    std::vector<BenchmarkCase> cases;
    Poset chain = make_chain(32);
    cases.push_back({"chain32", chain, chain_metric(32)});
    Poset funnel = make_funnel(18);
    cases.push_back({"funnel18", funnel, exact_rank_metric(funnel)});
    Poset layered = make_layered21();
    cases.push_back({"layered21", layered, exact_rank_metric(layered)});

    std::cerr << "BENCHMARK_CONFIG repetitions=" << repetitions
              << " error=" << error
              << " walk_length=" << walk_length
              << " user_metric_precompute_timed=0"
              << " body_revalidation_in_total=1"
              << " same_center=1 normalized_D1=1\n";
    std::cout << std::setprecision(17);
    print_raw_header();

    std::vector<Sample> samples;
    for (BenchmarkCase const& benchmark_case : cases) {
        ReducedOrderPolytopeVolumeProblem const default_reduction =
            reduce_order_polytope_volume_problem(benchmark_case.poset);
        std::cerr << "CASE name=" << benchmark_case.name
                  << " default_reduction_exact="
                  << (default_reduction.is_exact() ? 1 : 0)
                  << " default_residual_count="
                  << default_reduction.residual_posets.size();
        if (!default_reduction.residual_posets.empty())
            std::cerr << " first_residual_n="
                      << default_reduction.residual_posets.front().num_elem();
        std::cerr << '\n';
        OP polytope(benchmark_case.poset);
        VT identity = VT::Ones(benchmark_case.poset.num_elem());
        OrderPolytopeDiagonalRounding<Point> identity_metric(
            polytope, benchmark_case.truth.center, identity);
        OrderPolytopeDiagonalRounding<Point> rounded_metric(
            polytope, benchmark_case.truth.center,
            benchmark_case.truth.shape);

        for (unsigned int repetition = 0;
             repetition < repetitions; ++repetition) {
            unsigned int const seed = 20260818u + repetition;
            samples.push_back(run_arm<BenchmarkSphericalPolicy>(
                benchmark_case, "S", identity_metric, repetition, seed,
                error, walk_length));
            samples.push_back(run_arm<
                OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk>(
                benchmark_case, "D0", identity_metric, repetition, seed,
                error, walk_length));
            samples.push_back(run_arm<
                OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk>(
                benchmark_case, "D1", rounded_metric, repetition, seed,
                error, walk_length));
        }
    }

    for (Sample const& sample : samples) print_raw(sample);
    for (BenchmarkCase const& benchmark_case : cases)
        for (std::string const arm : {std::string("S"), std::string("D0"),
                                      std::string("D1")})
            print_summary(benchmark_case.name, arm, samples);

    return std::all_of(samples.begin(), samples.end(),
                       [](Sample const& sample) { return sample.success; })
               ? 0 : 1;
}
