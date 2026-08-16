// VolEsti (volume computation and sampling library)

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

// Counting linear extensions of a poset P through the order-polytope
// volume identity
//
//     vol(O(P)) = e(P) / n!    <=>    log e(P) = log vol(O(P)) + lgamma(n+1),
//
// with Gaussian-cooling volume estimation whose stage ratios are
// accumulated in the log domain (logsumexp of the log weights, never a
// mean of logs), so the result stays finite even when the raw volume or
// the raw count would under-/overflow a double.
//
// The estimator runs on a selectable Gaussian HMC backend; the
// purpose-built spherical event-queue walk is the default for order
// polytopes, and the classic generic CDHR sampler stays available via
// options.use_generic_cdhr.

#include <cmath>
#include <cstdio>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "misc/poset.h"
#include "convex_bodies/orderpolytope.h"
#include "random_walks/random_walks.hpp"
#include "volume/linear_extensions_volume.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef BoostRandomNumberGenerator<boost::mt19937, NT, 42> RNGType;

int main() {
    // The "N" poset on 4 elements (0 < 1, 0 < 3, 2 < 3) has exactly
    // e(P) = 5 linear extensions.  It is not series-parallel, but tiny
    // posets are normally solved exactly by the small-poset DP reduction
    // before any sampling happens - disable it here so the example
    // demonstrates the sampled path and its cooling-stage diagnostics.
    Poset::RV relations{{0, 1}, {0, 3}, {2, 3}};
    Poset poset(4, relations);

    LinearExtensionOptions options;
    options.backend = GaussianHMCBackend::Auto;   // spherical HMC walk
    options.error = 0.2;
    options.repetitions = 3;
    options.count_violations = true;
    options.reduction.enable_small_exact_dp = false;

    RNGType rng(poset.num_elem());
    auto const r = estimate_log_linear_extensions<RNGType, Point>(
        poset, rng, options);

    std::printf("log vol(O(P))   = %.6f  (se %.4f over %u repetitions)\n",
                r.log_volume, r.se_log_volume, r.repetitions);
    std::printf("log e(P)        = %.6f  (exact: log 5 = %.6f)\n",
                r.log_extensions, std::log(5.0));
    std::printf("e(P) estimate   = %.3f\n", std::exp(r.log_extensions));
    std::printf("samples         = %llu, containment violations = %llu\n",
                r.total_samples, r.violations);
    std::printf("cooling stages of the last run:\n");
    for (auto const& s : r.stages)
        std::printf("  a: %8.4f -> %8.4f   log-ratio %9.5f   "
                    "samples %6llu   weight CV^2 %.4f\n",
                    s.a_curr, s.a_next, s.log_ratio, s.samples, s.cv2);
    return 0;
}
