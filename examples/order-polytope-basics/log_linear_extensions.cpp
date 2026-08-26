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
// The estimator uses the purpose-built spherical event-queue HMC walk.

#include <cmath>
#include <cstdio>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "generators/boost_random_number_generator.hpp"
#include "misc/poset.h"
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
    // demonstrates the sampled path.
    Poset::RV relations{{0, 1}, {0, 3}, {2, 3}};
    Poset poset(4, relations);

    LinearExtensionOptions options;
    options.error = 0.2;
    options.repetitions = 3;
    options.reduction.exact_dp_max_n = 0;

    RNGType rng(poset.num_elem());
    auto const r = estimate_log_linear_extensions<RNGType, Point>(
        poset, rng, options);

    std::printf("log vol(O(P))   = %.6f  (se %.4f over %u repetitions)\n",
                r.log_volume, r.se_log_volume, r.repetitions);
    std::printf("log e(P)        = %.6f  (exact: log 5 = %.6f)\n",
                r.log_extensions, std::log(5.0));
    std::printf("e(P) estimate   = %.3f\n", std::exp(r.log_extensions));
    return 0;
}
