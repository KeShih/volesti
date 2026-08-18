<!--
Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.
-->

# User-provided diagonal rounding for order polytopes

VolEsti provides an explicit opt-in Gaussian-cooling path for an
`OrderPolytope` when the caller already has a strictly feasible center and a
positive diagonal rounding shape. The ordinary spherical cooling and exact-HMC
policy remain the default.

## Convention

For a center $c$ and $D=\operatorname{diag}(d_1,\ldots,d_n)$ with $d_i>0$,
the phase-$a$ target is

$$
  \exp\left(-a(x-c)^T D^{-1}(x-c)\right)1[x\in P].
$$

Here $D$ is a **shape**, not the phase covariance. The untruncated covariance
is $D/(2a)$. The matched constant HMC mass is $M=D^{-1}$, so refreshed
velocities have covariance $D$ and the free dynamics retain the single
frequency $\omega=\sqrt{2a}$.

The whole-space log normalizer is

$$
  \frac{n}{2}\log(\pi/a)+\frac{1}{2}\log\det D.
$$

The determinant term is included once by the rounded cooling implementation.
There is no final Jacobian because samples and volumes remain in the original
$x$ coordinates; no dense affine transformation is formed.

## API

Construct a validated metric from the same poset that will be counted:

```cpp
#include "cartesian_geom/cartesian_kernel.h"
#include "generators/boost_random_number_generator.hpp"
#include "volume/linear_extensions_volume.hpp"

using Point = Cartesian<double>::Point;
using Vector = Eigen::VectorXd;
using RNG = BoostRandomNumberGenerator<boost::mt19937, double>;

Poset poset = /* ... */;
Point center = /* strictly feasible for every true facet */;
Vector shape_diag = /* finite positive d_i values */;

OrderPolytopeDiagonalRounding<Point> metric(
    poset, center, shape_diag);
RNG rng(poset.num_elem());

OrderPolytopeDiagonalRoundingDiagnostics<double> diagnostics;
auto estimate = estimate_log_linear_extensions_diagonal_order_hmc<RNG, Point>(
    poset, rng, metric, 0.1, 1,
    OrderPolytopeVolumeReductionOptions(), &diagnostics);
```

For direct residual-volume work, use
`volume_cooling_gaussians_order_polytope_diagonal_result` from
`volume/order_polytope_diagonal_cooling.hpp`. Both entry points revalidate the
center and shape against the actual body before sampling. Invalid dimensions,
non-finite or nonpositive shape entries, non-strict centers, and unusable
derived arithmetic fail with an exception; they are not clamped or
regularized.

The diagonal policy is
`OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk`. It is intended
for the internal translated `DiagonalRoundingOrderPolytope`; use the cooling
entry points unless testing the walk itself.

## Geometry and diagnostics

Initialization uses true-facet distances in the rounded metric:

- lower wall: $c_i/\sqrt{d_i}$;
- upper wall: $(1-c_i)/\sqrt{d_i}$;
- cover $x_u\leq x_v$: $(c_v-c_u)/\sqrt{d_u+d_v}$.

The EventQueue, one-frequency root oracle, simultaneous-contact batching, and
incident-only invalidation are shared with the spherical backend. A cover
reflection uses the mass metric; bound reflections remain coordinate sign
flips. Shared-coordinate simultaneous contacts retain the existing
fail-closed behavior.

Diagnostics are optional. Per residual they record metric setup time, initial
$a_0$, cooling phase count, schedule and ratio-loop trajectory counts,
reflections, event recomputations, reflection-limit rejections, ambiguous-tie
failures, shared-contact failures, the largest observed bound and cover
violations, total cooling time, and microseconds per trajectory. The
counting-level diagnostics also report total estimator time and aggregate the
residual counters. Supplying no diagnostics selects a compile-time
uninstrumented diagonal walk, so event-loop counters are absent.

Trajectory failures have explicit semantics. Reaching the configured
reflection limit rolls the position and contact state back and records a
rejected transition. A numerically ambiguous event tie and a simultaneous
contact whose facets share a coordinate both roll back and throw; neither is
returned as a successful self-loop. Internal root classification may use
floating-point tolerances, but the final sample-release gate evaluates the
stored coordinates with exact floating-point expansion signs. Any
representable bound or cover violation therefore fails before the point is
handed to the cooling estimator.

## Isolated benchmark

The optional CMake target
`order_polytope_diagonal_rounding_benchmark` separates the backend cost from
the coordinate-scaling benefit:

- `S`: spherical event/reflection dynamics with the same supplied center and
  an identity target shape;
- `D0`: diagonal dynamics with $D=I$;
- `D1`: diagonal dynamics with a precomputed $D$ normalized so that
  $(\prod_i d_i)^{1/n}=1$.

Build and run it explicitly; it is excluded from the default test build:

```bash
cmake --build test/build \
  --target order_polytope_diagonal_rounding_benchmark -j8
./test/build/order_polytope_diagonal_rounding_benchmark 3 0.3 1 \
  > diagonal_rounding_raw.csv 2> diagonal_rounding_summary.txt
```

The CSV records $a_0$, phase and trajectory counts, time per trajectory,
total time, error, reflections, recomputations, failure counters, and observed
violations for every seed. Construction of the caller-provided metric is
deliberately outside the timed region; the mandatory binding/revalidation
against the actual body remains in total wall time and is also reported
separately. Compare `S/D0` for fixed backend overhead and `D0/D1` for the
benefit of the supplied relative coordinate scales; do not attribute the
combined `S/D1` ratio solely to rounding.

## Limitations

This path does not estimate a center or shape, normalize the caller's shape,
or implement dense, block, or phase-adaptive rounding. The absolute scale of
$D$ affects the physical mixing time while the production trajectory-time law
is deliberately unchanged. Callers should therefore supply a reasonably
scaled shape and use the diagnostics in a bounded benchmark before drawing a
performance conclusion. Extreme but arithmetically representable scales may
mix poorly; inputs that make required arithmetic non-finite fail explicitly.

The cooling result remains a stochastic estimate. A lower condition number,
lower time per trajectory, or a single fixed-seed result is not by itself a
speed or accuracy claim.
