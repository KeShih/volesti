<!--
Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.
-->

# OrderPolytope diagonal-rounding F3 benchmark manifest

This run isolates the diagonal backend cost from the effect of a supplied
diagonal shape. It is not an end-to-end metric-learning benchmark.

## Source and environment

- Base commit: `ce5905c8bf793c3478b28ff5ac299b9c1c8bd1ed`
- Branch: `feature/order-polytope-diagonal-rounding`
- Build type: `Release`
- Compiler: Apple clang 17.0.0 (`clang-1700.3.9.908`)
- Host: Darwin arm64, kernel 27.0.0
- HMC header SHA-256: `833080296432c3282b80ab51b5c0ee1308e568248d621b6dc14237e7c430d7da`
- Benchmark source SHA-256: `2e691d11f29ca0f0d219b93d4a302a74c226784ddba588a9695684a6495d8c85`
- Benchmark binary SHA-256: `b46ca6a7deffeaf3a216ce03bc7566e761f564b70efa614e41b8ad2a8eea5c77`
- Raw CSV SHA-256: `6d58498640c5c2ca5d0fdfee18608d3713b7c451d4d4ce911d10a1a272f1b60c`

The base commit identifies the synchronized starting point. The hashes above
pin the uncommitted F1--F3 worktree used for this run.

Exact commands:

```bash
cmake -S test -B test/build
cmake --build test/build \
  --target order_polytope_diagonal_rounding_benchmark --parallel 8
/usr/bin/time -p \
  ./test/build/order_polytope_diagonal_rounding_benchmark 3 0.3 1 \
  > /tmp/order_polytope_diagonal_rounding_f3_raw.csv \
  2> /tmp/order_polytope_diagonal_rounding_f3_summary.txt
```

The run completed successfully in 21.37 seconds. The three fixed seeds were
20260818, 20260819, and 20260820. User metric precomputation was excluded;
mandatory body binding/revalidation remained in total wall time.

## Arms

- `S`: spherical event/reflection dynamics, supplied center, identity target
  shape.
- `D0`: diagonal dynamics with the same center and `D=I`.
- `D1`: diagonal dynamics with the same center and a precomputed shape whose
  geometric mean is one.

`chain32` and `funnel18` are direct-body stress cases that the default LINEXT
reducer solves exactly. `layered21` is not removed by the current reducer: it
produces one residual of dimension 21, so it represents the actual MCMC path.

## Median results

| Case | Arm | Total us | a0 | Phases | Trajectories | us/trajectory | RMS log error |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain32 | S | 558971.2 | 10634.18 | 20 | 254798 | 2.194 | 0.0572 |
| chain32 | D0 | 2248299.9 | 10634.18 | 20 | 254789 | 8.824 | 0.0519 |
| chain32 | D1 | 2208463.9 | 14620.18 | 22 | 287127 | 7.692 | 0.1257 |
| funnel18 | S | 101152.7 | 895.30 | 22 | 146978 | 0.675 | 0.0982 |
| funnel18 | D0 | 564838.5 | 895.30 | 22 | 146978 | 3.768 | 0.0993 |
| funnel18 | D1 | 176618.7 | 49.74 | 8 | 43736 | 4.055 | 0.0954 |
| layered21 | S | 101842.5 | 567.61 | 10 | 76583 | 1.361 | 0.0624 |
| layered21 | D0 | 456588.1 | 567.61 | 10 | 71914 | 6.410 | 0.0440 |
| layered21 | D1 | 484186.5 | 652.88 | 11 | 81897 | 5.868 | 0.0888 |

Fixed backend overhead (`D0/S`) was 4.02x on chain32, 5.58x on funnel18,
and 4.48x on layered21. The isolated rounding ratio (`D0/D1`) was 1.02x,
3.20x, and 0.94x respectively. Thus the supplied diagonal shape gives a
clear trajectory-count win on funnel18, a small win on chain32, and no total
runtime win on the actual dimension-21 residual. This run supports
case-specific gains, not a universal speedup claim.

No reflection-limit rejection, ambiguous-tie failure, or shared-contact
failure was observed in any of the 27 runs. The exact diagonal release gate
reported no bound or cover violation in the 18 `D0`/`D1` runs. The `S` arm
retains production spherical semantics, so its zero violation fields are not
interpreted as an exact-feasibility measurement.

Raw data: [order_polytope_diagonal_rounding_f3_raw.csv](order_polytope_diagonal_rounding_f3_raw.csv)
