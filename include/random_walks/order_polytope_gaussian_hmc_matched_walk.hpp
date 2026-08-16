// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_HMC_MATCHED_WALK_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_HMC_MATCHED_WALK_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigen>

#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "random_walks/order_polytope_hmc_modal.hpp"

// Exact HMC for a *general* Gaussian target on an order polytope,
//
//     pi(x) ~ exp(-1/2 (x-c)^T A (x-c)) 1_{x in O(P)},   A SPD,
//
// using the matched mass policy M = A / omega^2.  Then M^{-1} A =
// omega^2 I, so every coordinate oscillates at the common frequency
// omega and the trajectory keeps the closed form
//
//     x(t) = c + (x0 - c) cos(omega t) + v0/omega sin(omega t),
//
// which preserves the whole angle-space machinery of the spherical
// walk: with alpha = x - c and beta = v/omega each facet value stays
// one-frequency, A_f cos(theta) + B_f sin(theta) = C_f with
// C_f = b_f - n_f.c, and the hardened outgoing-root solver applies
// verbatim.  What changes is the reflection: against the mass metric,
//
//     v+ = v- - 2 (n^T v-) / (n^T A^{-1} n) A^{-1} n
//
// (omega^2 cancels between M^{-1} and the quotient), and the momentum
// refresh, v ~ N(0, M^{-1}) = N(0, omega^2 A^{-1}), sampled as
// v = omega L^{-T} xi with A = L L^T and xi ~ N(0, I).
//
// Because A^{-1} n is generally dense, one reflection changes *every*
// velocity coordinate, so the spherical walk's incident-only event
// invalidation is unsound here.  The MatchedDenseAngleFullScan backend
// therefore rebuilds the picture after each reflection: it re-solves
// all true facets and takes the earliest outgoing crossing (no heap),
// still in trig-free angle space with quarter-period chunks.
//
// ---- Backend selection guide -----------------------------------------
//
// Fast paths, from most to least specialized (Auto picks the first
// applicable one):
//
//   Identity mass, spherical target      -> SphericalEventQueue
//       the tuned incident-only event queue (exact: reflections touch
//       one or two coordinates).
//   MatchedPrecision, diagonal A         -> MatchedDiagonalEventQueue
//       weighted reflections, still incident-only and exact because
//       A^{-1} n is supported on the hit coordinates.
//   MatchedPrecision, dense A            -> MatchedDenseAngleFullScan
//       common frequency retained (M = A/omega^2), but every reflection
//       moves all velocities, so all facets re-solve per reflection.
//   UserSpecified mass M (arbitrary SPD) -> ModalDenseFallback
//       correctness-first: generalized eigenmodes, multi-frequency facet
//       values, safeguarded full-scan root finding.  A user-specified M
//       is NEVER silently replaced by the matched mass and never routed
//       onto a fast path: incident-only invalidation and one-frequency
//       angle-space solving are both invalid for arbitrary dense A, M.
//
// When the caller supplies A but no mass, MatchedPrecision (M =
// A/omega^2) is the intended default: it keeps the closed-form
// one-frequency trajectory at the cost of a mass choice that is usually
// what one wants anyway (unit-frequency preconditioning).
//
// For linear-extension counting on top of these walks, note
// log_count = log_volume + lgamma(n + 1); see
// volume/linear_extensions_volume.hpp.
//
// Minimal example (arbitrary SPD target and mass):
//
//     typedef OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk
//         WalkPolicy;
//     WalkPolicy::parameters<NT> params(A, M, center);   // UserSpecified
//     WalkPolicy::Walk<OrderPolytope<Point>, RNG> walk(P, p, params, rng);
//     walk.apply(P, p, walk_length, rng);                // p ~ N(c, A^{-1}) on P

enum class GaussianMassPolicy {
    Identity,          // M = I: the standard spherical target exp(-a|x-c|^2)
    MatchedPrecision,  // M = A / omega^2 for a general SPD precision A
    UserSpecified      // arbitrary SPD mass M chosen by the caller
};

enum class GaussianHMCBackend {
    Auto,                       // Identity -> SphericalEventQueue,
                                // MatchedPrecision -> MatchedDiagonalEventQueue
                                //   for (detected) diagonal A,
                                //   MatchedDenseAngleFullScan otherwise
    SphericalEventQueue,        // incident-only event queue (spherical only)
    MatchedDenseAngleFullScan,  // full facet re-scan after each reflection
    MatchedDiagonalEventQueue,  // incident-only event queue with weighted
                                // reflections (diagonal A only: A^{-1} n is
                                // supported on the hit coordinates, so local
                                // invalidation stays exact)
    ModalDenseFallback          // correctness-first multi-frequency modal
                                // trajectory with a full-scan safeguarded
                                // root oracle: the only valid backend for
                                // arbitrary SPD A and user-specified M
};

struct OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk
{
    template <typename NT>
    struct parameters
    {
        typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
        typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;

        parameters(NT _omega)
            : omega(_omega), policy(GaussianMassPolicy::Identity),
              backend(GaussianHMCBackend::Auto),
              m_L(0), set_L(false), rho(0), set_rho(false)
        {}

        parameters(MT const& _A, VT const& _center, NT _omega,
                   GaussianHMCBackend _backend = GaussianHMCBackend::Auto)
            : A(_A), center(_center), omega(_omega),
              policy(GaussianMassPolicy::MatchedPrecision), backend(_backend),
              m_L(0), set_L(false), rho(0), set_rho(false)
        {}

        // Diagonal precision supplied directly as its diagonal: routed to
        // the incident-only event queue under Auto without any detection.
        parameters(VT const& _A_diag, VT const& _center, NT _omega,
                   GaussianHMCBackend _backend = GaussianHMCBackend::Auto)
            : A_diag(_A_diag), center(_center), omega(_omega),
              policy(GaussianMassPolicy::MatchedPrecision), backend(_backend),
              m_L(0), set_L(false), rho(0), set_rho(false)
        {}

        // Arbitrary user-specified SPD mass M: routed to the modal
        // fallback under Auto (never silently replaced by matched mass).
        parameters(MT const& _A, MT const& _M, VT const& _center,
                   GaussianHMCBackend _backend = GaussianHMCBackend::Auto)
            : A(_A), M_mass(_M), center(_center), omega(NT(1)),
              policy(GaussianMassPolicy::UserSpecified), backend(_backend),
              m_L(0), set_L(false), rho(0), set_rho(false)
        {}

        MT A;                        // target precision; ignored for Identity
        VT A_diag;                   // diagonal precision (nonempty wins over A)
        MT M_mass;                   // momentum mass (UserSpecified only)
        VT center;                   // Gaussian center c; empty means origin
        NT omega;                    // common trajectory frequency
        GaussianMassPolicy policy;
        GaussianHMCBackend backend;
        NT m_L;
        bool set_L;
        unsigned int rho;
        bool set_rho;
    };

    // Conservative diagonal detector for dense inputs: every off-diagonal
    // entry must be negligible against the geometric mean of its diagonal
    // pair (1e-15 relative, i.e. genuinely zero up to representation
    // noise).  Nearly-dense matrices are NOT classified as diagonal; any
    // malformed input (nonsquare, NaN) is routed to the dense branch,
    // whose validation rejects it loudly.  The bound multiplies two
    // square roots instead of taking one root of the product so it never
    // overflows to inf (which would wave every off-diagonal through).
    template <typename MT, typename NT>
    static bool is_diagonal_matrix(MT const& A)
    {
        if (A.rows() != A.cols()) return false;
        for (Eigen::Index i = 0; i < A.rows(); ++i)
            for (Eigen::Index j = 0; j < A.cols(); ++j) {
                if (i == j) continue;
                NT const bound = NT(1e-15) * std::sqrt(std::abs(A(i, i)))
                                           * std::sqrt(std::abs(A(j, j)));
                if (!(std::abs(A(i, j)) <= bound)) return false;
            }
        return true;
    }

    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;
        typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk SphericalPolicy;
        typedef typename SphericalPolicy::template Walk<Polytope, RandomNumberGenerator>
            SphericalWalk;

        Walk(Polytope &P, Point const& p, parameters<NT> const& params,
             RandomNumberGenerator &rng)
        {
            unsigned int const n = P.dimension();
            _omega = params.omega;
            if (!(_omega > NT(0)))
                throw std::invalid_argument("matched HMC: omega must be positive");
            _inv_omega = NT(1) / _omega;
            _angle_eps = _omega * time_eps();
            _center = params.center.size() == 0
                    ? VT(VT::Zero(n)) : VT(params.center);
            if (_center.size() != static_cast<Eigen::Index>(n))
                throw std::invalid_argument("matched HMC: center dimension mismatch");

            _backend = resolve_backend(params);

            _Len = params.set_L ? params.m_L : default_trajectory_length(P);
            _rho = params.set_rho ? params.rho : 100 * n;

            if (_backend == GaussianHMCBackend::SphericalEventQueue) {
                // Identity mass: the target is exp(-a |x - c|^2) with
                // a = omega^2 / 2 and the tuned spherical event-queue walk
                // is exact for it (its facet offsets encode any shift of P,
                // and center != 0 is not supported by that walk).
                if (_center.squaredNorm() != NT(0))
                    throw std::invalid_argument(
                        "matched HMC: SphericalEventQueue supports center = 0 only");
                typename SphericalPolicy::parameters sp(
                    double(_Len), true, _rho, true);
                _spherical.reset(new SphericalWalk(
                    P, p, _omega * _omega / NT(2), rng, sp));
                _A = MT();   // energy helpers treat empty A as omega^2 I
                return;
            }

            if (_backend == GaussianHMCBackend::ModalDenseFallback) {
                // Arbitrary A with an arbitrary (or matched/identity-
                // derived) SPD mass: multi-frequency dynamics, so no
                // quarter-period chunking and no incident-only shortcuts;
                // every leg runs on the modal kernel plus the full-scan
                // safeguarded root oracle.  Correctness beats speed here.
                MT A_eff, M_eff;
                switch (params.policy) {
                    case GaussianMassPolicy::UserSpecified:
                        A_eff = params.A;
                        M_eff = params.M_mass;
                        break;
                    case GaussianMassPolicy::MatchedPrecision:
                        A_eff = params.A_diag.size() != 0
                              ? MT(params.A_diag.asDiagonal()) : params.A;
                        M_eff = MT(A_eff / (_omega * _omega));
                        break;
                    default:   // Identity
                        A_eff = MT(_omega * _omega * MT::Identity(n, n));
                        M_eff = MT(MT::Identity(n, n));
                        break;
                }
                _A = A_eff;
                _M = M_eff;
                // The kernel validates symmetry/SPD of both matrices.
                _modal.reset(new ModalTraj(P, A_eff, M_eff, _center));
                _lltM.compute(_M);
                if (_lltM.info() != Eigen::Success)
                    throw std::invalid_argument("matched HMC: M is not SPD");
                _hard_negative_slack = 0;
                _modal_last_hit = -1;
                build_event_facets(P);
                build_reflection_data(_lltM);

                // Multi-frequency dynamics have no quarter-period, so the
                // default leg length is only diameter-capped.
                NT const diameter =
                    compute_diameter<Polytope>::template compute<NT>(P);
                if (!params.set_L) _Len = std::min(diameter, NT(1));

                _p = p;
                refresh_velocity(rng);
                NT const T = rng.sample_urdist() * _Len;
                advance_modal(T);
                return;
            }

            if (_backend == GaussianHMCBackend::MatchedDiagonalEventQueue) {
                VT const diag = params.A_diag.size() != 0
                              ? VT(params.A_diag) : VT(params.A.diagonal());
                if (diag.size() != static_cast<Eigen::Index>(n))
                    throw std::invalid_argument(
                        "matched HMC: A dimension mismatch");
                if (!(diag.minCoeff() > NT(0)))
                    throw std::invalid_argument("matched HMC: A is not SPD");

                // Diagnostic state (energy helpers, reflect_velocity,
                // sample_velocity) is shared with the dense backend; the
                // actual sampling delegates to the weighted event-queue
                // walk, whose incident-only invalidation is exact because
                // A^{-1} n touches only the hit coordinates.
                _A = MT(diag.asDiagonal());
                _llt.compute(_A);
                if (_llt.info() != Eigen::Success)
                    throw std::invalid_argument("matched HMC: A is not SPD");
                _last_hit_fid = no_facet();
                _hard_negative_slack = 0;
                build_event_facets(P);
                build_reflection_data();

                typename SphericalPolicy::parameters sp(
                    double(_Len), true, _rho, true);
                _spherical.reset(new SphericalWalk(
                    P, p, rng, diag, _center, _omega, sp));
                return;
            }

            // MatchedDenseAngleFullScan
            _A = params.policy == GaussianMassPolicy::Identity
               ? MT(_omega * _omega * MT::Identity(n, n))
               : params.A_diag.size() != 0
               ? MT(params.A_diag.asDiagonal()) : MT(params.A);
            if (_A.rows() != static_cast<Eigen::Index>(n) || _A.cols() != _A.rows())
                throw std::invalid_argument("matched HMC: A dimension mismatch");
            // Eigen's LLT reads only the lower triangle, so it would
            // silently factor the lower-triangle symmetrization of a
            // non-symmetric A while quad_form used A itself: reject the
            // mismatch instead of sampling a different target.
            if (!_A.isApprox(_A.transpose()))
                throw std::invalid_argument("matched HMC: A must be symmetric");
            _llt.compute(_A);
            if (_llt.info() != Eigen::Success)
                throw std::invalid_argument("matched HMC: A is not SPD");

            _last_hit_fid = no_facet();
            _hard_negative_slack = 0;
            build_event_facets(P);
            build_reflection_data();

            _p = p;
            refresh_velocity(rng);
            NT const T = rng.sample_urdist() * _Len;
            advance_full_scan(T);
        }

        GaussianHMCBackend backend() const { return _backend; }

        inline void apply(Polytope const& P, Point& p,
                          unsigned int const& walk_length, RandomNumberGenerator &rng)
        {
            if (_spherical) {
                _spherical->apply(P, p, NT(0), walk_length, rng);
                return;
            }
            for (auto j = 0u; j < walk_length; ++j) {
                NT const T = rng.sample_urdist() * _Len;
                Point const v0 = _v;
                refresh_velocity(rng);
                // Fresh momentum: clear the residual-root guard (see the
                // spherical walk for the outward-refresh leak this stops).
                _last_hit_fid = no_facet();
                _modal_last_hit = -1;
                Point p0 = _p;
                if (!(_modal ? advance_modal(T) : advance_full_scan(T))) {
                    // Restore both halves of the state: a lone position
                    // restore would leave (p, v) off any single trajectory.
                    _p = p0;
                    _v = v0;
                    _last_hit_fid = no_facet();
                }
            }
            p = _p;
        }

        // Deterministically advance one leg from (p, v) for time T without
        // touching the RNG; on success p and v hold the end state.  With
        // the SphericalEventQueue backend v is updated as well (the
        // delegate exposes its end velocity).  Returns false when the leg
        // aborts on the reflection budget; p and v are then unchanged.
        inline bool apply_leg(Polytope const& P, Point& p, Point& v, NT const& T)
        {
            if (_spherical) {
                if (!_spherical->apply_leg(P, p, v, T)) return false;
                v = _spherical->velocity();
                return true;
            }
            _p = p;
            _v = v;
            _last_hit_fid = no_facet();
            _modal_last_hit = -1;
            if (!(_modal ? advance_modal(T) : advance_full_scan(T))) {
                _p = p;
                _v = v;
                _last_hit_fid = no_facet();
                return false;
            }
            p = _p;
            v = _v;
            return true;
        }

        // Draw one momentum-refresh velocity, v ~ N(0, omega^2 A^{-1}),
        // without advancing the walk: the diagnostic entry point for
        // covariance tests.  Matched (dense or diagonal) backends only.
        inline Point sample_velocity(RandomNumberGenerator &rng) const
        {
            if (_A.size() == 0)
                throw std::logic_error(
                    "matched HMC: sample_velocity needs a matched backend");
            unsigned int const n = static_cast<unsigned int>(_center.size());
            Point xi = GetDirection<Point>::apply(n, rng, false);
            Point out(n);
            if (_modal) {
                // v = M^{-1} p with p = L_M xi ~ N(0, M), i.e.
                // v = L_M^{-T} xi ~ N(0, M^{-1}).
                out.set_coeffs(VT(_lltM.matrixU().solve(xi.getCoefficients())));
                return out;
            }
            // L^{-T} xi has covariance L^{-T} L^{-1} = A^{-1}; solving with
            // L instead would give (L^T L)^{-1} != A^{-1} — the covariance
            // test pins this factor orientation.
            VT v = _llt.matrixU().solve(xi.getCoefficients());
            v *= _omega;
            out.set_coeffs(v);
            return out;
        }

        // Matched reflection of a velocity vector on event facet fid,
        // independent of the trajectory state: the diagnostic entry point
        // for invariant tests (n^T v flips sign, v^T A v is preserved).
        // Matched (dense or diagonal) backends only.
        inline void reflect_velocity(unsigned int fid, VT& v) const
        {
            if (_A.size() == 0)
                throw std::logic_error(
                    "matched HMC: reflect_velocity needs a matched backend");
            NT const nv = normal_dot(fid, v);
            v.noalias() -= (NT(2) * nv / _q(fid)) * _w.col(fid);
        }

        // Energy in the matched Hamiltonian: H = U + K is invariant along
        // legs (free flight is exact, reflections preserve v^T A v).
        inline NT potential(Point const& x) const
        {
            VT const d = x.getCoefficients() - _center;
            return NT(0.5) * quad_form(d);
        }
        inline NT kinetic(Point const& v) const
        {
            VT const vc = v.getCoefficients();
            if (_M.size() != 0)
                return NT(0.5) * vc.dot(_M * vc);
            return NT(0.5) * _inv_omega * _inv_omega * quad_form(vc);
        }
        inline NT hamiltonian(Point const& x, Point const& v) const
        {
            return potential(x) + kinetic(v);
        }

        inline unsigned long long hard_negative_slack_count() const
        {
            return _spherical ? _spherical->hard_negative_slack_count()
                              : _hard_negative_slack;
        }

        // Number of true facets driving the oracle.
        // Matched (dense or diagonal) backends only.
        inline unsigned int num_event_facets() const
        {
            if (_A.size() == 0)
                throw std::logic_error(
                    "matched HMC: num_event_facets needs a matched backend");
            return static_cast<unsigned int>(_f_i0.size());
        }

        inline void update_delta(NT L)
        {
            _Len = L;
            if (_spherical) _spherical->update_delta(L);
        }

    private:

        static GaussianHMCBackend resolve_backend(parameters<NT> const& params)
        {
            bool const diag =
                params.A_diag.size() != 0 ||
                (params.A.size() != 0 &&
                 OrderPolytopeGaussianMatchedHamiltonianMonteCarloExactWalk::
                     template is_diagonal_matrix<MT, NT>(params.A));
            if (params.backend == GaussianHMCBackend::Auto) {
                if (params.policy == GaussianMassPolicy::UserSpecified)
                    return GaussianHMCBackend::ModalDenseFallback;
                if (params.policy == GaussianMassPolicy::Identity)
                    return GaussianHMCBackend::SphericalEventQueue;
                return diag ? GaussianHMCBackend::MatchedDiagonalEventQueue
                            : GaussianHMCBackend::MatchedDenseAngleFullScan;
            }
            if (params.policy == GaussianMassPolicy::UserSpecified &&
                params.backend != GaussianHMCBackend::ModalDenseFallback)
                // A user-specified mass is never silently replaced by the
                // matched mass, and no fast path is exact for it: the
                // multi-frequency modal fallback is the only sound backend.
                throw std::invalid_argument(
                    "matched HMC: a user-specified mass requires the "
                    "ModalDenseFallback backend");
            if (params.backend == GaussianHMCBackend::SphericalEventQueue &&
                params.policy == GaussianMassPolicy::MatchedPrecision)
                // A general A^{-1} n is dense: one reflection moves every
                // velocity coordinate, and incident-only invalidation would
                // silently miss crossings of non-incident facets.
                throw std::invalid_argument(
                    "matched HMC: dense precision cannot run on the "
                    "incident-only spherical event queue");
            if (params.backend == GaussianHMCBackend::MatchedDiagonalEventQueue &&
                !diag)
                // Same soundness argument: the incident-only queue is exact
                // only when A^{-1} n is supported on the hit coordinates.
                throw std::invalid_argument(
                    "matched HMC: MatchedDiagonalEventQueue requires a "
                    "diagonal precision");
            return params.backend;
        }

        inline NT default_trajectory_length(Polytope &P)
        {
            NT const diameter =
                compute_diameter<Polytope>::template compute<NT>(P);
            return std::min(std::min(diameter, NT(1)), chunk_cap() * _inv_omega);
        }

        // Same true-facet enumeration as the spherical walk, with the
        // Gaussian center folded into the offsets: the facet value along
        // the trajectory is s0*alpha(i0) + s1*alpha(i1) against
        // C = b - n.c (alpha = x - c).
        inline void build_event_facets(Polytope const& P)
        {
            unsigned int const n = P.dimension();
            VT const b_event = P.get_vec();
            NT const rel_scale =
                P.is_normalized() ? NT(1) / std::sqrt(NT(2)) : NT(1);

            _f_i0.clear(); _f_i1.clear(); _f_s0.clear(); _f_s1.clear();
            _f_C.clear(); _f_wall.clear();

            auto add_facet = [&](unsigned int i0, unsigned int i1,
                                 NT s0, NT s1, NT b, bool wall) {
                _f_i0.push_back(i0); _f_i1.push_back(i1);
                _f_s0.push_back(s0); _f_s1.push_back(s1);
                _f_C.push_back(b - (s0 * _center(i0) +
                                    (i1 != i0 ? s1 * _center(i1) : NT(0))));
                _f_wall.push_back(wall ? 1 : 0);
            };

            for (unsigned int i = 0; i < n; ++i)
                if (P.lower_bound_is_facet(i))
                    add_facet(i, i, NT(-1), NT(0), b_event(i), true);

            for (unsigned int i = 0; i < n; ++i)
                if (P.upper_bound_is_facet(i))
                    add_facet(i, i, NT(1), NT(0), b_event(n + i), true);

            for (unsigned int k = 0; k < P.num_order_relations(); ++k) {
                auto rel = P.get_order_relation(k);
                add_facet(rel.first, rel.second, rel_scale, -rel_scale,
                          b_event(2 * n + k), false);
            }
        }

        // Per-facet reflection data: w_f = A^{-1} n_f (dense columns,
        // Cholesky solves; no explicit inverse) and q_f = n_f^T A^{-1} n_f,
        // for the *unnormalized* outward normal n (walls +-e_i, covers
        // e_u - e_v; the reflection formula is invariant to the scale of n,
        // so row normalization of P does not enter here).
        inline void build_reflection_data() { build_reflection_data(_llt); }

        inline void build_reflection_data(Eigen::LLT<MT> const& llt)
        {
            unsigned int const n = static_cast<unsigned int>(_center.size());
            unsigned int const F = num_event_facets();
            MT N(MT::Zero(n, F));
            for (unsigned int fid = 0; fid < F; ++fid) {
                if (_f_wall[fid]) {
                    N(_f_i0[fid], fid) = _f_s0[fid];
                } else {
                    N(_f_i0[fid], fid) = NT(1);
                    N(_f_i1[fid], fid) = NT(-1);
                }
            }
            _w = llt.solve(N);
            _q.resize(F);
            for (unsigned int fid = 0; fid < F; ++fid)
                _q(fid) = normal_dot(fid, _w.col(fid));
        }

        template <typename Vec>
        inline NT normal_dot(unsigned int fid, Vec const& v) const
        {
            return _f_wall[fid]
                 ? _f_s0[fid] * v(_f_i0[fid])
                 : v(_f_i0[fid]) - v(_f_i1[fid]);
        }

        inline NT quad_form(VT const& d) const
        {
            if (_A.size() == 0)   // Identity policy: A = omega^2 I
                return _omega * _omega * d.squaredNorm();
            return d.dot(_A * d);
        }

        // v ~ N(0, M^{-1}) = N(0, omega^2 A^{-1}):  v = omega L^{-T} xi.
        inline void refresh_velocity(RandomNumberGenerator &rng)
        {
            _v = sample_velocity(rng);
        }

        // ModalDenseFallback leg: exact multi-frequency flight between
        // reflections found by the full-scan safeguarded oracle.  The
        // metric reflection uses w = M^{-1} n and q = n^T M^{-1} n, so
        // n^T v flips and v^T M v is preserved; a zero-time hit is a
        // simultaneous corner crossing and consumes no leg time.
        inline bool advance_modal(NT T)
        {
            VT x = _p.getCoefficients();
            VT v = _v.getCoefficients();
            NT t_rem = T;
            unsigned int it = 0;

            unsigned int const F = num_event_facets();
            while (t_rem > NT(0)) {
                _modal->set_state(x, v);
#ifdef VOLESTI_HMC_PROFILE
                // One full-scan oracle call solves every true facet.
                hmc_profile_counters::n_trig_calls.fetch_add(
                    F, std::memory_order_relaxed);
#endif
                auto const hit = first_hit_modal_dense(
                    *_modal, t_rem, _modal_last_hit,
                    ModalOracleOptions<NT>(), &_hard_negative_slack);
                if (hit.facet < 0) {
                    x = _modal->position(t_rem);
                    v = _modal->velocity(t_rem);
                    break;
                }
                x = _modal->position(hit.t);
                v = _modal->velocity(hit.t);
                VOLESTI_HMC_COUNT(n_reflections);
                NT const nv = normal_dot(hit.facet, v);
                v.noalias() -= (NT(2) * nv / _q(hit.facet)) * _w.col(hit.facet);
                _modal_last_hit = hit.facet;
                t_rem -= hit.t;
                if (++it >= _rho) return false;
            }

            // Audit the leg-end state: containment failures that only
            // materialize at the end of a leg would otherwise be invisible
            // to apply_leg callers (the oracle counter fires at the START
            // of the next call).
            _modal->set_state(x, v);
            ModalOracleOptions<NT> const audit;
            for (unsigned int f = 0; f < F; ++f) {
                NT const scale = std::max(NT(1),
                    std::abs(_modal->facet_offset(f)));
                if (_modal->facet_value(f, NT(0)) >
                        audit.hard_slack_tol * scale)
                    ++_hard_negative_slack;
            }

            _p.set_coeffs(x);
            _v.set_coeffs(v);
            return true;
        }

        inline bool advance_full_scan(NT T)
        {
            NT theta_rem = _omega * T;
            unsigned int it = 0;

            while (theta_rem > NT(0)) {
                NT const theta_chunk = std::min(theta_rem, chunk_cap());
                begin_chunk(theta_chunk);

                for (;;) {
                    unsigned int fid;
                    if (!next_event_full_scan(fid)) {
                        materialize_global_state(_end_cos, _end_sin);
                        theta_rem -= theta_chunk;
                        break;
                    }
                    VOLESTI_HMC_COUNT(n_reflections);
                    reflect_event(fid);
                    if (++it >= _rho) return false;
                }
            }
            return true;
        }

        inline void begin_chunk(NT theta_chunk)
        {
            _alpha = _p.getCoefficients() - _center;
            _beta  = _v.getCoefficients() * _inv_omega;
            _end_cos = std::cos(theta_chunk);
            _end_sin = std::sin(theta_chunk);
            _key_end = order_polytope_hmc_detail::one_minus_cos_stable(
                _end_cos, _end_sin);
            _cur_cos = NT(1);
            _cur_sin = NT(0);
        }

        // Earliest outgoing crossing over *all* true facets in the window
        // (theta_cur, theta_end].  Root selection, the last-hit residual
        // guard, the simultaneous-crossing snap and the hard-negative-slack
        // diagnostics mirror the hardened spherical solver; there is no
        // event queue to keep consistent because every reflection changes
        // all velocity coordinates and the whole scan re-runs anyway.
        inline bool next_event_full_scan(unsigned int& fid_out)
        {
            unsigned int const F = num_event_facets();
            bool found = false;
            NT best_key = NT(0), best_cos = NT(0), best_sin = NT(0);

            for (unsigned int fid = 0; fid < F; ++fid) {
                NT key, c, s;
                if (!solve_facet(fid, key, c, s)) continue;
                if (!found || key < best_key) {
                    found = true;
                    best_key = key; best_cos = c; best_sin = s;
                    fid_out = fid;
                }
            }
            if (!found) return false;

            _last_hit_fid = fid_out;
            _cur_cos = best_cos;
            _cur_sin = best_sin;
            return true;
        }

        inline bool solve_facet(unsigned int fid, NT& key_out,
                                NT& cos_out, NT& sin_out)
        {
            VOLESTI_HMC_COUNT(n_trig_calls);
            NT const A = _f_s0[fid] * _alpha(_f_i0[fid]) + _f_s1[fid] * _alpha(_f_i1[fid]);
            NT const B = _f_s0[fid] * _beta(_f_i0[fid]) + _f_s1[fid] * _beta(_f_i1[fid]);
            NT const C = _f_C[fid];

            NT const R2 = A * A + B * B;
            NT const scale = std::max(NT(1),
                std::max(std::abs(A) + std::abs(B), std::abs(C)));
            NT const veps = value_tol() * scale;
            NT const disc = R2 - C * C;
            if (disc < -veps * scale || R2 <= veps * veps)
                return false;

            NT const val_cur = A * _cur_cos + B * _cur_sin;
            NT const slack = C - val_cur;
            if (slack < -veps) {
                if (slack < -hard_slack_tol() * scale)
                    ++_hard_negative_slack;
                return false;
            }

            NT const D = disc > NT(0) ? std::sqrt(disc) : NT(0);
            NT const inv_R2 = NT(1) / R2;
            NT const c = (A * C + B * D) * inv_R2;
            NT const s = (B * C - A * D) * inv_R2;

            NT const cross = _cur_cos * s - _cur_sin * c;
            if (fid == _last_hit_fid) {
                if (cross <= _angle_eps) return false;
            } else if (cross <= NT(0)) {
                if (cross <= -_angle_eps || D <= NT(0) ||
                    _cur_cos * c + _cur_sin * s <= NT(0)) return false;
                key_out = order_polytope_hmc_detail::one_minus_cos_stable(
                    _cur_cos, _cur_sin);
                cos_out = _cur_cos;
                sin_out = _cur_sin;
                return true;
            }

            // A real crossing in the window is at most a quarter turn ahead
            // of the current direction (dot >= 0 up to rounding).  This
            // rejects the antipodal root of an already-reflected C == 0
            // facet, whose key can sneak inside the tolerance band when a
            // reflection lands within ~1e-12 of a full quarter chunk's end
            // (the spherical walk is protected structurally: it drops that
            // facet's event and never re-solves non-incident facets, but
            // the full scan re-solves everything after each reflection).
            if (_cur_cos * c + _cur_sin * s < NT(-0.5)) return false;

            NT const k = order_polytope_hmc_detail::one_minus_cos_stable(c, s);
            if (k > _key_end + key_tol()) return false;

            key_out = k; cos_out = c; sin_out = s;
            return true;
        }

        // Matched reflection at the current direction, then rebase alpha
        // and beta for the whole state: A^{-1} n is dense, so every
        // coordinate's velocity (hence beta) changes.
        inline void reflect_event(unsigned int fid)
        {
            NT const cV = _cur_cos;
            NT const sV = _cur_sin;
            VT X = _alpha * cV + _beta * sV;    // x(theta*) - c
            VT Y = _beta * cV - _alpha * sV;    // v(theta*) / omega

            NT const nY = normal_dot(fid, Y);
            Y.noalias() -= (NT(2) * nY / _q(fid)) * _w.col(fid);

            _alpha = X * cV - Y * sV;
            _beta  = X * sV + Y * cV;
        }

        inline void materialize_global_state(NT c, NT s)
        {
            _p.set_coeffs(_center + _alpha * c + _beta * s);
            _v.set_coeffs((_beta * c - _alpha * s) * _omega);
        }

        static inline unsigned int no_facet()
        {
            return std::numeric_limits<unsigned int>::max();
        }
        static inline NT chunk_cap() { return NT(1.57079632679489661923); }
        inline NT time_eps() const { return NT(1e-10); }
        inline NT key_tol() const { return NT(1e-12); }
        inline NT value_tol() const { return NT(1e-12); }
        inline NT hard_slack_tol() const { return NT(1e-7); }

        GaussianHMCBackend _backend;
        std::unique_ptr<SphericalWalk> _spherical;
        typedef OrderPolytopeModalGaussianTrajectory<Polytope> ModalTraj;
        std::unique_ptr<ModalTraj> _modal;
        Eigen::LLT<MT> _lltM;
        MT _M;
        int _modal_last_hit;

        NT _omega, _inv_omega, _angle_eps, _Len;
        unsigned int _rho;
        VT _center;
        MT _A;
        Eigen::LLT<MT> _llt;

        Point _p, _v;
        VT _alpha, _beta;
        NT _end_cos, _end_sin, _key_end, _cur_cos, _cur_sin;
        unsigned int _last_hit_fid;
        unsigned long long _hard_negative_slack;

        // Facet value data (scaled like the polytope rows) and matched
        // reflection data (unnormalized normals).
        std::vector<unsigned int> _f_i0, _f_i1;
        std::vector<NT> _f_s0, _f_s1, _f_C;
        std::vector<unsigned char> _f_wall;
        MT _w;     // n x F: columns A^{-1} n_f
        VT _q;     // F:     n_f^T A^{-1} n_f
    };
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_HMC_MATCHED_WALK_HPP
