// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigen>

// Modal trajectory kernel for exact Gaussian HMC on an order polytope
// with an *arbitrary* SPD target precision A and an *arbitrary* SPD
// momentum mass M:
//
//     U(x) = 1/2 (x - c)^T A (x - c),      p ~ N(0, M),  v = M^{-1} p,
//
//     xdot = v,   vdot = -M^{-1} A (x - c).
//
// The dynamics decouple in generalized eigencoordinates.  With the
// Cholesky reduction M = L L^T, form the symmetric
//
//     B = L^{-1} A L^{-T} = Q Lambda Q^T,      U = L^{-T} Q,
//
// so that A u_k = lambda_k M u_k and U^T M U = Q^T Q = I.  For
// y = x - c = U q the modes evolve independently at frequencies
// omega_k = sqrt(lambda_k) (> 0 for SPD A):
//
//     q_k(t) = q_k(0) cos(omega_k t) + qdot_k(0)/omega_k sin(omega_k t),
//     q(0)   = U^T M y(0) = Q^T L^T y(0),
//     qdot(0)= U^T M v(0) = Q^T L^T v(0).
//
// Facet values are multi-frequency trigonometric sums: for a true order-
// polytope facet with (sparse) normal n_f and offset b_f,
//
//     g_f(t) = n_f^T x(t) - b_f
//            = sum_k W_fk (q_k cos(omega_k t) + qdot_k/omega_k sin) - C_f,
//
// with the modal projection W_fk = n_f^T u_k and C_f = b_f - n_f^T c.
// Only the true facets enter (lower bounds of minimal elements, upper
// bounds of maximal elements, Hasse cover edges), and W is assembled
// from single rows / row differences of U — no dense normals are built.
//
// This kernel is deliberately correctness-first: every evaluation is an
// O(n) modal sum, there is no one-frequency angle-space shortcut, and it
// exists to back the ModalDenseFallback boundary oracle and walk.

template <typename Polytope>
struct OrderPolytopeModalGaussianTrajectory
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::VT VT;
    typedef typename Polytope::MT MT;

    OrderPolytopeModalGaussianTrajectory(Polytope const& P, MT const& A,
                                         MT const& M, VT const& center)
    {
        unsigned int const n = P.dimension();
        if (A.rows() != static_cast<Eigen::Index>(n) || A.cols() != A.rows() ||
            M.rows() != static_cast<Eigen::Index>(n) || M.cols() != M.rows())
            throw std::invalid_argument("modal HMC: A/M dimension mismatch");
        if (!A.isApprox(A.transpose()) || !M.isApprox(M.transpose()))
            throw std::invalid_argument("modal HMC: A and M must be symmetric");
        _center = center.size() == 0 ? VT(VT::Zero(n)) : VT(center);
        if (_center.size() != static_cast<Eigen::Index>(n))
            throw std::invalid_argument("modal HMC: center dimension mismatch");
        _A = A;
        _M = M;

        Eigen::LLT<MT> lltM(M);
        if (lltM.info() != Eigen::Success)
            throw std::invalid_argument("modal HMC: M is not SPD");
        MT const L = lltM.matrixL();

        // B = L^{-1} A L^{-T}, symmetrized against rounding.
        MT B = L.template triangularView<Eigen::Lower>().solve(A);
        B = L.template triangularView<Eigen::Lower>()
             .solve(MT(B.transpose()));
        B = NT(0.5) * (B + MT(B.transpose()));

        Eigen::SelfAdjointEigenSolver<MT> es(B);
        if (es.info() != Eigen::Success)
            throw std::invalid_argument("modal HMC: eigen decomposition failed");
        if (!(es.eigenvalues().minCoeff() > NT(0)))
            throw std::invalid_argument("modal HMC: A is not SPD");

        // U = L^{-T} Q;  U^T M = Q^T L^T (used for the state projection).
        _U = L.transpose().template triangularView<Eigen::Upper>()
              .solve(MT(es.eigenvectors()));
        _proj = MT(es.eigenvectors().transpose()) * L.transpose();
        _omega.resize(n);
        for (unsigned int k = 0; k < n; ++k)
            _omega(k) = std::sqrt(es.eigenvalues()(k));
        _omega_max = _omega.maxCoeff();

        build_facets(P);
        _q0 = VT::Zero(n);
        _qd0 = VT::Zero(n);
    }

    // Project a phase-space state onto the modes; all evaluations below
    // refer to the trajectory launched from this state at t = 0.
    void set_state(VT const& x, VT const& v)
    {
        _q0.noalias() = _proj * (x - _center);
        _qd0.noalias() = _proj * v;
    }

    unsigned int dimension() const
    {
        return static_cast<unsigned int>(_center.size());
    }
    unsigned int num_facets() const
    {
        return static_cast<unsigned int>(_f_C.size());
    }
    NT omega_max() const { return _omega_max; }
    VT const& frequencies() const { return _omega; }
    NT facet_offset(unsigned int f) const { return _f_C(f); }

    VT position(NT t) const
    {
        unsigned int const n = dimension();
        VT q(n);
        for (unsigned int k = 0; k < n; ++k) {
            NT const w = _omega(k);
            q(k) = _q0(k) * std::cos(w * t) + _qd0(k) / w * std::sin(w * t);
        }
        return _center + _U * q;
    }

    VT velocity(NT t) const
    {
        unsigned int const n = dimension();
        VT qd(n);
        for (unsigned int k = 0; k < n; ++k) {
            NT const w = _omega(k);
            qd(k) = -_q0(k) * w * std::sin(w * t) + _qd0(k) * std::cos(w * t);
        }
        return _U * qd;
    }

    NT coordinate(unsigned int i, NT t) const
    {
        NT val = _center(i);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const w = _omega(k);
            val += _U(i, k) * (_q0(k) * std::cos(w * t)
                               + _qd0(k) / w * std::sin(w * t));
        }
        return val;
    }

    NT velocity_coordinate(unsigned int i, NT t) const
    {
        NT val = NT(0);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const w = _omega(k);
            val += _U(i, k) * (-_q0(k) * w * std::sin(w * t)
                               + _qd0(k) * std::cos(w * t));
        }
        return val;
    }

    // g_f(t) = n_f^T x(t) - b_f: negative inside, zero on the facet.
    NT facet_value(unsigned int f, NT t) const
    {
        NT val = -_f_C(f);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const w = _omega(k);
            val += _W(f, k) * (_q0(k) * std::cos(w * t)
                               + _qd0(k) / w * std::sin(w * t));
        }
        return val;
    }

    NT facet_derivative(unsigned int f, NT t) const
    {
        NT val = NT(0);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const w = _omega(k);
            val += _W(f, k) * (-_q0(k) * w * std::sin(w * t)
                               + _qd0(k) * std::cos(w * t));
        }
        return val;
    }

    NT energy(VT const& x, VT const& v) const
    {
        VT const d = x - _center;
        return NT(0.5) * d.dot(_A * d) + NT(0.5) * v.dot(_M * v);
    }

    // Facet metadata for walk-level reflections: sparse normal pattern
    // (walls: sign * e_i; covers: e_u - e_v scaled by rel_scale in the
    // *value* row, unscaled in the reflection normal).
    unsigned int facet_i0(unsigned int f) const { return _f_i0[f]; }
    unsigned int facet_i1(unsigned int f) const { return _f_i1[f]; }
    bool facet_is_wall(unsigned int f) const { return _f_wall[f] != 0; }
    NT facet_s0(unsigned int f) const { return _f_s0[f]; }
    NT facet_s1(unsigned int f) const { return _f_s1[f]; }

private:

    void build_facets(Polytope const& P)
    {
        unsigned int const n = P.dimension();
        VT const b_event = P.get_vec();
        NT const rel_scale =
            P.is_normalized() ? NT(1) / std::sqrt(NT(2)) : NT(1);

        std::vector<unsigned int> i0s, i1s;
        std::vector<NT> s0s, s1s, bs;
        std::vector<unsigned char> walls;

        for (unsigned int i = 0; i < n; ++i)
            if (P.lower_bound_is_facet(i)) {
                i0s.push_back(i); i1s.push_back(i);
                s0s.push_back(NT(-1)); s1s.push_back(NT(0));
                bs.push_back(b_event(i)); walls.push_back(1);
            }
        for (unsigned int i = 0; i < n; ++i)
            if (P.upper_bound_is_facet(i)) {
                i0s.push_back(i); i1s.push_back(i);
                s0s.push_back(NT(1)); s1s.push_back(NT(0));
                bs.push_back(b_event(n + i)); walls.push_back(1);
            }
        for (unsigned int k = 0; k < P.num_order_relations(); ++k) {
            auto rel = P.get_order_relation(k);
            i0s.push_back(rel.first); i1s.push_back(rel.second);
            s0s.push_back(rel_scale); s1s.push_back(-rel_scale);
            bs.push_back(b_event(2 * n + k)); walls.push_back(0);
        }

        unsigned int const F = static_cast<unsigned int>(bs.size());
        _f_i0 = i0s; _f_i1 = i1s; _f_s0 = s0s; _f_s1 = s1s; _f_wall = walls;
        _W.resize(F, n);
        _f_C.resize(F);
        for (unsigned int f = 0; f < F; ++f) {
            _W.row(f) = _f_s0[f] * _U.row(_f_i0[f]);
            NT cdot = _f_s0[f] * _center(_f_i0[f]);
            if (_f_i1[f] != _f_i0[f]) {
                _W.row(f) += _f_s1[f] * _U.row(_f_i1[f]);
                cdot += _f_s1[f] * _center(_f_i1[f]);
            }
            _f_C(f) = bs[f] - cdot;
        }
    }

    MT _A, _M, _U, _proj, _W;
    VT _center, _omega, _q0, _qd0, _f_C;
    NT _omega_max;
    std::vector<unsigned int> _f_i0, _f_i1;
    std::vector<NT> _f_s0, _f_s1;
    std::vector<unsigned char> _f_wall;
};


template <typename NT>
struct ModalOracleOptions
{
    // Grid step is pi / (grid_divisor * omega_max): at the default 8 the
    // fastest mode is sampled 16 times per period, so only sub-cell
    // grazes (a facet value poking above zero and back inside one cell)
    // can be missed; those self-heal like the eps-band misses of the
    // one-frequency walks and are counted when large.
    NT grid_divisor = NT(8);
    // Residual-root suppression window for the just-reflected facet only.
    NT t_eps = NT(1e-10);
    NT value_tol = NT(1e-12);
    NT hard_slack_tol = NT(1e-7);
    unsigned int bisect_iters = 100;
};

template <typename NT>
struct ModalFirstHit
{
    NT t;
    int facet;   // -1: no hit before t_max
};

// Correctness-first earliest-hit oracle for the modal trajectory: full
// scan over all true facets, no event queue, no incident-only
// invalidation, no one-frequency shortcut.  Each facet value g_f(t) is a
// multi-frequency trigonometric sum, so roots are located by a
// safeguarded scan: march a grid fine enough for the fastest mode,
// bracket the first inside-to-outside sign change, and refine it by
// bisection.  Only the just-reflected facet's residual near-zero root is
// suppressed (within t_eps); a near-zero root of any *other* facet is a
// real simultaneous crossing and is returned (possibly at t == 0).
template <typename Polytope>
ModalFirstHit<typename Polytope::PointType::FT>
first_hit_modal_dense(OrderPolytopeModalGaussianTrajectory<Polytope> const& traj,
                      typename Polytope::PointType::FT t_max,
                      int last_hit_facet,
                      ModalOracleOptions<typename Polytope::PointType::FT> const&
                          opt = ModalOracleOptions<typename Polytope::PointType::FT>(),
                      unsigned long long* hard_negative_slack = nullptr)
{
    typedef typename Polytope::PointType::FT NT;

    ModalFirstHit<NT> best{std::numeric_limits<NT>::max(), -1};
    if (!(t_max > NT(0))) {
        best.t = t_max;
        return best;
    }

    NT const pi = NT(3.14159265358979323846);
    NT const delta = pi / (opt.grid_divisor * traj.omega_max());
    unsigned int const F = traj.num_facets();

    for (unsigned int f = 0; f < F; ++f) {
        NT const scale = std::max(NT(1), std::abs(traj.facet_offset(f)));
        NT const ztol = opt.value_tol * scale;

        NT const g0 = traj.facet_value(f, NT(0));
        if (g0 > ztol) {
            // The state is outside this facet (a crossing was skipped
            // within tolerance upstream).  The re-entry must not be
            // reflected — and the march below cannot bracket one, since
            // brackets require g_prev <= ztol first — but the facet is
            // NOT skipped for the whole call: after the flow re-enters,
            // a genuine second exit later in the leg is caught normally
            // (modal legs have no chunk boundaries to re-scan from).
            // Large violations are a real containment failure and are
            // counted.
            if (g0 > opt.hard_slack_tol * scale && hard_negative_slack)
                ++*hard_negative_slack;
        } else if (g0 >= -ztol && static_cast<int>(f) != last_hit_facet &&
                   traj.facet_derivative(f, NT(0)) > NT(0)) {
            // On the facet and genuinely leaving: an exactly simultaneous
            // crossing with the previous reflection — reflect now.
            best.t = NT(0);
            best.facet = static_cast<int>(f);
            continue;
        }

        // Bisection towards g == 0 (NOT towards g == ztol, which would
        // bias every hit a full tolerance band outside the facet and make
        // the landing state read as an upstream containment violation).
        auto refine = [&](NT lo, NT hi) {
            for (unsigned int it = 0; it < opt.bisect_iters &&
                                      hi - lo > NT(1e-15) * (NT(1) + hi);
                 ++it) {
                NT const mid = NT(0.5) * (lo + hi);
                if (traj.facet_value(f, mid) > NT(0)) hi = mid;
                else lo = mid;
            }
            return NT(0.5) * (lo + hi);
        };

        // March the grid for the first inside-to-outside transition.
        NT t_prev = NT(0);
        NT g_prev = g0;
        NT gd_prev = traj.facet_derivative(f, NT(0));
        while (t_prev < t_max && t_prev < best.t) {
            NT const t_next = std::min(t_prev + delta, t_max);
            NT const g_next = traj.facet_value(f, t_next);
            NT const gd_next = traj.facet_derivative(f, t_next);
            NT root = NT(-1);
            if (g_prev <= ztol && g_next > ztol) {
                root = refine(t_prev, t_next);
            } else if (g_prev <= ztol && g_next <= ztol &&
                       gd_prev > NT(0) && gd_next < NT(0)) {
                // Both endpoints inside but the value peaks within the
                // cell: locate the local maximum by bisecting the
                // derivative and check whether the peak pokes outside —
                // this catches sub-cell grazes the plain sign scan would
                // miss (out and back in within one grid step).
                NT lo = t_prev, hi = t_next;
                for (unsigned int it = 0; it < 60; ++it) {
                    NT const mid = NT(0.5) * (lo + hi);
                    if (traj.facet_derivative(f, mid) > NT(0)) lo = mid;
                    else hi = mid;
                }
                NT const t_peak = NT(0.5) * (lo + hi);
                if (traj.facet_value(f, t_peak) > ztol)
                    root = refine(t_prev, t_peak);
            }
            if (root >= NT(0) && traj.facet_derivative(f, root) <= NT(0))
                // Not a genuine inside-to-outside crossing: a state in the
                // (0, ztol] band with no zero in the cell collapses the
                // bisection onto t_prev with an *inbound* derivative (the
                // trajectory is entering, not leaving), and a perfectly
                // tangent contact reflects as a no-op anyway.  Fabricating
                // a hit here would flip an inbound velocity outbound and
                // arm the residual guard on the wrong facet - keep
                // marching instead.
                root = NT(-1);
            if (root >= NT(0)) {
                bool const residual =
                    static_cast<int>(f) == last_hit_facet && root <= opt.t_eps;
                if (!residual) {
                    if (root < best.t) {
                        best.t = root;
                        best.facet = static_cast<int>(f);
                    }
                    break;   // earliest crossing of this facet found
                }
                // Residual root of the just-reflected facet: keep marching.
            }
            t_prev = t_next;
            g_prev = g_next;
            gd_prev = gd_next;
        }
    }

    if (best.facet == -1) best.t = t_max;
    return best;
}

#endif // RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP
