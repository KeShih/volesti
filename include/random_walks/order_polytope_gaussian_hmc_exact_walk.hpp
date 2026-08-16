// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
#include <cstdio>
#endif

#include "random_walks/compute_diameter.hpp"
#include "random_walks/gaussian_hamiltonian_monte_carlo_exact_walk.hpp"

#ifdef VOLESTI_HMC_PROFILE
#include <atomic>
namespace hmc_profile_counters {
inline std::atomic<unsigned long long> n_trajectories{0}, n_reflections{0};
inline std::atomic<unsigned long long> n_stale_purge{0}, n_trig_calls{0}, n_trig_hits{0};
inline std::atomic<unsigned long long> n_heap_insert{0}, n_heap_remove{0}, n_rebase{0};
inline std::atomic<unsigned long long> n_hard_negative_slack{0};
inline void bump(std::atomic<unsigned long long>& c)
{
    c.fetch_add(1, std::memory_order_relaxed);
}
}
#define VOLESTI_HMC_COUNT(counter) hmc_profile_counters::bump(hmc_profile_counters::counter)
#else
#define VOLESTI_HMC_COUNT(counter) ((void)0)
#endif

namespace order_polytope_hmc_detail {
// 1 - cos(theta) from the unit direction (c, s) = (cos(theta), sin(theta)).
// The literal subtraction cancels catastrophically at small angles (below
// theta ~ 1e-8 all keys collapse onto {0, 1.1e-16}, losing the ordering of
// distinct events); for c away from 1 it is exact.  Since c^2 + s^2 = 1,
// 1 - c == s^2 / (1 + c), and the quotient form keeps full relative
// precision where the subtraction loses it.
template <typename NT>
inline NT one_minus_cos_stable(NT c, NT s)
{
    return c > NT(0.5) ? s * s / (NT(1) + c) : NT(1) - c;
}
}

// Specialized exact Gaussian HMC walk for OrderPolytope.
//
// An event-queue boundary oracle exploits the sparse facet structure of order
// polytopes: each reflection only recomputes the facets incident to the
// reflected coordinates (O(degree) work instead of O(all facets)).
//
// All boundary-hit computations are carried out in *angle space*.  With
// theta = omega*t the trajectory is p(theta) = alpha*cos(theta) +
// beta*sin(theta) and each facet value is v(theta) = A*cos(theta) +
// B*sin(theta), hit when v = C.  The hit *directions* (cos(theta*),
// sin(theta*)) of the two roots per turn are obtained algebraically:
//
//     D = sqrt(R^2 - C^2),  R^2 = A^2 + B^2,
//     (cos, sin) = ((A*C -+ B*D) / R^2, (B*C +- A*D) / R^2),
//
// so the reflection loop needs no atan2/acos/sincos at all.  Events are
// ordered by the key 1 - cos(theta*), which is monotone in theta on [0, pi];
// legs are processed in chunks of at most a quarter period so the key stays
// monotone and no wrap-around bookkeeping is needed.  Forward/eps tests use
// the cross product sin(theta* - theta_cur) of the stored unit directions.

struct OrderPolytopeGaussianHamiltonianMonteCarloExactWalk
{
    struct parameters
    {
        parameters(double L, bool set, unsigned int _rho, bool _set_rho,
                   unsigned int _rebase_interval = 1024)
            : m_L(L), set_L(set), rho(_rho), set_rho(_set_rho),
              rebase_interval(_rebase_interval)
        {}

        double m_L;
        bool set_L;
        unsigned int rho;
        bool set_rho;
        unsigned int rebase_interval;
    };

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalk(double L, unsigned int _rho)
        : param(L, true, _rho, true) {}

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalk(double L)
        : param(L, true, 0, false) {}

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalk()
        : param(0, false, 0, false) {}

    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng)
            : Walk(P, p, a_i, rng, parameters(0, false, 0, false))
        {}

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng,
             parameters const& params)
        {
            _omega = std::sqrt(NT(2) * a_i);
            _inv_omega = NT(1) / _omega;
            _angle_eps = _omega * time_eps();
            _weighted = false;
            _has_center = false;
            _center = VT::Zero(P.dimension());
            _Len = params.set_L ? params.m_L : default_trajectory_length(P);
            _rho = params.set_rho ? params.rho : 100 * P.dimension();
            _rebase_interval = params.rebase_interval;
            _last_hit_fid = no_facet();
            _hard_negative_slack = 0;
            build_event_facets(P);
            initialize(P, p, rng);
        }

        // Matched-mass walk for a *diagonal* Gaussian precision
        // A = diag(a_diag) with center c and mass M = A / omega^2: the
        // trajectory x(theta) = c + alpha cos(theta) + beta sin(theta)
        // keeps the common frequency omega, and because A^{-1} n is
        // supported on the hit coordinate(s) only, a reflection changes
        // only those coordinates — the incident-only event-queue
        // invalidation stays exact.  Wall reflections still flip one
        // velocity; cover reflections use the weighted formula
        //     v_u+ = v_u - r_u (v_u - v_v),   r_u = 2 a_v / (a_u + a_v),
        //     v_v+ = v_v + r_v (v_u - v_v),   r_v = 2 a_u / (a_u + a_v),
        // which reduces to the spherical velocity swap for a_u = a_v.
        // Momentum refresh is v_i ~ N(0, omega^2 / a_i).
        Walk(Polytope &P, Point const& p, RandomNumberGenerator &rng,
             VT const& a_diag, VT const& center, NT const& omega,
             parameters const& params)
        {
            unsigned int const n = P.dimension();
            _omega = omega;
            _inv_omega = NT(1) / _omega;
            _angle_eps = _omega * time_eps();
            _weighted = true;
            _a_diag = a_diag;
            _center = center;
            _has_center = _center.squaredNorm() != NT(0);
            _v_sigma.resize(n);
            for (unsigned int i = 0; i < n; ++i)
                _v_sigma(i) = _omega / std::sqrt(_a_diag(i));
            _Len = params.set_L ? params.m_L : default_trajectory_length(P);
            _rho = params.set_rho ? params.rho : 100 * n;
            _rebase_interval = params.rebase_interval;
            _last_hit_fid = no_facet();
            _hard_negative_slack = 0;
            build_event_facets(P);
            initialize(P, p, rng);
        }

        // The a_i argument is ignored: omega is fixed at construction, like
        // in GaussianHamiltonianMonteCarloExactWalk (callers construct a
        // fresh walk per annealing phase).
        inline void apply(Polytope const& P, Point& p, NT const&,
                          unsigned int const& walk_length, RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            NT T;

            for (auto j=0u; j<walk_length; ++j)
            {
                T = rng.sample_urdist() * _Len;
                Point const v0 = _v;
                refresh_velocity(n, rng);
                // A fresh momentum invalidates the residual-root guard: the
                // state may still sit exactly on the last reflected facet,
                // but with the new velocity a root at the current angle is a
                // real crossing whenever the motion is outward (handled by
                // the simultaneous-crossing path in next_event_for_facet)
                // and only the incoming root otherwise, which is never
                // scheduled.  Keeping the guard armed here would suppress
                // the true hit of an outward refresh and leak the leg.
                _last_hit_fid = no_facet();
                Point p0 = _p;

                if (!advance_event_queue(P, T)) {
                    // Restore the pre-leg state, velocity included, so that
                    // (position, velocity()) stays one trajectory state.
                    // The residual-root guard is re-cleared: the aborted
                    // leg's last hit is unrelated to p0, and a p0 sitting
                    // exactly on a facet is handled by the simultaneous-
                    // crossing path like any fresh leg.
                    _p = p0;
                    _v = v0;
                    _last_hit_fid = no_facet();
                }
            }
            p = _p;
        }

        inline void update_delta(NT L) { _Len = L; }

        // Deterministically advance one HMC leg from state (p, v) for time
        // T without touching the RNG: the entry point for tests and
        // diagnostic harnesses.  Like a momentum refresh, starting a leg
        // clears the residual-root guard (see apply).  Returns false when
        // the leg aborts on the reflection budget; p is then unchanged.
        inline bool apply_leg(Polytope const& P, Point& p, Point const& v, NT const& T)
        {
            _p = p;
            _v = v;
            _last_hit_fid = no_facet();
            if (!advance_event_queue(P, T)) {
                _p = p;
                _v = v;
                _last_hit_fid = no_facet();
                return false;
            }
            p = _p;
            return true;
        }

        // Number of solves that found the state on the outer side of a
        // facet by more than hard_slack_tol(): a real containment violation
        // upstream, not an eps-band graze self-healing.
        inline unsigned long long hard_negative_slack_count() const
        {
            return _hard_negative_slack;
        }

        // End velocity of the last completed leg (matches the state of the
        // position returned by apply / apply_leg).
        inline Point velocity() const { return _v; }

    private:

        // Indexed D-ary min-heap over pending facet events.  The heap only
        // ever contains facets with a pending hit inside the current chunk,
        // so its depth tracks the (small) active set, not the facet count.
        class IndexedDHeap
        {
            static constexpr int D = 4;
            struct Entry { NT key; unsigned int id; };
            std::vector<Entry> _h;
            std::vector<int> _pos;

            int par(int i) const { return (i - 1) / D; }
            int child0(int i) const { return D * i + 1; }

            // Hole-based sifts: carry the displaced entry in a register and
            // shift blockers down/up, one position write per level.  Keys
            // live inline in the heap array, so comparisons never gather
            // through a side table (a 4-child group spans one cache line).
            void sift_up(int i)
            {
                Entry const e = _h[i];
                while (i > 0) {
                    int const p = par(i);
                    if (e.key < _h[p].key) {
                        _h[i] = _h[p];
                        _pos[_h[i].id] = i;
                        i = p;
                    } else break;
                }
                _h[i] = e;
                _pos[e.id] = i;
            }

            void sift_down(int i)
            {
                int const n = static_cast<int>(_h.size());
                Entry const e = _h[i];
                for (;;) {
                    int const fc = child0(i);
                    if (fc >= n) break;
                    int const lc = std::min(fc + D, n);
                    int best = fc;
                    NT best_key = _h[fc].key;
                    for (int c = fc + 1; c < lc; ++c) {
                        if (_h[c].key < best_key) {
                            best_key = _h[c].key;
                            best = c;
                        }
                    }
                    if (best_key < e.key) {
                        _h[i] = _h[best];
                        _pos[_h[i].id] = i;
                        i = best;
                    } else break;
                }
                _h[i] = e;
                _pos[e.id] = i;
            }

        public:
            void init(unsigned int cap)
            {
                _h.clear();
                _h.reserve(cap);
                _pos.assign(cap, -1);
            }

            void clear()
            {
                for (Entry const& e : _h) _pos[e.id] = -1;
                _h.clear();
            }

            bool empty() const { return _h.empty(); }

            void insert_or_update(unsigned int id, NT key)
            {
                VOLESTI_HMC_COUNT(n_heap_insert);
                int const i = _pos[id];
                if (i >= 0) {
                    NT const old_key = _h[i].key;
                    _h[i].key = key;
                    if (key < old_key) sift_up(i);
                    else sift_down(i);
                } else {
                    int const j = static_cast<int>(_h.size());
                    _h.push_back(Entry{key, id});
                    _pos[id] = j;
                    sift_up(j);
                }
            }

            void remove(unsigned int id)
            {
                int const i = _pos[id];
                if (i < 0) return;
                int const last = static_cast<int>(_h.size()) - 1;
                _pos[id] = -1;
                if (i == last) { _h.pop_back(); return; }
                NT const removed_key = _h[i].key;
                _h[i] = _h[last];
                _pos[_h[i].id] = i;
                _h.pop_back();
                if (_h[i].key < removed_key) sift_up(i);
                else sift_down(i);
            }

            bool extract_min_leq(NT limit, unsigned int& id_out, NT& key_out)
            {
                if (_h.empty() || _h[0].key > limit) return false;
                VOLESTI_HMC_COUNT(n_heap_remove);
                id_out = _h[0].id;
                key_out = _h[0].key;
                remove(id_out);
                return true;
            }
        };

        inline NT default_trajectory_length(Polytope &P)
        {
            // Quarter period fully decorrelates the free Gaussian dynamics
            // (x(T) = v(0)/omega at omega*T = pi/2), so longer legs only add
            // reflections without mixing benefit.  The cap matters in
            // annealing schedules where a_i (hence omega) grows large.
            NT diameter = compute_diameter<Polytope>::template compute<NT>(P);
            return std::min(std::min(diameter, NT(1)), chunk_cap() * _inv_omega);
        }

        inline void build_event_facets(Polytope const& P)
        {
            unsigned int n = P.dimension();
            VT b_event = P.get_vec();
            NT const rel_scale = P.is_normalized() ? NT(1) / std::sqrt(NT(2)) : NT(1);

            _f_i0.clear(); _f_i1.clear(); _f_s0.clear(); _f_s1.clear();
            _f_C.clear(); _f_wall.clear(); _f_row.clear();
            _f_ru.clear(); _f_rv.clear();

            // The Gaussian center is folded into the offsets: the value
            // along the trajectory is s0*alpha(i0) + s1*alpha(i1) with
            // alpha = x - c, hit when it reaches C = b - n.c.  Cover
            // reflections carry the weighted coefficients r_u, r_v (both 1
            // in the spherical case, where they reduce to the swap).
            auto add_facet = [&](unsigned int i0, unsigned int i1, NT s0, NT s1,
                                 NT b, bool wall, unsigned int row) {
                _f_i0.push_back(i0); _f_i1.push_back(i1);
                _f_s0.push_back(s0); _f_s1.push_back(s1);
                _f_C.push_back(b - (s0 * _center(i0) +
                                    (i1 != i0 ? s1 * _center(i1) : NT(0))));
                _f_wall.push_back(wall ? 1 : 0); _f_row.push_back(row);
                if (!wall && _weighted) {
                    NT const au = _a_diag(i0), av = _a_diag(i1);
                    _f_ru.push_back(NT(2) * av / (au + av));
                    _f_rv.push_back(NT(2) * au / (au + av));
                } else {
                    _f_ru.push_back(NT(1));
                    _f_rv.push_back(NT(1));
                }
            };

            for (unsigned int i = 0; i < n; ++i)
                if (P.lower_bound_is_facet(i))
                    add_facet(i, i, NT(-1), NT(0), b_event(i), true, i);

            for (unsigned int i = 0; i < n; ++i)
                if (P.upper_bound_is_facet(i))
                    add_facet(i, i, NT(1), NT(0), b_event(n + i), true, n + i);

            for (unsigned int k = 0; k < P.num_order_relations(); ++k) {
                auto rel = P.get_order_relation(k);
                add_facet(rel.first, rel.second, rel_scale, -rel_scale,
                          b_event(2 * n + k), false, 2 * n + k);
            }

            unsigned int const F = static_cast<unsigned int>(_f_i0.size());

            // CSR of facets incident to each coordinate (wall facets are
            // registered once; cover facets under both endpoints).
            _inc_off.assign(n + 1, 0);
            for (unsigned int fid = 0; fid < F; ++fid) {
                ++_inc_off[_f_i0[fid] + 1];
                if (_f_i1[fid] != _f_i0[fid]) ++_inc_off[_f_i1[fid] + 1];
            }
            for (unsigned int i = 0; i < n; ++i) _inc_off[i + 1] += _inc_off[i];
            _inc_ids.resize(_inc_off[n]);
            {
                std::vector<unsigned int> cursor(_inc_off.begin(), _inc_off.end() - 1);
                for (unsigned int fid = 0; fid < F; ++fid) {
                    _inc_ids[cursor[_f_i0[fid]]++] = fid;
                    if (_f_i1[fid] != _f_i0[fid])
                        _inc_ids[cursor[_f_i1[fid]]++] = fid;
                }
            }

            _events.init(F);
            _ev_cos.assign(F, NT(0));
            _ev_sin.assign(F, NT(0));
            _affected_mask.assign(F, 0);
            _affected_list.resize(F);
        }

        inline void initialize(Polytope const& P, Point const& p, RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            _p = p;
            refresh_velocity(n, rng);

            NT T = rng.sample_urdist() * _Len;
            advance_event_queue(P, T);
        }

        // Momentum refresh: N(0, I) for the spherical walk, per-coordinate
        // N(0, omega^2 / a_i) for the weighted diagonal walk.
        inline void refresh_velocity(unsigned int n, RandomNumberGenerator &rng)
        {
            _v = GetDirection<Point>::apply(n, rng, false);
            if (_weighted)
                _v.set_coeffs(_v.getCoefficients().cwiseProduct(_v_sigma));
        }

        inline bool advance_event_queue(Polytope const& P, NT T)
        {
            (void)P;
            VOLESTI_HMC_COUNT(n_trajectories);

            NT theta_rem = _omega * T;
            unsigned int it = 0;
            unsigned int reflections_since_rebase = 0;

            while (theta_rem > NT(0))
            {
                NT const theta_chunk = std::min(theta_rem, chunk_cap());
                begin_chunk(theta_chunk);

                for (;;)
                {
                    unsigned int fid;
                    if (!pop_next_valid_event(fid)) {
                        materialize_global_state(_end_cos, _end_sin);
                        theta_rem -= theta_chunk;
                        break;
                    }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
                    verify_event_against_full_scan(P, fid);
#endif

                    accept_event(fid);
                    VOLESTI_HMC_COUNT(n_reflections);
                    reflect_event(fid);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
                    verify_post_reflection_feasibility(fid);
#endif
                    ++it;
                    ++reflections_since_rebase;

                    if (it >= _rho) return false;

                    if (_rebase_interval > 0 &&
                        reflections_since_rebase >= _rebase_interval &&
                        _sin_rem > _angle_eps) {
                        // Renormalize alpha/beta from materialized state to
                        // curb floating-point drift on very long legs.  The
                        // one atan2 here runs once per _rebase_interval
                        // reflections.
                        VOLESTI_HMC_COUNT(n_rebase);
                        materialize_global_state(_cur_cos, _cur_sin);
                        theta_rem -= std::atan2(_cur_sin, _cur_cos);
                        reflections_since_rebase = 0;
                        break;
                    }

                    recompute_incident_facets(fid);
                }
            }

            return true;
        }

        // Start a chunk covering angles (0, theta_chunk] from the current
        // global state (_p, _v).  The single sincos call per chunk is the
        // only trigonometric evaluation on the non-exceptional path.
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
            _sin_rem = _end_sin;
            _one_minus_cos_rem = _key_end;

            _events.clear();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            for (unsigned int fid = 0; fid < F; ++fid)
                push_or_update_event(fid);
        }

        // Earliest outgoing hit of facet fid in the window
        // (theta_cur, theta_end], as a unit direction plus its ordering
        // key 1 - cos(theta*).
        inline bool next_event_for_facet(unsigned int fid, NT& key_out,
                                         NT& cos_out, NT& sin_out) const
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

            // Max-reach pruning: over the remaining angle De the value can
            // grow by at most |v|*(1 - cos(De)) + |v'|*sin(De)  (De <= pi/2).
            NT const val_cur = A * _cur_cos + B * _cur_sin;
            NT const slack = C - val_cur;
            if (slack < -veps) {
                // The state is on the outer side of this facet (a crossing
                // was skipped within the eps guard, e.g. at a corner).  Do
                // not schedule the re-entry crossing as a reflection: free
                // flight re-enters the polytope on its own, and the facet
                // is re-solved on the next incident reflection or chunk.
                // Rounding-level slack self-heals; anything larger is a real
                // containment violation and is counted, not hidden.
                if (slack < -hard_slack_tol() * scale) {
                    ++_hard_negative_slack;
                    VOLESTI_HMC_COUNT(n_hard_negative_slack);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
                    std::fprintf(stderr,
                        "[verify] hard negative slack: fid=%u wall=%d "
                        "slack=%.17g scale=%.17g cur=(%.17g, %.17g)\n",
                        fid, int(_f_wall[fid]), double(slack), double(scale),
                        double(_cur_cos), double(_cur_sin));
#endif
                }
                return false;
            }
            if (slack > NT(0)) {
                NT const dval_cur = B * _cur_cos - A * _cur_sin;
                NT const reach = std::abs(val_cur) * _one_minus_cos_rem
                               + std::abs(dval_cur) * _sin_rem;
                if (slack > reach)
                    return false;
            }

            NT const D = disc > NT(0) ? std::sqrt(disc) : NT(0);
            NT const inv_R2 = NT(1) / R2;
            // Of the two roots of A*cos + B*sin = C per turn, the value's
            // trajectory derivative B*cos - A*sin equals +D at
            //     (c, s) = ((A*C + B*D) / R^2, (B*C - A*D) / R^2)
            // and -D at the other root, so this root is always the outgoing
            // (inside -> outside) crossing and the other one the incoming.
            // Only an outgoing crossing reflects: the incoming root is never
            // scheduled, which also keeps the eps band around the boundary
            // from turning a re-entry into a spurious outward reflection.
            NT const c = (A * C + B * D) * inv_R2;
            NT const s = (B * C - A * D) * inv_R2;

            NT const cross = _cur_cos * s - _cur_sin * c;
            if (fid == _last_hit_fid) {
                // Only the just-reflected facet uses the wider eps guard, to
                // exclude its own numerical-residual root at the current
                // angle; for any other facet a root at or near the current
                // direction is a real crossing (see below) and skipping it
                // would leak the trajectory out of the polytope.
                if (cross <= _angle_eps) return false;
            } else if (cross <= NT(0)) {
                // The outgoing root coincides with the current direction up
                // to rounding: the second of two exactly simultaneous
                // crossings on disjoint facets at a corner, or a leg
                // starting on the boundary with outward velocity.  D > 0
                // certifies the trajectory genuinely leaves through the
                // facet; reflect at the current direction instead of
                // dropping the event.  A root behind by more than the eps
                // guard means the next outgoing crossing is a full turn
                // away, beyond any chunk.  The dot product tells a root at
                // the current angle (dot > 0) from the antipodal root half a
                // turn away, whose cross is also 0 (sin(pi) = 0) but which
                // lies far outside the window.
                if (cross <= -_angle_eps || D <= NT(0) ||
                    _cur_cos * c + _cur_sin * s <= NT(0)) return false;
                VOLESTI_HMC_COUNT(n_trig_hits);
                key_out = order_polytope_hmc_detail::one_minus_cos_stable(
                    _cur_cos, _cur_sin);
                cos_out = _cur_cos;
                sin_out = _cur_sin;
                return true;
            }

            // The root is inside the window iff its key does not exceed the
            // chunk end key.  Roots on the lower half circle always fail the
            // cross or key test because the window is within a quarter turn.
            NT const k = order_polytope_hmc_detail::one_minus_cos_stable(c, s);
            if (k > _key_end + key_tol()) return false;

            VOLESTI_HMC_COUNT(n_trig_hits);
            key_out = k; cos_out = c; sin_out = s;
            return true;
        }

        inline void push_or_update_event(unsigned int fid)
        {
            NT key, c, s;
            if (next_event_for_facet(fid, key, c, s)) {
                _ev_cos[fid] = c;
                _ev_sin[fid] = s;
                _events.insert_or_update(fid, key);
            } else {
                _events.remove(fid);
            }
        }

        inline bool pop_next_valid_event(unsigned int& fid_out)
        {
            unsigned int fid;
            NT key;
            while (_events.extract_min_leq(_key_end + key_tol(), fid, key)) {
                // Events ahead of the current direction are real crossings —
                // including near-simultaneous hits of disjoint facets at
                // corners — and must be reflected, or the trajectory leaks
                // through the second facet.
                NT const cross = _cur_cos * _ev_sin[fid] - _cur_sin * _ev_cos[fid];
                if (cross > NT(0)) {
                    fid_out = fid;
                    return true;
                }
                // The stored root is at or behind the current direction:
                // either a residual of FP-rounded coincident roots (stale)
                // or an exactly simultaneous crossing of a disjoint facet
                // whose entry predates the reflection that advanced the
                // position onto it.  Re-solve from the current state to
                // decide: a genuine simultaneous crossing comes back snapped
                // to the current direction and is reflected now; a future
                // root is re-queued; anything else is stale and dropped.
                VOLESTI_HMC_COUNT(n_stale_purge);
                NT k2, c2, s2;
                if (fid != _last_hit_fid && next_event_for_facet(fid, k2, c2, s2)) {
                    _ev_cos[fid] = c2;
                    _ev_sin[fid] = s2;
                    NT const cross2 = _cur_cos * s2 - _cur_sin * c2;
                    if (cross2 > NT(0)) {
                        _events.insert_or_update(fid, k2);
                        continue;
                    }
                    fid_out = fid;   // snapped to the current direction
                    return true;
                }
            }
            return false;
        }

        // Advance the current position to the popped event's direction.
        inline void accept_event(unsigned int fid)
        {
            _last_hit_fid = fid;
            _cur_cos = _ev_cos[fid];
            _cur_sin = _ev_sin[fid];
            _sin_rem = _end_sin * _cur_cos - _end_cos * _cur_sin;
            _one_minus_cos_rem = order_polytope_hmc_detail::one_minus_cos_stable(
                _end_cos * _cur_cos + _end_sin * _cur_sin, _sin_rem);
        }

        inline void reflect_event(unsigned int fid)
        {
            NT const cosV = _cur_cos;
            NT const sinV = _cur_sin;
            unsigned int const u = _f_i0[fid];

            if (_f_wall[fid]) {
                // Reflect: flip the velocity of coordinate u.
                NT const x = _alpha(u) * cosV + _beta(u) * sinV;
                NT const y = -_alpha(u) * sinV + _beta(u) * cosV; // v/omega
                _alpha(u) = x * cosV + y * sinV;
                _beta(u)  = x * sinV - y * cosV;
            } else if (!_weighted) {
                // Reflect on x_u <= x_w: swap the velocities of u and w.
                unsigned int const w = _f_i1[fid];
                NT const xu = _alpha(u) * cosV + _beta(u) * sinV;
                NT const xw = _alpha(w) * cosV + _beta(w) * sinV;
                NT const yu = -_alpha(u) * sinV + _beta(u) * cosV;
                NT const yw = -_alpha(w) * sinV + _beta(w) * cosV;
                _alpha(u) = xu * cosV - yw * sinV;
                _beta(u)  = xu * sinV + yw * cosV;
                _alpha(w) = xw * cosV - yu * sinV;
                _beta(w)  = xw * sinV + yu * cosV;
            } else {
                // Weighted diagonal cover reflection (the metric reflection
                // for M = diag(a)/omega^2): only u and w change, so the
                // incident-only invalidation stays exact.
                unsigned int const w = _f_i1[fid];
                NT const xu = _alpha(u) * cosV + _beta(u) * sinV;
                NT const xw = _alpha(w) * cosV + _beta(w) * sinV;
                NT const yu = -_alpha(u) * sinV + _beta(u) * cosV;
                NT const yw = -_alpha(w) * sinV + _beta(w) * cosV;
                NT const d = yu - yw;
                NT const ynu = yu - _f_ru[fid] * d;
                NT const ynw = yw + _f_rv[fid] * d;
                _alpha(u) = xu * cosV - ynu * sinV;
                _beta(u)  = xu * sinV + ynu * cosV;
                _alpha(w) = xw * cosV - ynw * sinV;
                _beta(w)  = xw * sinV + ynw * cosV;
            }
        }

        inline void recompute_incident_facets(unsigned int hit_fid)
        {
            unsigned int count = 0;

            auto mark_vertex = [&](unsigned int vertex) {
                for (unsigned int idx = _inc_off[vertex]; idx < _inc_off[vertex + 1]; ++idx) {
                    unsigned int const fid = _inc_ids[idx];
                    if (!_affected_mask[fid]) {
                        _affected_mask[fid] = 1;
                        _affected_list[count++] = fid;
                    }
                }
            };

            mark_vertex(_f_i0[hit_fid]);
            if (_f_i1[hit_fid] != _f_i0[hit_fid])
                mark_vertex(_f_i1[hit_fid]);

            for (unsigned int j = 0; j < count; ++j) {
                unsigned int const fid = _affected_list[j];
                _affected_mask[fid] = 0;
                if (fid == hit_fid && _f_C[fid] == NT(0)) {
                    // After reflecting on A*cos + B*sin = 0 the facet's own
                    // next root is exactly half a turn away — always beyond
                    // a quarter-period chunk.
                    _events.remove(fid);
                } else {
                    push_or_update_event(fid);
                }
            }
        }

        inline void materialize_global_state(NT c, NT s)
        {
            _p.set_coeffs(_center + _alpha * c + _beta * s);
            _v.set_coeffs((_beta * c - _alpha * s) * _omega);
        }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        // Cross-check the popped event against its own invariants and the
        // full-scan oracle, from the pre-event state (runs before
        // accept_event).
        inline void verify_event_against_full_scan(Polytope const& P, unsigned int fid)
        {
            NT const ec = _ev_cos[fid];
            NT const es = _ev_sin[fid];
            NT const A = _f_s0[fid] * _alpha(_f_i0[fid]) + _f_s1[fid] * _alpha(_f_i1[fid]);
            NT const B = _f_s0[fid] * _beta(_f_i0[fid]) + _f_s1[fid] * _beta(_f_i1[fid]);
            NT const C = _f_C[fid];
            NT const scale = std::max(NT(1),
                std::max(std::abs(A) + std::abs(B), std::abs(C)));
            NT const ftol = (NT(1e-9) + _angle_eps) * scale;
            NT const cross = _cur_cos * es - _cur_sin * ec;
            NT const dval_ev = B * ec - A * es;

            auto fail = [&](char const* what) {
                std::fprintf(stderr,
                    "[verify] %s: fid=%u wall=%d C=%.17g ev=(%.17g, %.17g) "
                    "cur=(%.17g, %.17g) end=(%.17g, %.17g) last=%d\n",
                    what, fid, int(_f_wall[fid]), double(C),
                    double(ec), double(es),
                    double(_cur_cos), double(_cur_sin),
                    double(_end_cos), double(_end_sin),
                    _last_hit_fid == no_facet() ? -1 : int(_last_hit_fid));
                assert(false && "order-polytope event-queue verify failed");
            };

            // The event direction lies on its own facet ...
            if (!(std::abs(A * ec + B * es - C) <= ftol))
                fail("event off its facet");
            // ... going outward (the incoming root never reflects) ...
            if (!(dval_ev >= -ftol))
                fail("event not an outgoing crossing");
            // ... forward within the current chunk (a snapped simultaneous
            // crossing sits exactly at the current direction, cross == 0) ...
            if (!(cross >= NT(0)))
                fail("event behind the current direction");
            if (!(order_polytope_hmc_detail::one_minus_cos_stable(ec, es)
                    <= _key_end + NT(2) * key_tol()))
                fail("event beyond the chunk end");
            // ... and respecting the last-hit-only suppression rule: the
            // guarded facet's own residual root must never come back.
            if (fid == _last_hit_fid && !(cross > _angle_eps))
                fail("residual root of the last-hit facet popped");

            NT const dt = std::atan2(cross, _cur_cos * ec + _cur_sin * es)
                        * _inv_omega;

            if (dt <= time_eps()) {
                // A simultaneous crossing processed at the current angle:
                // the full-scan oracle cannot represent t == 0 hits (it
                // demands strictly positive roots), so the event stands on
                // the direct checks above; require outward motion (up to
                // rounding of a near-tangent contact) instead of the oracle
                // comparison.
                if (!(dval_ev > -ftol))
                    fail("zero-time event without outward motion");
                return;
            }

            if (_has_center) {
                // The full-scan oracle models the trajectory as
                // r cos + v/omega sin around the origin; with a nonzero
                // Gaussian center folded into the offsets it would solve a
                // different trajectory, so the event stands on the direct
                // checks above (post-reflection feasibility still runs).
                return;
            }

            Point p_tmp(static_cast<unsigned int>(_alpha.rows()));
            Point v_tmp(static_cast<unsigned int>(_alpha.rows()));
            p_tmp.set_coeffs(_alpha * _cur_cos + _beta * _cur_sin);
            v_tmp.set_coeffs((_beta * _cur_cos - _alpha * _cur_sin) * _omega);

            int prev = -1;
            auto full = P.trigonometric_positive_intersect(p_tmp, v_tmp, _omega, prev);

            // The pre-event state lies exactly on the last reflected facet,
            // whose numerical-residual root the oracle reports as a spurious
            // near-zero hit (it is called with facet_prev == -1).  Re-query
            // with that facet marked; near-zero hits of *other* facets are
            // real crossings and must match the popped event.
            if (full.first <= time_eps() && full.second >= 0 &&
                _last_hit_fid != no_facet() &&
                static_cast<unsigned int>(full.second) == _f_row[_last_hit_fid]) {
                prev = full.second;
                full = P.trigonometric_positive_intersect(p_tmp, v_tmp, _omega, prev);
            }

            // With a popped event in hand, an oracle "no hit" is a real
            // discrepancy, not an automatic pass.  The only remaining
            // exception is a residual graze of a facet the chunk end
            // happened to land on exactly (not the last reflected one),
            // which the oracle reports at ~0 while the event is later.
            bool const residual_graze = full.first <= NT(1e-13);
            bool const ok = residual_graze ||
                            (full.first < std::numeric_limits<NT>::max() &&
                             std::abs(full.first - dt) <= NT(1e-7));
            if (!ok) {
                std::fprintf(stderr,
                    "[verify] fid=%u wall=%d C=%.17g dt=%.17g full=(%.17g, facet %d) "
                    "cur=(%.17g, %.17g) end=(%.17g, %.17g)\n",
                    fid, int(_f_wall[fid]), double(_f_C[fid]), double(dt),
                    double(full.first), full.second,
                    double(_cur_cos), double(_cur_sin),
                    double(_end_cos), double(_end_sin));
            }
            assert(ok);
        }

        // After a reflection the state must stay feasible: every facet value
        // at the current direction remains below its offset up to tolerance.
        // Runs against the updated alpha/beta, so it catches both a bad
        // reflection and a missed simultaneous crossing (the leaked facet
        // shows up immediately, instead of only corrupting the samples).
        inline void verify_post_reflection_feasibility(unsigned int hit_fid) const
        {
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            for (unsigned int fid = 0; fid < F; ++fid) {
                NT const A = _f_s0[fid] * _alpha(_f_i0[fid]) + _f_s1[fid] * _alpha(_f_i1[fid]);
                NT const B = _f_s0[fid] * _beta(_f_i0[fid]) + _f_s1[fid] * _beta(_f_i1[fid]);
                NT const C = _f_C[fid];
                NT const scale = std::max(NT(1),
                    std::max(std::abs(A) + std::abs(B), std::abs(C)));
                NT const val = A * _cur_cos + B * _cur_sin;
                bool const feasible = val - C <= (NT(1e-9) + _angle_eps) * scale;
                if (!feasible) {
                    std::fprintf(stderr,
                        "[verify] infeasible after reflection: hit=%u fid=%u "
                        "wall=%d val=%.17g C=%.17g cur=(%.17g, %.17g)\n",
                        hit_fid, fid, int(_f_wall[fid]), double(val), double(C),
                        double(_cur_cos), double(_cur_sin));
                    assert(feasible && "state left the polytope after a reflection");
                }
            }
        }
#endif

        static inline unsigned int no_facet() { return std::numeric_limits<unsigned int>::max(); }

        // Quarter period in angle units.
        static inline NT chunk_cap() { return NT(1.57079632679489661923); }

        inline NT time_eps() const { return NT(1e-10); }
        inline NT key_tol() const { return NT(1e-12); }
        inline NT value_tol() const { return NT(1e-12); }
        // Negative slack beyond this (times scale) at solve time is a real
        // containment violation, not an eps-band graze that self-heals.
        inline NT hard_slack_tol() const { return NT(1e-7); }

        unsigned int _rho;
        NT _Len;
        Point _p, _v;
        NT _omega, _inv_omega, _angle_eps;
        VT _alpha, _beta;
        NT _end_cos, _end_sin, _key_end, _cur_cos, _cur_sin, _sin_rem, _one_minus_cos_rem;
        unsigned int _rebase_interval;
        // Facet whose reflection produced the current position; its residual
        // root at the current angle is excluded by the eps guard.  Persists
        // across chunk boundaries within one continuous trajectory (the
        // state can sit exactly on it), but is cleared on every momentum
        // refresh: with a fresh velocity a root at the current angle is a
        // real crossing whenever the motion is outward.
        unsigned int _last_hit_fid;
        // Diagnostic: solves that found the state beyond hard_slack_tol()
        // outside a facet (see hard_negative_slack_count).
        mutable unsigned long long _hard_negative_slack;
        // Matched-diagonal extension (the defaults — _weighted == false,
        // zero center — reproduce the spherical walk bit for bit).
        bool _weighted, _has_center;
        VT _center, _a_diag, _v_sigma;

        // Facet data, structure-of-arrays: value along the trajectory is
        // v(theta) = s0*q(i0) + s1*q(i1) for q in {alpha, beta}.
        std::vector<unsigned int> _f_i0, _f_i1;
        std::vector<NT> _f_s0, _f_s1, _f_C;
        std::vector<unsigned char> _f_wall;
        std::vector<unsigned int> _f_row;
        // Weighted cover-reflection coefficients (1 for walls / spherical).
        std::vector<NT> _f_ru, _f_rv;

        // CSR lists of facets incident to each coordinate.
        std::vector<unsigned int> _inc_off, _inc_ids;

        // Pending events: unit direction per facet plus the indexed heap.
        std::vector<NT> _ev_cos, _ev_sin;
        IndexedDHeap _events;
        std::vector<unsigned char> _affected_mask;
        std::vector<unsigned int> _affected_list;
    };

    parameters param;
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
