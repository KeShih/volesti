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
#include "root_finders/trigonometric_equation_solvers.hpp"
#endif

#include "random_walks/compute_diameter.hpp"
#include "random_walks/gaussian_hamiltonian_monte_carlo_exact_walk.hpp"

#ifdef VOLESTI_HMC_PROFILE
#include <atomic>
namespace hmc_profile_counters {
inline std::atomic<unsigned long long> n_trajectories{0}, n_reflections{0};
inline std::atomic<unsigned long long> n_stale_purge{0}, n_trig_calls{0}, n_trig_hits{0};
inline std::atomic<unsigned long long> n_heap_insert{0}, n_heap_remove{0}, n_rebase{0};
inline void bump(std::atomic<unsigned long long>& c)
{
    c.fetch_add(1, std::memory_order_relaxed);
}
}
#define VOLESTI_HMC_COUNT(counter) hmc_profile_counters::bump(hmc_profile_counters::counter)
#else
#define VOLESTI_HMC_COUNT(counter) ((void)0)
#endif

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

        enum class trajectory_status {
            success,
            reflection_limit,
            shared_coordinate_contact,
            numerical_failure
        };

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng)
            : Walk(P, p, a_i, rng, parameters(0, false, 0, false))
        {}

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng,
             parameters const& params)
        {
            _omega = std::sqrt(NT(2) * a_i);
            _inv_omega = NT(1) / _omega;
            _angle_eps = _omega * time_eps();
            _Len = params.set_L ? params.m_L : default_trajectory_length(P);
            _rho = params.set_rho ? params.rho : 100 * P.dimension();
            _rebase_interval = params.rebase_interval;
            _last_status = trajectory_status::success;
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
                _v = GetDirection<Point>::apply(n, rng, false);
                Point p0 = _p;
                _saved_last_hit_facets.assign(_last_hit_facets.begin(),
                                              _last_hit_facets.end());

                if (!advance_event_queue(P, T)) {
                    // Restore all persistent state changed by this leg.  The
                    // next leg fully refreshes velocity and rebuilds its
                    // event queue, but its residual-root guard must still
                    // describe the restored position.
                    _p = p0;
                    set_last_hit_facets(_saved_last_hit_facets);
                    if (_last_status != trajectory_status::reflection_limit)
                        break;
                }
            }
            p = _p;
        }

        inline void update_delta(NT L) { _Len = L; }

        inline trajectory_status last_trajectory_status() const { return _last_status; }

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

            bool min_key(NT& key_out) const
            {
                if (_h.empty()) return false;
                key_out = _h[0].key;
                return true;
            }

            inline bool contains(unsigned int id) const
            {
                return _pos[id] >= 0;
            }

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
            _f_C.clear(); _f_wall.clear();

            auto add_facet = [&](unsigned int i0, unsigned int i1, NT s0, NT s1,
                                 NT C, bool wall) {
                _f_i0.push_back(i0); _f_i1.push_back(i1);
                _f_s0.push_back(s0); _f_s1.push_back(s1); _f_C.push_back(C);
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
            _contact_batch.reserve(F);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
            _verify_facets.reserve(F);
#endif
            _last_hit_mask.assign(F, 0);
            _last_hit_facets.reserve(F);
            _saved_last_hit_facets.reserve(F);
            _affected_mask.assign(F, 0);
            _affected_list.resize(F);
        }

        inline void initialize(Polytope const& P, Point const& p, RandomNumberGenerator &rng)
        {
            unsigned int n = P.dimension();
            _p = p;
            _v = GetDirection<Point>::apply(n, rng, false);

            NT T = rng.sample_urdist() * _Len;
            Point const p0 = _p;
            _saved_last_hit_facets.assign(_last_hit_facets.begin(),
                                          _last_hit_facets.end());
            if (!advance_event_queue(P, T)) {
                _p = p0;
                set_last_hit_facets(_saved_last_hit_facets);
            }
        }

        inline bool advance_event_queue(Polytope const& P, NT T)
        {
            (void)P;
            VOLESTI_HMC_COUNT(n_trajectories);
            _last_status = trajectory_status::success;

            if (!std::isfinite(T) || T < NT(0) ||
                !std::isfinite(_omega) || _omega <= NT(0) ||
                !std::isfinite(_inv_omega) ||
                !_p.getCoefficients().allFinite() ||
                !_v.getCoefficients().allFinite()) {
                _last_status = trajectory_status::numerical_failure;
                return false;
            }

            NT theta_rem = _omega * T;
            unsigned int it = 0;
            unsigned int reflections_since_rebase = 0;

            while (theta_rem > NT(0))
            {
                NT const theta_chunk = std::min(theta_rem, chunk_cap());
                if (!begin_chunk(theta_chunk)) {
                    _last_status = trajectory_status::numerical_failure;
                    return false;
                }

                for (;;)
                {
                    bool has_event = false;
                    if (!pop_next_contact_batch(has_event)) {
                        _last_status = trajectory_status::numerical_failure;
                        return false;
                    }
                    if (!has_event) {
                        materialize_global_state(_end_cos, _end_sin);
                        theta_rem -= theta_chunk;
                        break;
                    }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
                    verify_contact_batch_against_full_scan(P);
#endif

                    // Batch membership is established against the same
                    // pre-reflection state.  Identity-mass reflections on
                    // facets with disjoint coordinate supports commute.  A
                    // shared-coordinate contact is deliberately unresolved:
                    // abort the leg so apply()/initialize() can roll back.
                    if (_contact_batch.size() > 1 && !contact_batch_is_disjoint()) {
                        _last_status = trajectory_status::shared_coordinate_contact;
                        return false;
                    }

                    if (_contact_batch.size() == 1) {
                        unsigned int const fid = _contact_batch.front();
                        accept_event(fid);
                        VOLESTI_HMC_COUNT(n_reflections);
                        reflect_event(fid);
                        ++it;
                        ++reflections_since_rebase;
                    } else {
                        accept_contact_batch();
                        for (unsigned int const fid : _contact_batch) {
                            VOLESTI_HMC_COUNT(n_reflections);
                            reflect_event(fid);
                            ++it;
                            ++reflections_since_rebase;
                        }
                    }

                    if (it >= _rho) {
                        _last_status = trajectory_status::reflection_limit;
                        return false;
                    }

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

                    if (_contact_batch.size() == 1)
                        recompute_incident_facets(_contact_batch.front());
                    else
                        recompute_incident_facets_batch();

                    if (_event_oracle_failed) {
                        _last_status = trajectory_status::numerical_failure;
                        return false;
                    }
                }
            }

            return true;
        }

        // Start a chunk covering angles (0, theta_chunk] from the current
        // global state (_p, _v).  The single sincos call per chunk is the
        // only trigonometric evaluation on the non-exceptional path.
        inline bool begin_chunk(NT theta_chunk)
        {
            _alpha = _p.getCoefficients();
            _beta  = _v.getCoefficients() * _inv_omega;
            _end_cos = std::cos(theta_chunk);
            _end_sin = std::sin(theta_chunk);
            _key_end = NT(1) - _end_cos;
            _cur_cos = NT(1);
            _cur_sin = NT(0);
            _sin_rem = _end_sin;
            _one_minus_cos_rem = _key_end;

            _event_oracle_failed = false;
            _events.clear();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            for (unsigned int fid = 0; fid < F; ++fid) {
                push_or_update_event(fid);
                if (_event_oracle_failed) return false;
            }
            return true;
        }

        // Earliest hit of facet fid in the window (theta_cur, theta_end],
        // as a unit direction plus its ordering key 1 - cos(theta*).
        inline bool next_event_for_facet(unsigned int fid, NT& key_out,
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

            // Max-reach pruning: over the remaining angle De the value can
            // grow by at most |v|*(1 - cos(De)) + |v'|*sin(De)  (De <= pi/2).
            NT const val_cur = A * _cur_cos + B * _cur_sin;
            NT const slack = C - val_cur;
            if (slack < -veps) {
                // Reaching the outer side means an earlier contact was not
                // resolved.  Never convert its later re-entry into a valid
                // trajectory: propagate a numerical failure so the entire
                // leg is rolled back to its last feasible state.
                _event_oracle_failed = true;
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
            NT const AC = A * C, BC = B * C, AD = A * D, BD = B * D;
            NT const c1 = (AC + BD) * inv_R2, s1 = (BC - AD) * inv_R2;
            NT const c2 = (AC - BD) * inv_R2, s2 = (BC + AD) * inv_R2;

            // A root is inside the window iff it lies strictly ahead of the
            // current direction (cross > 0) and its key does not exceed the
            // chunk end key.  Roots on the lower half circle always fail one
            // of the two tests because the window is within a quarter turn.
            // Only the just-reflected facet uses the wider eps guard, to
            // exclude its own numerical-residual root at the current angle;
            // for all other facets a nearby forward root is a real crossing
            // and skipping it would leak the trajectory out of the polytope.
            NT const fwd_eps = is_last_hit_facet(fid) ? _angle_eps : NT(0);
            NT const limit = _key_end + key_tol();
            bool found = false;
            NT best_key = NT(0), best_cos = NT(0), best_sin = NT(0);

            auto consider = [&](NT c, NT s) {
                NT const cross = _cur_cos * s - _cur_sin * c;
                if (cross > fwd_eps) {
                    NT const k = NT(1) - c;
                    if (k <= limit && (!found || k < best_key)) {
                        found = true;
                        best_key = k; best_cos = c; best_sin = s;
                    }
                }
            };
            consider(c1, s1);
            consider(c2, s2);

            if (!found) return false;

            VOLESTI_HMC_COUNT(n_trig_hits);
            key_out = best_key; cos_out = best_cos; sin_out = best_sin;
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

        inline bool pop_next_contact_batch(bool& has_event)
        {
            unsigned int fid;
            NT key;
            has_event = false;
            bool found = false;
            while (_events.extract_min_leq(_key_end + key_tol(), fid, key)) {
                // Every exact pre-reflection tie should have been removed as
                // one batch.  A remaining event at or behind the current
                // direction is therefore unresolved numerical state; reject
                // the leg instead of silently skipping a possible contact.
                NT const cross = _cur_cos * _ev_sin[fid] - _cur_sin * _ev_cos[fid];
                if (cross <= NT(0)) {
                    VOLESTI_HMC_COUNT(n_stale_purge);
                    return false;
                }
                found = true;
                break;
            }

            if (!found) return true;

            has_event = true;
            _contact_batch.clear();
            _contact_batch.push_back(fid);

            NT peek_key;
            if (!_events.min_key(peek_key) || peek_key > key + key_tol())
                return true;

            NT const first_cos = _ev_cos[fid];
            NT const first_sin = _ev_sin[fid];
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            for (unsigned int candidate = 0; candidate < F; ++candidate) {
                if (!_events.contains(candidate)) continue;
                NT const tie_cross = first_cos * _ev_sin[candidate]
                                   - first_sin * _ev_cos[candidate];
                NT const tie_dot = first_cos * _ev_cos[candidate]
                                 + first_sin * _ev_sin[candidate];
                if (std::abs(tie_cross) <= contact_angle_tol() &&
                    tie_dot >= NT(1) - contact_angle_tol()) {
                    _events.remove(candidate);
                    _contact_batch.push_back(candidate);
                }
            }
            return true;
        }

        inline bool contact_batch_is_disjoint() const
        {
            for (unsigned int i = 0; i < _contact_batch.size(); ++i) {
                unsigned int const fi = _contact_batch[i];
                unsigned int const i0 = _f_i0[fi], i1 = _f_i1[fi];
                for (unsigned int j = i + 1; j < _contact_batch.size(); ++j) {
                    unsigned int const fj = _contact_batch[j];
                    unsigned int const j0 = _f_i0[fj], j1 = _f_i1[fj];
                    if (i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1)
                        return false;
                }
            }
            return true;
        }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        // Correctness reference: scan every structured facet independently
        // of heap state/incident updates.  It deliberately uses the atan2
        // trigonometric solver rather than the queue's algebraic direction
        // roots, then returns the complete earliest pre-reflection set.
        inline bool full_scan_contact_batch(std::vector<unsigned int>& facets,
                                            NT& cos_out, NT& sin_out) const
        {
            facets.clear();
            bool found = false;
            NT best_time = std::numeric_limits<NT>::max();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            static NT const pi = std::acos(NT(-1));
            NT const period = NT(2) * pi * _inv_omega;
            NT const current_angle = std::atan2(_cur_sin, _cur_cos);
            NT const end_angle = std::atan2(_end_sin, _end_cos);
            NT const current_time = current_angle * _inv_omega;
            NT const end_time = end_angle * _inv_omega;

            for (unsigned int fid = 0; fid < F; ++fid) {
                NT const A = _f_s0[fid] * _alpha(_f_i0[fid])
                           + _f_s1[fid] * _alpha(_f_i1[fid]);
                NT const B = _f_s0[fid] * _beta(_f_i0[fid])
                           + _f_s1[fid] * _beta(_f_i1[fid]);
                NT const min_time = current_time +
                    (is_last_hit_facet(fid) ? time_eps() : NT(0));
                auto const root = first_trigonometric_solution(
                    A, B, _f_C[fid], _inv_omega, period, min_time,
                    NT(1e-12), value_tol());
                if (!root.second || !(root.first > current_time) ||
                    root.first > end_time + NT(1e-12))
                    continue;

                NT const angle = _omega * root.first;
                NT const c = std::cos(angle), s = std::sin(angle);
                bool tied = false;
                if (found) {
                    NT const tie_cross = cos_out * s - sin_out * c;
                    NT const tie_dot = cos_out * c + sin_out * s;
                    tied = std::abs(tie_cross) <= contact_angle_tol() &&
                           tie_dot >= NT(1) - contact_angle_tol();
                }
                if (!found || (!tied && root.first < best_time)) {
                    found = true;
                    best_time = root.first;
                    facets.clear();
                    facets.push_back(fid);
                    cos_out = c;
                    sin_out = s;
                } else if (tied) {
                    facets.push_back(fid);
                }
            }

            if (!found) return false;
            std::sort(facets.begin(), facets.end());
            return true;
        }
#endif

        // Advance once to the common contact before reflecting the batch.
        inline void accept_event(unsigned int fid)
        {
            set_last_hit_facet(fid);
            _cur_cos = _ev_cos[fid];
            _cur_sin = _ev_sin[fid];
            _sin_rem = _end_sin * _cur_cos - _end_cos * _cur_sin;
            _one_minus_cos_rem = NT(1) - (_end_cos * _cur_cos + _end_sin * _cur_sin);
        }

        inline void accept_contact_batch()
        {
            unsigned int const fid = _contact_batch.front();
            set_last_hit_facets(_contact_batch);
            _cur_cos = _ev_cos[fid];
            _cur_sin = _ev_sin[fid];
            _sin_rem = _end_sin * _cur_cos - _end_cos * _cur_sin;
            _one_minus_cos_rem = NT(1) - (_end_cos * _cur_cos + _end_sin * _cur_sin);
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
            } else {
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
            }
        }

        inline void recompute_incident_facets_batch()
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

            for (unsigned int const hit_fid : _contact_batch) {
                mark_vertex(_f_i0[hit_fid]);
                if (_f_i1[hit_fid] != _f_i0[hit_fid])
                    mark_vertex(_f_i1[hit_fid]);
            }

            for (unsigned int j = 0; j < count; ++j) {
                unsigned int const fid = _affected_list[j];
                _affected_mask[fid] = 0;
                bool was_hit = false;
                for (unsigned int const hit_fid : _contact_batch) {
                    if (fid == hit_fid) {
                        was_hit = true;
                        break;
                    }
                }
                if (was_hit && _f_C[fid] == NT(0))
                    _events.remove(fid);
                else
                    push_or_update_event(fid);
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
                if (fid == hit_fid && _f_C[fid] == NT(0))
                    _events.remove(fid);
                else
                    push_or_update_event(fid);
            }
        }

        inline void materialize_global_state(NT c, NT s)
        {
            _p.set_coeffs(_alpha * c + _beta * s);
            _v.set_coeffs((_beta * c - _alpha * s) * _omega);
        }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        // Cross-check the complete earliest contact set against a full scan
        // from the same pre-reflection state.
        inline void verify_contact_batch_against_full_scan(Polytope const& P)
        {
            (void)P;
            NT full_cos = NT(0), full_sin = NT(0);
            bool const found = full_scan_contact_batch(
                _verify_facets, full_cos, full_sin);

            std::sort(_contact_batch.begin(), _contact_batch.end());
            bool const rows_ok = found && _contact_batch == _verify_facets;
            unsigned int const first = _contact_batch.front();
            NT const direction_cross = full_cos * _ev_sin[first]
                                     - full_sin * _ev_cos[first];
            NT const direction_dot = full_cos * _ev_cos[first]
                                   + full_sin * _ev_sin[first];
            bool const direction_ok = found &&
                std::abs(direction_cross) <= contact_angle_tol() &&
                direction_dot >= NT(1) - contact_angle_tol();
            bool const ok = rows_ok && direction_ok;
            if (!ok) {
                std::fprintf(stderr,
                    "[verify] contacts=%zu full_contacts=%zu found=%d "
                    "dir_cross=%.17g dir_dot=%.17g cur=(%.17g, %.17g) "
                    "end=(%.17g, %.17g)\n",
                    _contact_batch.size(), _verify_facets.size(), int(found),
                    double(direction_cross), double(direction_dot),
                    double(_cur_cos), double(_cur_sin),
                    double(_end_cos), double(_end_sin));
            }
            assert(ok);
        }
#endif

        inline bool is_last_hit_facet(unsigned int fid) const
        {
            return !_last_hit_mask.empty() && _last_hit_mask[fid] != 0;
        }

        inline void set_last_hit_facets(std::vector<unsigned int> const& facets)
        {
            for (unsigned int const fid : _last_hit_facets)
                _last_hit_mask[fid] = 0;
            _last_hit_facets.assign(facets.begin(), facets.end());
            for (unsigned int const fid : _last_hit_facets)
                _last_hit_mask[fid] = 1;
        }

        inline void set_last_hit_facet(unsigned int fid)
        {
            for (unsigned int const previous : _last_hit_facets)
                _last_hit_mask[previous] = 0;
            _last_hit_facets.clear();
            _last_hit_facets.push_back(fid);
            _last_hit_mask[fid] = 1;
        }

        // Quarter period in angle units.
        static inline NT chunk_cap() { return NT(1.57079632679489661923); }

        inline NT time_eps() const { return NT(1e-10); }
        inline NT key_tol() const { return NT(1e-12); }
        inline NT value_tol() const { return NT(1e-12); }
        inline NT contact_angle_tol() const {
            return NT(64) * std::numeric_limits<NT>::epsilon();
        }
        unsigned int _rho;
        NT _Len;
        Point _p, _v;
        NT _omega, _inv_omega, _angle_eps;
        VT _alpha, _beta;
        NT _end_cos, _end_sin, _key_end, _cur_cos, _cur_sin, _sin_rem, _one_minus_cos_rem;
        unsigned int _rebase_interval;
        trajectory_status _last_status;
        bool _event_oracle_failed = false;
        // Facets whose reflection produced the current position.  Their
        // residual roots at the current angle are excluded by the eps guard.
        std::vector<unsigned char> _last_hit_mask;
        std::vector<unsigned int> _last_hit_facets;
        std::vector<unsigned int> _saved_last_hit_facets;

        // Facet data, structure-of-arrays: value along the trajectory is
        // v(theta) = s0*q(i0) + s1*q(i1) for q in {alpha, beta}.
        std::vector<unsigned int> _f_i0, _f_i1;
        std::vector<NT> _f_s0, _f_s1, _f_C;
        std::vector<unsigned char> _f_wall;

        // CSR lists of facets incident to each coordinate.
        std::vector<unsigned int> _inc_off, _inc_ids;

        // Pending events: unit direction per facet plus the indexed heap.
        std::vector<NT> _ev_cos, _ev_sin;
        IndexedDHeap _events;
        std::vector<unsigned int> _contact_batch;
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        std::vector<unsigned int> _verify_facets;
#endif
        std::vector<unsigned char> _affected_mask;
        std::vector<unsigned int> _affected_list;
    };

    parameters param;
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
