// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
#include <cstdio>
#endif

#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "sampling/sphere.hpp"

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

namespace order_polytope_exact_hmc_detail {

struct SphericalDynamics
{
    static constexpr bool validate_endpoint_feasibility = false;
    static constexpr bool fail_on_nonfinite_event_arithmetic = false;
    static constexpr bool normalize_event_rows = false;
    static constexpr bool stable_event_keys = false;
    static constexpr bool reset_last_hit_on_velocity_refresh = false;

    template <typename Polytope>
    struct State
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;

        explicit State(Polytope const&) {}

        inline bool body_is_compatible(Polytope const&) const { return true; }

        template <typename RandomNumberGenerator>
        inline Point draw_velocity(unsigned int n, RandomNumberGenerator& rng) const
        {
            // Preserve the established spherical refresh exactly.
            return GetDirection<Point>::apply(n, rng, false);
        }

        inline void reflect_cover_eta(unsigned int, unsigned int,
                                      NT& eta_u, NT& eta_v) const
        {
            std::swap(eta_u, eta_v);
        }

        inline void count_trajectory() const {}
        inline void count_reflection() const {}
        inline void count_event_recomputation() const {}
        inline bool velocity_is_valid(Point const&) const { return true; }
        inline bool position_is_feasible(Point const&) const { return true; }
        template <typename Vector>
        inline bool modal_state_is_valid(Vector const&, Vector const&) const
        {
            return true;
        }
    };
};

struct DiagonalDynamics
{
    static constexpr bool validate_endpoint_feasibility = true;
    static constexpr bool fail_on_nonfinite_event_arithmetic = true;
    static constexpr bool normalize_event_rows = true;
    static constexpr bool stable_event_keys = true;
    static constexpr bool reset_last_hit_on_velocity_refresh = true;

    template <typename Polytope>
    struct State
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;

        explicit State(Polytope const& polytope)
            : _polytope(&polytope),
              _center(&polytope.rounding_center()),
              _shape_diag(&polytope.rounding_shape_diag()),
              _sqrt_shape_diag(&polytope.rounding_sqrt_shape_diag()),
              _diagnostics(polytope.rounding_diagnostics())
        {}

        inline bool body_is_compatible(Polytope const& polytope) const
        {
            return _polytope == &polytope;
        }

        template <typename RandomNumberGenerator>
        inline Point draw_velocity(unsigned int n, RandomNumberGenerator& rng) const
        {
            Point velocity = GetDirection<Point>::apply(n, rng, false);
            for (unsigned int i = 0; i < n; ++i)
                velocity.set_coord(i, velocity[i] * (*_sqrt_shape_diag)(i));
            return velocity;
        }

        inline void reflect_cover_eta(unsigned int u, unsigned int v,
                                      NT& eta_u, NT& eta_v) const
        {
            NT const d_u = (*_shape_diag)(u);
            NT const d_v = (*_shape_diag)(v);
            NT const denom = d_u + d_v;
            NT const delta = eta_u - eta_v;
            // The validated metric guarantees a finite positive denominator.
            // Form weights before multiplying delta to avoid needless overflow.
            eta_u -= NT(2) * (d_u / denom) * delta;
            eta_v += NT(2) * (d_v / denom) * delta;
        }

        inline void count_trajectory() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled) {
                if (_diagnostics->active_stage ==
                    OrderPolytopeDiagonalCoolingStage::Schedule)
                    ++_diagnostics->schedule_trajectories;
                else
                    ++_diagnostics->ratio_trajectories;
            }
        }

        inline void count_reflection() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++_diagnostics->reflection_count;
        }

        inline void count_event_recomputation() const
        {
            if constexpr (Polytope::rounding_diagnostics_enabled)
                ++_diagnostics->event_recomputation_count;
        }

        inline bool velocity_is_valid(Point const& velocity) const
        {
            return velocity.getCoefficients().allFinite();
        }

        inline bool position_is_feasible(Point const& centered) const
        {
            auto const& coordinates = centered.getCoefficients();
            unsigned int const n = _polytope->dimension();
            for (unsigned int i = 0; i < n; ++i) {
                NT const center_i = (*_center)(i);
                NT const coordinate_i = coordinates(i);
                if (!std::isfinite(center_i) ||
                    !std::isfinite(coordinate_i) ||
                    exact_sum_sign(center_i, coordinate_i) < 0 ||
                    exact_sum_sign(center_i, coordinate_i, NT(-1)) > 0)
                    return false;
            }
            for (unsigned int k = 0;
                 k < _polytope->num_order_relations(); ++k) {
                auto const relation = _polytope->get_order_relation(k);
                NT const center_u = (*_center)(relation.first);
                NT const coordinate_u = coordinates(relation.first);
                NT const center_v = (*_center)(relation.second);
                NT const coordinate_v = coordinates(relation.second);
                if (!std::isfinite(center_u) ||
                    !std::isfinite(coordinate_u) ||
                    !std::isfinite(center_v) ||
                    !std::isfinite(coordinate_v) ||
                    exact_sum_sign(center_u, coordinate_u,
                                   -center_v, -coordinate_v) > 0)
                    return false;
            }
            return true;
        }

        template <typename Vector>
        inline bool modal_state_is_valid(Vector const& alpha,
                                         Vector const& beta) const
        {
            return alpha.allFinite() && beta.allFinite();
        }

        inline NT rounding_center_coordinate(unsigned int i) const
        {
            return (*_center)(i);
        }

    private:
        // Sign of the exact sum of up to four floating-point inputs.  The
        // short nonoverlapping expansion prevents a final rounded addition
        // from hiding an ulp-scale bound or cover violation.
        static inline int exact_sum_sign(
            NT a, NT b, NT c = NT(0), NT d = NT(0))
        {
            NT const terms[4] = {a, b, c, d};
            NT expansion[4] = {NT(0), NT(0), NT(0), NT(0)};
            unsigned int expansion_size = 0;
            for (unsigned int term_index = 0; term_index < 4; ++term_index) {
                NT next[4] = {NT(0), NT(0), NT(0), NT(0)};
                unsigned int next_size = 0;
                NT q = terms[term_index];
                for (unsigned int i = 0; i < expansion_size; ++i) {
                    NT const sum = q + expansion[i];
                    NT const virtual_expansion = sum - q;
                    NT const error =
                        (q - (sum - virtual_expansion)) +
                        (expansion[i] - virtual_expansion);
                    if (error != NT(0)) next[next_size++] = error;
                    q = sum;
                }
                if (q != NT(0) || next_size == 0) next[next_size++] = q;
                expansion_size = next_size;
                for (unsigned int i = 0; i < next_size; ++i)
                    expansion[i] = next[i];
            }
            for (unsigned int i = expansion_size; i > 0; --i) {
                if (expansion[i - 1] > NT(0)) return 1;
                if (expansion[i - 1] < NT(0)) return -1;
            }
            return 0;
        }

        Polytope const* _polytope;
        VT const* _center;
        VT const* _shape_diag;
        VT const* _sqrt_shape_diag;
        OrderPolytopeDiagonalCoolingDiagnostics<NT>* _diagnostics;
    };
};

} // namespace order_polytope_exact_hmc_detail

template <typename Dynamics>
struct OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy
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

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy(double L, unsigned int _rho)
        : param(L, true, _rho, true) {}

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy(double L)
        : param(L, true, 0, false) {}

    OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy()
        : param(0, false, 0, false) {}

    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk : private Dynamics::template State<Polytope>
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;
        typedef typename Dynamics::template State<Polytope> DynamicsState;

        enum class trajectory_status {
            success,
            reflection_limit,
            shared_coordinate_contact,
            numerical_failure
        };

#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
        struct testing_trace {
            bool captured_contact = false;
            bool captured_post_reflection = false;
            bool full_scan_checked = false;
            bool full_scan_agreed = true;
            unsigned int full_scan_contact_count = 0;
            NT hit_angle = std::numeric_limits<NT>::quiet_NaN();
            NT max_full_scan_direction_error = NT(0);
            std::vector<unsigned int> contact_facets;
            Point collision_position, pre_reflection_velocity;
            Point post_reflection_velocity, final_position, final_velocity;
        };

        inline testing_trace const& get_testing_trace() const
        {
            return _testing_trace;
        }
#endif

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng)
            : Walk(P, p, a_i, rng, parameters(0, false, 0, false))
        {}

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng,
             parameters const& params)
            : DynamicsState(P)
        {
            _last_status = trajectory_status::success;
            _omega = std::sqrt(NT(2) * a_i);
            _inv_omega = NT(1) / _omega;
            if (!std::isfinite(_omega) || _omega <= NT(0) ||
                !std::isfinite(_inv_omega) ||
                !p.getCoefficients().allFinite()) {
                _last_status = trajectory_status::numerical_failure;
                throw_trajectory_failure(_last_status);
            }
            _angle_eps = _omega * time_eps();
            _Len = params.set_L ? params.m_L : default_trajectory_length(P);
            _rho = params.set_rho ? params.rho : 100 * P.dimension();
            _rebase_interval = params.rebase_interval;
            _last_hit_fid = no_facet();
            _saved_last_hit_fid = no_facet();
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
                _v = this->draw_velocity(n, rng);
                if constexpr (Dynamics::reset_last_hit_on_velocity_refresh) {
                    clear_last_hit_batch();
                    _last_hit_fid = no_facet();
                }
                Point p0 = _p;
                save_last_hit_state();

                _last_status = advance_event_queue(P, T);
                if (_last_status != trajectory_status::success) {
                    // Restore all persistent state changed by this leg.  The
                    // next leg fully refreshes velocity and rebuilds its
                    // event queue, but its residual-root guard must still
                    // describe the restored position.
                    _p = p0;
                    restore_last_hit_state();
                    if (_last_status != trajectory_status::reflection_limit) {
                        p = _p;
                        throw_trajectory_failure(_last_status);
                    }
                }
            }
            p = _p;
        }

        inline void update_delta(NT L) { _Len = L; }

        inline trajectory_status last_trajectory_status() const { return _last_status; }

    private:

        enum class contact_batch_result {
            none,
            singleton,
            disjoint_batch,
            shared_coordinate_contact,
            numerical_failure
        };

        // Indexed D-ary min-heap over pending facet events.  The heap only
        // ever contains facets with a pending hit inside the current chunk,
        // so its depth tracks the (small) active set, not the facet count.
        class IndexedDHeap
        {
            static constexpr int D = 8;
            struct Entry { NT key; unsigned int id; };
            std::vector<Entry> _h;
            std::vector<int> _pos;

            int par(int i) const { return (i - 1) / D; }
            int child0(int i) const { return D * i + 1; }

            // Hole-based sifts: carry the displaced entry in a register and
            // shift blockers down/up, one position write per level.  Keys
            // live inline in the heap array, so each contiguous child group
            // is scanned without gathers through a side table.
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

            template <typename Visitor>
            inline void for_each_key_leq(NT limit, Visitor const& visitor) const
            {
                for (Entry const& event : _h)
                    if (event.key <= limit) visitor(event.id);
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

            // Return 0 when no event is in range, 1 for an isolated minimum,
            // and 2 when the new heap minimum is close enough to require the
            // pre-reflection tie check.
#if defined(__GNUC__) || defined(__clang__)
            __attribute__((always_inline))
#endif
            inline unsigned int extract_min_leq(
                NT limit, NT tie_tol, unsigned int& id_out, NT& key_out)
            {
                if (_h.empty() || _h[0].key > limit) return 0;
                VOLESTI_HMC_COUNT(n_heap_remove);
                id_out = _h[0].id;
                key_out = _h[0].key;
                _pos[id_out] = -1;
                if (_h.size() == 1) {
                    _h.pop_back();
                } else {
                    _h[0] = _h.back();
                    _pos[_h[0].id] = 0;
                    _h.pop_back();
                    // The replacement came from below the minimum, so it can
                    // only move down.  Avoid the generic indexed remove path.
                    sift_down(0);
                }
                return !_h.empty() && _h[0].key <= key_out + tie_tol ? 2u : 1u;
            }
        };

        inline NT default_trajectory_length(Polytope &P)
        {
            // Quarter period fully decorrelates the free Gaussian dynamics
            // (x(T) = v(0)/omega at omega*T = pi/2), so longer legs only add
            // reflections without mixing benefit.  The cap matters in
            // annealing schedules where a_i (hence omega) grows large.
            // This policy is structurally restricted to OrderPolytope and its
            // diagonal wrapper; both have the same sqrt(d) Euclidean diameter.
            NT diameter = std::sqrt(NT(P.dimension()));
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
                if constexpr (Dynamics::normalize_event_rows) {
                    // A strictly feasible translated center gives C>0.  Use
                    // an exact binary power-of-two row scale so C is O(1).
                    // Dividing by C would round the equation itself and can
                    // measurably move a shallow root.
                    if (!std::isfinite(C) || C <= NT(0))
                        throw std::invalid_argument(
                            "diagonal rounding event row has nonpositive slack");
                    int row_exponent = 0;
                    std::frexp(C, &row_exponent);
                    s0 = std::ldexp(s0, -row_exponent);
                    s1 = std::ldexp(s1, -row_exponent);
                    C = std::ldexp(C, -row_exponent);
                    if (!std::isfinite(s0) || !std::isfinite(s1) ||
                        !std::isfinite(C) || C <= NT(0))
                        throw std::invalid_argument(
                            "diagonal rounding event row has unusable scale");
                }
                _f_i0.push_back(i0); _f_i1.push_back(i1);
                _f_s0.push_back(s0); _f_s1.push_back(s1); _f_C.push_back(C);
                _f_wall.push_back(wall ? 1 : 0);
            };

            for (unsigned int i = 0; i < n; ++i)
                if (P.lower_bound_is_facet(i)) {
                    if constexpr (Dynamics::normalize_event_rows)
                        add_facet(i, i, NT(-1), NT(0),
                                  this->rounding_center_coordinate(i), true);
                    else
                        add_facet(i, i, NT(-1), NT(0), b_event(i), true);
                }

            for (unsigned int i = 0; i < n; ++i)
                if (P.upper_bound_is_facet(i)) {
                    if constexpr (Dynamics::normalize_event_rows)
                        add_facet(i, i, NT(1), NT(0),
                                  NT(1) -
                                  this->rounding_center_coordinate(i), true);
                    else
                        add_facet(i, i, NT(1), NT(0), b_event(n + i), true);
                }

            for (unsigned int k = 0; k < P.num_order_relations(); ++k) {
                auto rel = P.get_order_relation(k);
                if constexpr (Dynamics::normalize_event_rows)
                    add_facet(
                        rel.first, rel.second, NT(1), NT(-1),
                        this->rounding_center_coordinate(rel.second) -
                        this->rounding_center_coordinate(rel.first), false);
                else
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
            _v = this->draw_velocity(n, rng);

            NT T = rng.sample_urdist() * _Len;
            Point const p0 = _p;
            save_last_hit_state();
            _last_status = advance_event_queue(P, T);
            if (_last_status != trajectory_status::success) {
                _p = p0;
                restore_last_hit_state();
                if (_last_status != trajectory_status::reflection_limit)
                    throw_trajectory_failure(_last_status);
            }
        }

        inline trajectory_status advance_event_queue(Polytope const& P, NT T)
        {
            VOLESTI_HMC_COUNT(n_trajectories);
            this->count_trajectory();
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            _testing_trace = testing_trace();
#endif

            if (!this->body_is_compatible(P) ||
                !std::isfinite(T) || T < NT(0)) {
                return trajectory_status::numerical_failure;
            }
            if (!this->velocity_is_valid(_v))
                return trajectory_status::numerical_failure;
            if constexpr (Dynamics::validate_endpoint_feasibility)
                if (!this->position_is_feasible(_p))
                    return trajectory_status::numerical_failure;

            NT theta_rem = _omega * T;
            if constexpr (Dynamics::fail_on_nonfinite_event_arithmetic)
                if (!std::isfinite(theta_rem))
                    return trajectory_status::numerical_failure;
            unsigned int it = 0;
            unsigned int reflections_since_rebase = 0;

            while (theta_rem > NT(0))
            {
                NT const theta_chunk = std::min(theta_rem, chunk_cap());
                if (!begin_chunk(theta_chunk)) {
                    return trajectory_status::numerical_failure;
                }

                for (;;)
                {
                    unsigned int fid = no_facet();
                    contact_batch_result const contact = pop_next_contact_batch(fid);
#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
                    if (contact != contact_batch_result::numerical_failure)
                        if (!verify_contact_result_against_full_scan(
                                P, contact, fid))
                            return trajectory_status::numerical_failure;
#endif
                    if (contact != contact_batch_result::singleton) {
                        if (contact == contact_batch_result::none) {
                            materialize_global_state(_end_cos, _end_sin);
                            theta_rem -= theta_chunk;
                            break;
                        }
                        if (contact != contact_batch_result::disjoint_batch) {
                            return contact == contact_batch_result::shared_coordinate_contact
                                ? trajectory_status::shared_coordinate_contact
                                : trajectory_status::numerical_failure;
                        }
                        bool rebased = false;
                        trajectory_status const status = process_contact_batch(
                            fid, it, reflections_since_rebase, theta_rem, rebased);
                        if (status != trajectory_status::success)
                            return status;
                        if (rebased)
                            break;
                        continue;
                    }

                    set_last_hit_facet(fid);
                    accept_event_direction(fid);
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
                    record_testing_contact_pre(fid, nullptr);
#endif
                    VOLESTI_HMC_COUNT(n_reflections);
                    this->count_reflection();
                    reflect_event(fid);
                    if constexpr (Dynamics::normalize_event_rows) {
                        bool arithmetic_ok = true;
                        int const derivative_sign =
                            facet_derivative_sign_at_direction(
                                fid, _cur_cos, _cur_sin, arithmetic_ok);
                        if (!arithmetic_ok || derivative_sign >= 0) {
                            return trajectory_status::numerical_failure;
                        }
                        if (!reflected_incident_state_is_feasible(fid))
                            return trajectory_status::numerical_failure;
                    }
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
                    record_testing_contact_post();
#endif
                    ++it;
                    ++reflections_since_rebase;

                    if (it >= _rho)
                        return trajectory_status::reflection_limit;

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
                    if (_event_oracle_failed) {
                        return trajectory_status::numerical_failure;
                    }
                }
            }

            if (!_p.getCoefficients().allFinite() ||
                !this->velocity_is_valid(_v))
                return trajectory_status::numerical_failure;
            if constexpr (Dynamics::validate_endpoint_feasibility)
                if (!this->position_is_feasible(_p))
                    return trajectory_status::numerical_failure;
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            _testing_trace.final_position = _p;
            _testing_trace.final_velocity = _v;
#endif
            return trajectory_status::success;
        }

        // Start a chunk covering angles (0, theta_chunk] from the current
        // global state (_p, _v).  The single sincos call per chunk is the
        // only trigonometric evaluation on the non-exceptional path.
        inline bool begin_chunk(NT theta_chunk)
        {
            _alpha = _p.getCoefficients();
            _beta  = _v.getCoefficients() * _inv_omega;
            if (!this->modal_state_is_valid(_alpha, _beta))
                return false;
            if constexpr (Dynamics::normalize_event_rows)
                if (!chunk_start_is_resolved())
                    return false;
            _end_cos = std::cos(theta_chunk);
            _end_sin = std::sin(theta_chunk);
            if constexpr (Dynamics::stable_event_keys)
                _key_end = (_end_sin * _end_sin) / (NT(1) + _end_cos);
            else
                _key_end = NT(1) - _end_cos;
            _cur_cos = NT(1);
            _cur_sin = NT(0);
            _sin_rem = _end_sin;
            _one_minus_cos_rem = _key_end;

            _event_oracle_failed = false;
            _events.clear();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            if (_last_hit_fid != no_facet() || _last_hit_facets.empty()) {
                for (unsigned int fid = 0; fid < F; ++fid)
                    push_or_update_event<false>(fid);
            } else {
                for (unsigned int fid = 0; fid < F; ++fid)
                    push_or_update_event<true>(fid);
            }
            return !_event_oracle_failed;
        }

        template <typename CoordinateVector>
        inline NT facet_linear_value(
            unsigned int fid, CoordinateVector const& coordinates) const
        {
            if constexpr (Dynamics::normalize_event_rows) {
                unsigned int const i0 = _f_i0[fid], i1 = _f_i1[fid];
                if (_f_wall[fid])
                    return _f_s0[fid] * coordinates[i0];
                // Order-facet coefficients are opposite. Subtract before the
                // potentially large 1/slack row scaling to avoid cancellation.
                return _f_s0[fid] * (coordinates[i0] - coordinates[i1]);
            }
            return _f_s0[fid] * coordinates[_f_i0[fid]]
                 + _f_s1[fid] * coordinates[_f_i1[fid]];
        }

        template <unsigned int ComponentCount>
        static inline bool exact_component_sum(
            NT const (&terms)[ComponentCount], NT& value_out, int& sign_out)
        {
            for (NT term : terms)
                if (!std::isfinite(term)) return false;

            NT expansion[ComponentCount] = {};
            unsigned int expansion_size = 0;
            for (unsigned int term_index = 0;
                 term_index < ComponentCount; ++term_index) {
                NT next[ComponentCount] = {};
                unsigned int next_size = 0;
                NT q = terms[term_index];
                for (unsigned int i = 0; i < expansion_size; ++i) {
                    NT const sum = q + expansion[i];
                    NT const virtual_expansion = sum - q;
                    NT const error =
                        (q - (sum - virtual_expansion)) +
                        (expansion[i] - virtual_expansion);
                    if (error != NT(0)) next[next_size++] = error;
                    q = sum;
                }
                if (q != NT(0) || next_size == 0) next[next_size++] = q;
                expansion_size = next_size;
                for (unsigned int i = 0; i < next_size; ++i)
                    expansion[i] = next[i];
            }

            int sign = 0;
            for (unsigned int i = expansion_size; i > 0; --i) {
                if (expansion[i - 1] > NT(0)) { sign = 1; break; }
                if (expansion[i - 1] < NT(0)) { sign = -1; break; }
            }
            NT value = NT(0);
            for (unsigned int i = 0; i < expansion_size; ++i)
                value += expansion[i];
            if (!std::isfinite(value)) return false;
            value_out = value;
            sign_out = sign;
            return true;
        }

        // Compute A^2+B^2-C^2 after scaling A,B,C into [-1,1].  Each product
        // is split with FMA and the six exact components are accumulated as a
        // short nonoverlapping expansion.  This keeps a shallow but
        // representable crossing from inheriting the absolute cancellation
        // error of a single rounded discriminant.
        static inline bool scaled_radial_gap(
            NT A, NT B, NT C, NT& radial_out)
        {
            NT const product_A = A * A;
            NT const product_B = B * B;
            NT const product_C = C * C;
            NT const terms[6] = {
                product_A, std::fma(A, A, -product_A),
                product_B, std::fma(B, B, -product_B),
                -product_C, -std::fma(C, C, -product_C)};
            int sign = 0;
            return exact_component_sum(terms, radial_out, sign) &&
                   sign > 0 && radial_out > NT(0);
        }

        static inline bool exact_product_difference(
            NT a, NT b, NT c, NT d, NT& difference_out, int& sign_out)
        {
            NT const product_ab = a * b;
            NT const product_cd = c * d;
            NT const terms[4] = {
                product_ab, std::fma(a, b, -product_ab),
                -product_cd, -std::fma(c, d, -product_cd)};
            return exact_component_sum(terms, difference_out, sign_out);
        }

        struct structural_facet_evaluation
        {
            NT residual = NT(0);
            NT derivative = NT(0);
            NT residual_scale = NT(0);
            NT derivative_scale = NT(0);
            int residual_sign = 0;
            int derivative_sign = 0;
        };

        static inline void append_exact_product(
            NT coefficient, NT lhs, NT rhs, NT* terms,
            unsigned int& term_count)
        {
            NT const product = lhs * rhs;
            terms[term_count++] = coefficient * product;
            terms[term_count++] = coefficient * std::fma(lhs, rhs, -product);
        }

        // Evaluate the original structured diagonal-rounding facet, before
        // any cached row scaling or rounded center subtraction.  Product FMA
        // tails and all center/coordinate terms are accumulated as a short
        // expansion, so a shallow root cannot be accepted merely because a
        // rounded cached equation happens to vanish.
        inline bool structural_facet_evaluation_at_direction(
            unsigned int fid, NT c, NT s,
            structural_facet_evaluation& evaluation) const
        {
            if constexpr (!Dynamics::normalize_event_rows) {
                NT const A = facet_linear_value(fid, _alpha);
                NT const B = facet_linear_value(fid, _beta);
                NT const residual_terms[6] = {
                    A * c, std::fma(A, c, -(A * c)),
                    B * s, std::fma(B, s, -(B * s)),
                    -_f_C[fid], NT(0)};
                NT const derivative_terms[4] = {
                    B * c, std::fma(B, c, -(B * c)),
                    -(A * s), -std::fma(A, s, -(A * s))};
                for (NT const term : residual_terms)
                    evaluation.residual_scale += std::abs(term);
                for (NT const term : derivative_terms)
                    evaluation.derivative_scale += std::abs(term);
                return std::isfinite(evaluation.residual_scale) &&
                       std::isfinite(evaluation.derivative_scale) &&
                       exact_component_sum(
                           residual_terms, evaluation.residual,
                           evaluation.residual_sign) &&
                       exact_component_sum(
                           derivative_terms, evaluation.derivative,
                           evaluation.derivative_sign);
            } else {
                NT residual_terms[16] = {};
                NT derivative_terms[12] = {};
                unsigned int residual_count = 0;
                unsigned int derivative_count = 0;
                unsigned int const i0 = _f_i0[fid];

                auto append_coordinate = [&](NT coefficient,
                                             unsigned int coordinate) {
                    residual_terms[residual_count++] = coefficient *
                        this->rounding_center_coordinate(coordinate);
                    append_exact_product(
                        coefficient, _alpha(coordinate), c,
                        residual_terms, residual_count);
                    append_exact_product(
                        coefficient, _beta(coordinate), s,
                        residual_terms, residual_count);
                    append_exact_product(
                        -coefficient, _alpha(coordinate), s,
                        derivative_terms, derivative_count);
                    append_exact_product(
                        coefficient, _beta(coordinate), c,
                        derivative_terms, derivative_count);
                };

                if (_f_wall[fid]) {
                    NT const coefficient = _f_s0[fid] < NT(0)
                        ? NT(-1) : NT(1);
                    append_coordinate(coefficient, i0);
                    if (coefficient > NT(0))
                        residual_terms[residual_count++] = NT(-1);
                } else {
                    append_coordinate(NT(1), i0);
                    append_coordinate(NT(-1), _f_i1[fid]);
                }

                if (residual_count > 16 || derivative_count > 12)
                    return false;
                for (unsigned int i = 0; i < residual_count; ++i)
                    evaluation.residual_scale += std::abs(residual_terms[i]);
                for (unsigned int i = 0; i < derivative_count; ++i)
                    evaluation.derivative_scale +=
                        std::abs(derivative_terms[i]);
                return std::isfinite(evaluation.residual_scale) &&
                       std::isfinite(evaluation.derivative_scale) &&
                       exact_component_sum(
                           residual_terms, evaluation.residual,
                           evaluation.residual_sign) &&
                       exact_component_sum(
                           derivative_terms, evaluation.derivative,
                           evaluation.derivative_sign) &&
                       std::isfinite(evaluation.residual) &&
                       std::isfinite(evaluation.derivative);
            }
        }

        inline bool structural_residual_is_within_contact_roundoff(
            structural_facet_evaluation const& evaluation) const
        {
            NT const scale = std::max(
                evaluation.residual_scale, evaluation.derivative_scale);
            NT const tolerance = contact_angle_tol() * scale;
            return std::isfinite(scale) && std::isfinite(tolerance) &&
                   std::abs(evaluation.residual) <= tolerance;
        }

        typedef boost::multiprecision::cpp_rational exact_rational;
        typedef boost::multiprecision::number<
            boost::multiprecision::cpp_dec_float<100>> structural_wide_float;

        static inline exact_rational exact_rational_from_scalar(NT value)
        {
            if (value == NT(0)) return exact_rational(0);
            int exponent = 0;
            NT const fraction = std::frexp(value, &exponent);
            int const digits = std::numeric_limits<NT>::digits;
            NT const scaled_mantissa = std::ldexp(std::abs(fraction), digits);
            boost::multiprecision::cpp_int numerator(scaled_mantissa);
            if (fraction < NT(0)) numerator = -numerator;
            int const shift = exponent - digits;
            if (shift >= 0) {
                numerator <<= shift;
                return exact_rational(numerator);
            }
            boost::multiprecision::cpp_int denominator(1);
            denominator <<= -shift;
            return exact_rational(numerator, denominator);
        }

        struct exact_facet_equation
        {
            exact_rational A, B, C;
        };

        inline exact_facet_equation exact_structural_facet_equation(
            unsigned int fid) const
        {
            unsigned int const i0 = _f_i0[fid];
            auto const alpha = [&](unsigned int i) {
                return exact_rational_from_scalar(_alpha(i));
            };
            auto const beta = [&](unsigned int i) {
                return exact_rational_from_scalar(_beta(i));
            };
            auto const center = [&](unsigned int i) {
                return exact_rational_from_scalar(
                    this->rounding_center_coordinate(i));
            };

            exact_facet_equation equation;
            if (_f_wall[fid]) {
                if (_f_s0[fid] < NT(0)) {
                    equation.A = -alpha(i0);
                    equation.B = -beta(i0);
                    equation.C = center(i0);
                } else {
                    equation.A = alpha(i0);
                    equation.B = beta(i0);
                    equation.C = exact_rational(1) - center(i0);
                }
            } else {
                unsigned int const i1 = _f_i1[fid];
                equation.A = alpha(i0) - alpha(i1);
                equation.B = beta(i0) - beta(i1);
                equation.C = center(i1) - center(i0);
            }
            return equation;
        }

        inline bool refine_structural_root_direction(
            unsigned int fid, NT& c, NT& s) const
        {
            unsigned int const i0 = _f_i0[fid];
            structural_wide_float A, B, C;
            if (_f_wall[fid]) {
                structural_wide_float const alpha0(_alpha(i0));
                structural_wide_float const beta0(_beta(i0));
                structural_wide_float const center0(
                    this->rounding_center_coordinate(i0));
                if (_f_s0[fid] < NT(0)) {
                    A = -alpha0;
                    B = -beta0;
                    C = center0;
                } else {
                    A = alpha0;
                    B = beta0;
                    C = structural_wide_float(1) - center0;
                }
            } else {
                unsigned int const i1 = _f_i1[fid];
                A = structural_wide_float(_alpha(i0)) -
                    structural_wide_float(_alpha(i1));
                B = structural_wide_float(_beta(i0)) -
                    structural_wide_float(_beta(i1));
                C = structural_wide_float(
                        this->rounding_center_coordinate(i1)) -
                    structural_wide_float(
                        this->rounding_center_coordinate(i0));
            }
            using boost::multiprecision::abs;
            using boost::multiprecision::sqrt;
            structural_wide_float const radius_squared = A * A + B * B;
            structural_wide_float const radial_gap =
                radius_squared - C * C;
            if (radius_squared <= 0 || radial_gap <= 0) return false;
            structural_wide_float const D = sqrt(radial_gap);
            structural_wide_float const candidate_cos[2] = {
                (A * C + B * D) / radius_squared,
                (A * C - B * D) / radius_squared};
            structural_wide_float const candidate_sin[2] = {
                (B * C - A * D) / radius_squared,
                (B * C + A * D) / radius_squared};
            structural_wide_float const input_cos(c), input_sin(s);
            structural_wide_float const dot0 =
                input_cos * candidate_cos[0] + input_sin * candidate_sin[0];
            structural_wide_float const dot1 =
                input_cos * candidate_cos[1] + input_sin * candidate_sin[1];
            unsigned int const selected = dot1 > dot0 ? 1u : 0u;
            structural_wide_float const refined_cos = candidate_cos[selected];
            structural_wide_float const refined_sin = candidate_sin[selected];
            structural_wide_float const direction_cross =
                input_cos * refined_sin - input_sin * refined_cos;
            if (abs(direction_cross) > structural_wide_float("0.01"))
                return false;
            c = refined_cos.template convert_to<NT>();
            s = refined_sin.template convert_to<NT>();
            NT const norm = std::hypot(c, s);
            if (!std::isfinite(c) || !std::isfinite(s) ||
                !std::isfinite(norm) || norm <= NT(0))
                return false;
            c /= norm;
            s /= norm;
            return std::isfinite(c) && std::isfinite(s);
        }

        inline bool move_root_direction_to_structural_feasible_side(
            unsigned int fid, NT& c, NT& s) const
        {
            NT const norm = std::hypot(c, s);
            if (!std::isfinite(c) || !std::isfinite(s) ||
                !std::isfinite(norm) || norm <= NT(0))
                return false;
            c /= norm;
            s /= norm;
            structural_facet_evaluation evaluation;
            bool const evaluated = structural_facet_evaluation_at_direction(
                fid, c, s, evaluation);
            return evaluated &&
                (evaluation.residual_sign <= 0 ||
                 structural_residual_is_within_contact_roundoff(evaluation));
        }

        // Cold multi-contact predicate over exact dyadic input equations.
        // Parallel proportional equations share both roots.  Otherwise their
        // unique linear intersection is a common trigonometric root exactly
        // when c^2+s^2=1.  Any tolerance-only near contact is deliberately
        // rejected by the caller instead of being promoted to a batch.
        inline bool facets_have_exact_common_outward_root(
            unsigned int first_fid, unsigned int candidate_fid,
            NT first_cos, NT first_sin, bool& arithmetic_ok) const
        {
            exact_facet_equation const first =
                exact_structural_facet_equation(first_fid);
            exact_facet_equation const candidate =
                exact_structural_facet_equation(candidate_fid);
            exact_rational const determinant =
                first.A * candidate.B - candidate.A * first.B;

            exact_rational common_cos, common_sin;
            if (determinant == 0) {
                bool const proportional =
                    first.A * candidate.C == candidate.A * first.C &&
                    first.B * candidate.C == candidate.B * first.C;
                if (!proportional) return false;
                common_cos = exact_rational_from_scalar(first_cos);
                common_sin = exact_rational_from_scalar(first_sin);
            } else {
                common_cos =
                    (first.C * candidate.B - candidate.C * first.B) /
                    determinant;
                common_sin =
                    (first.A * candidate.C - candidate.A * first.C) /
                    determinant;
                if (common_cos * common_cos + common_sin * common_sin != 1)
                    return false;
            }

            exact_rational const first_derivative =
                -first.A * common_sin + first.B * common_cos;
            exact_rational const candidate_derivative =
                -candidate.A * common_sin + candidate.B * common_cos;
            if (first_derivative <= 0 || candidate_derivative <= 0)
                return false;

            if (determinant != 0) {
                NT const c = common_cos.template convert_to<NT>();
                NT const s = common_sin.template convert_to<NT>();
                if (!std::isfinite(c) || !std::isfinite(s)) {
                    arithmetic_ok = false;
                    return false;
                }
                NT const cross = first_cos * s - first_sin * c;
                NT const dot = first_cos * c + first_sin * s;
                if (!std::isfinite(cross) || !std::isfinite(dot)) {
                    arithmetic_ok = false;
                    return false;
                }
                if (std::abs(cross) > contact_angle_tight_tol() ||
                    dot < NT(1) - contact_angle_tight_tol())
                    return false;
            }
            return true;
        }

        // Earliest hit of facet fid in the window (theta_cur, theta_end],
        // as a unit direction plus its ordering key 1 - cos(theta*).
        template <bool BatchLastHit>
        inline bool next_event_for_facet(unsigned int fid, NT& key_out,
                                         NT& cos_out, NT& sin_out)
        {
            VOLESTI_HMC_COUNT(n_trig_calls);
            this->count_event_recomputation();

            NT const A = facet_linear_value(fid, _alpha);
            NT const B = facet_linear_value(fid, _beta);
            NT const C = _f_C[fid];

            NT R2, scale, veps, disc, amplitude = NT(0);
            if constexpr (Dynamics::normalize_event_rows) {
                amplitude = std::hypot(A, B);
                scale = std::max(NT(1),
                    std::max(amplitude, std::abs(C)));
                veps = value_tol() * std::max(NT(1), std::abs(C));
                if (!std::isfinite(A) || !std::isfinite(B) ||
                    !std::isfinite(C) || !std::isfinite(amplitude) ||
                    !std::isfinite(scale) || !std::isfinite(veps)) {
                    _event_oracle_failed = true;
                    return false;
                }
                // hypot avoids overflow after the diagonal row has been
                // power-of-two scaled to C=O(1).  A separate expansion below
                // handles the case
                // where hypot rounds a genuinely two-root equation to an
                // apparent tangent.
                if (amplitude <= veps)
                    return false;
                NT const amplitude_guard = NT(8) *
                    std::numeric_limits<NT>::epsilon() * scale;
                if (!std::isfinite(amplitude_guard)) {
                    _event_oracle_failed = true;
                    return false;
                }
                if (std::abs(C) > amplitude) {
                    if (std::abs(C) - amplitude > amplitude_guard)
                        return false;
                    // hypot can round below a radius whose exact squared
                    // value is just above C^2.  This near-tangent band is not
                    // a proved no-root case, so fail closed.
                    _event_oracle_failed = true;
                    return false;
                }
                if (amplitude - std::abs(C) <= amplitude_guard) {
                    // The symmetric near-tangent band is unresolved even if
                    // hypot rounded just above |C|.
                    _event_oracle_failed = true;
                    return false;
                }
            } else {
                R2 = A * A + B * B;
                scale = std::max(NT(1),
                    std::max(std::abs(A) + std::abs(B), std::abs(C)));
                veps = value_tol() * scale;
                disc = R2 - C * C;
                if (disc < -veps * scale || R2 <= veps * veps)
                    return false;
            }

            // Max-reach pruning: over the remaining angle De the value can
            // grow by at most |v|*(1 - cos(De)) + |v'|*sin(De)  (De <= pi/2).
            NT val_cur, current_value_eps = veps;
            if constexpr (Dynamics::normalize_event_rows) {
                NT const Ac = A * _cur_cos;
                NT const Bs = B * _cur_sin;
                val_cur = Ac + Bs;
                NT const current_scale = std::max(
                    NT(1), std::abs(Ac) + std::abs(Bs) + std::abs(C));
                current_value_eps = value_tol() * current_scale;
                if (!std::isfinite(val_cur) ||
                    !std::isfinite(current_value_eps)) {
                    _event_oracle_failed = true;
                    return false;
                }
            } else {
                val_cur = A * _cur_cos + B * _cur_sin;
            }
            NT const slack = C - val_cur;
            if (slack < -current_value_eps) {
                // Reaching the outer side means an earlier contact was not
                // resolved.  Never convert its later re-entry into a valid
                // trajectory: propagate a numerical failure so the entire
                // leg is rolled back to its last feasible state.
                _event_oracle_failed = true;
                return false;
            }
            if (slack > NT(0) && !Dynamics::normalize_event_rows) {
                NT const dval_cur = B * _cur_cos - A * _cur_sin;
                NT const reach = std::abs(val_cur) * _one_minus_cos_rem
                               + std::abs(dval_cur) * _sin_rem;
                if constexpr (Dynamics::normalize_event_rows) {
                    if (!std::isfinite(dval_cur) || !std::isfinite(reach)) {
                        _event_oracle_failed = true;
                        return false;
                    }
                }
                if (slack > reach)
                    return false;
            }

            NT c1, s1, c2, s2;
            bool refine_structural_roots = false;
            if constexpr (Dynamics::normalize_event_rows) {
                NT const coefficient_scale = std::max(
                    std::abs(A), std::max(std::abs(B), std::abs(C)));
                int coefficient_exponent = 0;
                std::frexp(coefficient_scale, &coefficient_exponent);
                // A power-of-two scale is exact for every normal input and
                // avoids moving a shallow root merely by rounding A/R, B/R,
                // or C/R before the discriminant is formed.
                NT const scaled_A = std::ldexp(A, -coefficient_exponent);
                NT const scaled_B = std::ldexp(B, -coefficient_exponent);
                NT const scaled_C = std::ldexp(C, -coefficient_exponent);
                auto const scale_round_trips = [&](NT original, NT scaled) {
                    return original == NT(0) ||
                           (scaled != NT(0) &&
                            std::ldexp(scaled, coefficient_exponent) == original);
                };
                NT scaled_radial = NT(0);
                if (!std::isfinite(coefficient_scale) ||
                    coefficient_scale <= NT(0) ||
                    !std::isfinite(scaled_A) ||
                    !std::isfinite(scaled_B) ||
                    !std::isfinite(scaled_C) ||
                    !scale_round_trips(A, scaled_A) ||
                    !scale_round_trips(B, scaled_B) ||
                    !scale_round_trips(C, scaled_C) ||
                    !scaled_radial_gap(
                        scaled_A, scaled_B, scaled_C, scaled_radial)) {
                    // Zero/negative or unrepresentable radial gap is not a
                    // resolvable two-root equation.
                    _event_oracle_failed = true;
                    return false;
                }
                NT const scaled_D = std::sqrt(scaled_radial);
                NT const scaled_R2 = std::fma(
                    scaled_A, scaled_A, scaled_B * scaled_B);
                NT const normalized_derivative =
                    scaled_D / std::sqrt(scaled_R2);
                if (!std::isfinite(scaled_D) ||
                    !std::isfinite(scaled_R2) || scaled_R2 <= NT(0) ||
                    !std::isfinite(normalized_derivative) ||
                    normalized_derivative <= contact_angle_tight_tol()) {
                    _event_oracle_failed = true;
                    return false;
                }
                refine_structural_roots =
                    normalized_derivative <= NT(0.0625);
                NT const inv_scaled_R2 = NT(1) / scaled_R2;
                NT const scaled_AC = scaled_A * scaled_C;
                NT const scaled_BC = scaled_B * scaled_C;
                c1 = std::fma(scaled_B, scaled_D, scaled_AC) *
                     inv_scaled_R2;
                s1 = std::fma(-scaled_A, scaled_D, scaled_BC) *
                     inv_scaled_R2;
                c2 = std::fma(-scaled_B, scaled_D, scaled_AC) *
                     inv_scaled_R2;
                s2 = std::fma(scaled_A, scaled_D, scaled_BC) *
                     inv_scaled_R2;
                NT const norm1 = std::hypot(c1, s1);
                NT const norm2 = std::hypot(c2, s2);
                c1 /= norm1; s1 /= norm1;
                c2 /= norm2; s2 /= norm2;
                if (!std::isfinite(c1) || !std::isfinite(s1) ||
                    !std::isfinite(c2) || !std::isfinite(s2)) {
                    _event_oracle_failed = true;
                    return false;
                }
            } else {
                NT const D = disc > NT(0) ? std::sqrt(disc) : NT(0);
                NT const inv_R2 = NT(1) / R2;
                NT const AC = A * C, BC = B * C, AD = A * D, BD = B * D;
                c1 = (AC + BD) * inv_R2;
                s1 = (BC - AD) * inv_R2;
                c2 = (AC - BD) * inv_R2;
                s2 = (BC + AD) * inv_R2;
            }


            if constexpr (Dynamics::normalize_event_rows) {
                if (refine_structural_roots &&
                    (!refine_structural_root_direction(fid, c1, s1) ||
                     !refine_structural_root_direction(fid, c2, s2))) {
                    _event_oracle_failed = true;
                    return false;
                }
            }

            // A root is inside the window iff it lies strictly ahead of the
            // current direction (cross > 0) and its key does not exceed the
            // chunk end key.  Roots on the lower half circle always fail one
            // of the two tests because the window is within a quarter turn.
            // Only the just-reflected facet uses the wider eps guard, to
            // exclude its own numerical-residual root at the current angle;
            // for all other facets a nearby forward root is a real crossing
            // and skipping it would leak the trajectory out of the polytope.
            NT fwd_eps;
            if constexpr (BatchLastHit)
                fwd_eps = _last_hit_mask[fid] ? _angle_eps : NT(0);
            else
                fwd_eps = (fid == _last_hit_fid) ? _angle_eps : NT(0);
            NT const limit = _key_end + key_tol();
            bool found = false;
            NT best_key = NT(0), best_cos = NT(0), best_sin = NT(0);

            auto consider = [&](NT c, NT s) {
                NT cross = _cur_cos * s - _cur_sin * c;
                if (cross > fwd_eps) {
                    NT const current_dot = _cur_cos * c + _cur_sin * s;
                    if constexpr (Dynamics::stable_event_keys) {
                        if (!std::isfinite(current_dot)) {
                            _event_oracle_failed = true;
                            return;
                        }
                        // A chunk spans at most pi/2.  The antipodal root can
                        // have a tiny positive cross and a tiny endpoint cross
                        // when the physical root is extremely early, but its
                        // negative dot proves that it is outside this chunk.
                        // Reject it before endpoint-close ambiguity handling.
                        if (current_dot < NT(0)) return;
                    }
                    NT k;
                    bool within_window = true;
                    if constexpr (Dynamics::stable_event_keys) {
                        NT end_cross = c * _end_sin - s * _end_cos;
                        if (!std::isfinite(end_cross)) {
                            _event_oracle_failed = true;
                            return;
                        }
                        if (end_cross < -contact_angle_tol()) return;
                        if (!move_root_direction_to_structural_feasible_side(
                                fid, c, s)) {
                            _event_oracle_failed = true;
                            return;
                        }
                        cross = _cur_cos * s - _cur_sin * c;
                        end_cross = c * _end_sin - s * _end_cos;
                        if (!std::isfinite(cross) ||
                            !std::isfinite(end_cross) ||
                            cross <= fwd_eps) {
                            _event_oracle_failed = true;
                            return;
                        }
                        if (std::abs(end_cross) <= contact_angle_tol()) {
                            structural_facet_evaluation end_evaluation;
                            if (!structural_facet_evaluation_at_direction(
                                    fid, _end_cos, _end_sin,
                                    end_evaluation)) {
                                _event_oracle_failed = true;
                                return;
                            }
                            if (end_evaluation.residual_sign < 0) {
                                if (end_cross < NT(0)) return;
                                _event_oracle_failed = true;
                                return;
                            }
                            if (end_evaluation.residual_sign == 0) {
                                c = _end_cos;
                                s = _end_sin;
                                cross = _cur_cos * s - _cur_sin * c;
                                end_cross = NT(0);
                            } else if (end_cross < NT(0)) {
                                _event_oracle_failed = true;
                                return;
                            }
                        }
                        if (end_cross < NT(0)) return;
                        // The in-window root has c>=0 for a quarter chunk.  The
                        // fallback keeps the irrelevant antipodal root finite
                        // even if its rounded c is exactly -1.
                        k = c < NT(0)
                            ? NT(1) - c
                            : (s * s) / (NT(1) + c);
                    } else {
                        k = NT(1) - c;
                    }
                    within_window = k <= limit;
                    if constexpr (Dynamics::stable_event_keys) {
                        if (!std::isfinite(k)) {
                            _event_oracle_failed = true;
                            return;
                        }
                    }
                    if (within_window && (!found || k < best_key)) {
                        found = true;
                        best_key = k; best_cos = c; best_sin = s;
                    }
                } else if constexpr (Dynamics::normalize_event_rows) {
                    if (fwd_eps == NT(0)) {
                        NT const current_dot =
                            _cur_cos * c + _cur_sin * s;
                        if (!std::isfinite(current_dot)) {
                            _event_oracle_failed = true;
                            return;
                        }
                        if (std::abs(cross) <= contact_angle_tight_tol() &&
                            current_dot >=
                                NT(1) - contact_angle_tight_tol()) {
                            structural_facet_evaluation evaluation;
                            if (!structural_facet_evaluation_at_direction(
                                    fid, _cur_cos, _cur_sin, evaluation)) {
                                _event_oracle_failed = true;
                                return;
                            }
                            if (evaluation.residual_sign < 0 &&
                                evaluation.derivative_sign > 0) {
                                // A strictly forward exit collapsed onto the
                                // current represented direction.  Do not skip
                                // it and discover the miss only at the endpoint.
                                _event_oracle_failed = true;
                            }
                        }
                    }
                }
            };
            consider(c1, s1);
            consider(c2, s2);

            if constexpr (Dynamics::stable_event_keys)
                if (_event_oracle_failed) return false;
            if (!found) return false;

            if constexpr (Dynamics::normalize_event_rows) {
                bool arithmetic_ok = true;
                bool const on_structural_facet = facet_is_at_direction(
                    fid, best_cos, best_sin, arithmetic_ok);
                bool const outward = facet_points_outward_at_direction(
                    fid, best_cos, best_sin, arithmetic_ok);
                if (!arithmetic_ok || !on_structural_facet || !outward) {
                    _event_oracle_failed = true;
                    return false;
                }
            }

            VOLESTI_HMC_COUNT(n_trig_hits);
            key_out = best_key; cos_out = best_cos; sin_out = best_sin;
            return true;
        }

        template <bool BatchLastHit>
        inline void push_or_update_event(unsigned int fid)
        {
            NT key, c, s;
            if (next_event_for_facet<BatchLastHit>(fid, key, c, s)) {
                _ev_cos[fid] = c;
                _ev_sin[fid] = s;
                _events.insert_or_update(fid, key);
            } else {
                _events.remove(fid);
            }
        }

#if defined(__GNUC__) || defined(__clang__)
        __attribute__((always_inline))
#endif
        inline contact_batch_result pop_next_contact_batch(
            unsigned int& fid_out)
        {
            unsigned int fid;
            NT key;
            unsigned int const extracted = _events.extract_min_leq(
                _key_end + key_tol(), key_tol(), fid, key);
            if (extracted == 0) return contact_batch_result::none;

            // Every exact pre-reflection tie should have been removed as one
            // batch.  A remaining event at or behind the current direction is
            // unresolved numerical state; reject rather than silently skip.
            NT const cross = _cur_cos * _ev_sin[fid] - _cur_sin * _ev_cos[fid];
            if (cross <= NT(0)) {
                VOLESTI_HMC_COUNT(n_stale_purge);
                return contact_batch_result::numerical_failure;
            }
            if constexpr (Dynamics::normalize_event_rows) {
                bool arithmetic_ok = true;
                bool const on_popped_facet = facet_is_at_direction(
                    fid, _ev_cos[fid], _ev_sin[fid], arithmetic_ok);
                bool const outward = facet_points_outward_at_direction(
                    fid, _ev_cos[fid], _ev_sin[fid], arithmetic_ok);
                if (!arithmetic_ok || !on_popped_facet || !outward) {
                    return contact_batch_result::numerical_failure;
                }
            }

            fid_out = fid;
            if (extracted == 1)
                return contact_batch_result::singleton;

            return collect_contact_batch(fid, key);
        }

        // This path is entered only when a second heap key is numerically
        // coincident with the first.  Keep its active-heap scan out of the
        // ordinary singleton reflection loop.
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline))
#endif
        contact_batch_result collect_contact_batch(unsigned int fid, NT key)
        {
            _contact_batch.clear();
            NT const first_cos = _ev_cos[fid];
            NT const first_sin = _ev_sin[fid];
            _events.for_each_key_leq(key + key_tol(), [&](unsigned int candidate) {
                NT const tie_cross = first_cos * _ev_sin[candidate]
                                   - first_sin * _ev_cos[candidate];
                NT const tie_dot = first_cos * _ev_cos[candidate]
                                 + first_sin * _ev_sin[candidate];
                if constexpr (Dynamics::normalize_event_rows) {
                    if (!std::isfinite(tie_cross) ||
                        !std::isfinite(tie_dot)) {
                        _event_oracle_failed = true;
                        return;
                    }
                    // The key tie window is deliberately wider than the
                    // direction tie window near theta=0, where k=1-cos(theta)
                    // is quadratic.  A candidate whose direction is clearly
                    // later remains in the heap for the next singleton; do
                    // not invoke the exact common-root classifier or turn a
                    // harmless key collision into a numerical failure.
                    if (tie_cross > contact_angle_tight_tol()) return;
                    if (tie_cross < -contact_angle_tight_tol() ||
                        tie_dot < NT(1) - contact_angle_tight_tol()) {
                        _event_oracle_failed = true;
                        return;
                    }
                }
                bool const direction_tied =
                    std::abs(tie_cross) <= contact_angle_tight_tol() &&
                    tie_dot >= NT(1) - contact_angle_tight_tol();
                bool tied = direction_tied;
                if constexpr (Dynamics::normalize_event_rows) {
                    // Both an angular match and an independently evaluated
                    // zero residual are required.  A zero residual outside
                    // the operational angle window is numerically ambiguous:
                    // fail closed instead of guessing simultaneous semantics.
                    bool arithmetic_ok = true;
                    bool const at_first_direction = facet_is_at_direction(
                        candidate, first_cos, first_sin, arithmetic_ok);
                    bool const outward = facet_points_outward_at_direction(
                        candidate, first_cos, first_sin, arithmetic_ok);
                    bool const exact_common_root =
                        facets_have_exact_common_outward_root(
                            fid, candidate, first_cos, first_sin,
                            arithmetic_ok);
                    if (!arithmetic_ok) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (at_first_direction && !outward) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (at_first_direction && !direction_tied) {
                        _event_oracle_failed = true;
                        return;
                    }
                    // Heap-key and residual tolerances are only a reason to
                    // inspect a possible tie.  Promote it to a pre-reflection
                    // batch only when the two exact dyadic structural
                    // equations share this outward root.  A merely close
                    // pair is unresolved here, never guessed simultaneous.
                    if (!exact_common_root) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (!at_first_direction || !outward ||
                        !direction_tied) {
                        _event_oracle_failed = true;
                        return;
                    }
                    tied = direction_tied && at_first_direction &&
                           exact_common_root;
                }
                if (tied)
                    _contact_batch.push_back(candidate);
            });
            if constexpr (Dynamics::normalize_event_rows)
                if (_event_oracle_failed)
                    return contact_batch_result::numerical_failure;
            if (_contact_batch.empty())
                return contact_batch_result::singleton;

            _contact_batch.push_back(fid);
            for (unsigned int const hit_fid : _contact_batch)
                if (hit_fid != fid) _events.remove(hit_fid);

            if (_contact_batch.size() > 1 && !contact_batch_is_disjoint())
                return contact_batch_result::shared_coordinate_contact;
            return contact_batch_result::disjoint_batch;
        }

        inline bool facet_is_at_direction(unsigned int fid, NT c, NT s,
                                           bool& arithmetic_ok) const
        {
            if constexpr (Dynamics::normalize_event_rows) {
                structural_facet_evaluation evaluation;
                if (!structural_facet_evaluation_at_direction(
                        fid, c, s, evaluation)) {
                    arithmetic_ok = false;
                    return false;
                }
                // Candidate roots are constructed (or cold-path refined)
                // before this predicate.  Here the remaining safety question
                // is one-sided: the represented pre-reflection state may be
                // on the facet or a few ulps inside, but never outside it.
                // Exact common-root algebra, not this predicate, decides
                // simultaneous membership.
                if (evaluation.residual_sign > 0 &&
                    !structural_residual_is_within_contact_roundoff(
                        evaluation)) {
                    arithmetic_ok = false;
                    return false;
                }
                return true;
            }

            NT const A = facet_linear_value(fid, _alpha);
            NT const B = facet_linear_value(fid, _beta);
            NT const Ac = A * c;
            NT const Bs = B * s;
            NT const Ac_error = std::fma(A, c, -Ac);
            NT const Bs_error = std::fma(B, s, -Bs);
            NT const sum = Ac + Bs;
            NT const sum_virtual_b = sum - Ac;
            NT const sum_error = (Ac - (sum - sum_virtual_b)) +
                                 (Bs - sum_virtual_b);
            NT const residual_head = sum - _f_C[fid];
            NT const residual_virtual_b = residual_head - sum;
            NT const residual_error =
                (sum - (residual_head - residual_virtual_b)) +
                (-_f_C[fid] - residual_virtual_b);
            NT const product_error_sum = Ac_error + Bs_error;
            NT const addition_error_sum = sum_error + residual_error;
            NT const correction = product_error_sum + addition_error_sum;
            NT const residual = residual_head + correction;
            NT const scale = std::max(
                NT(1), std::abs(Ac) + std::abs(Bs) + std::abs(_f_C[fid]));
            // A narrow absolute band confirms a contact; a wider evaluated-
            // term band only bounds cancellation error.  Values between them
            // are deliberately unresolved rather than treated as a tie.
            NT const absolute_tolerance = NT(64) *
                std::numeric_limits<NT>::epsilon() *
                std::max(NT(1), std::abs(_f_C[fid]));
            NT const evaluated_tolerance = NT(256) *
                std::numeric_limits<NT>::epsilon() * scale;
            NT const rounding_bound = std::numeric_limits<NT>::epsilon() *
                (std::abs(Ac_error) + std::abs(Bs_error) +
                 std::abs(sum_error) + std::abs(residual_error) +
                 std::abs(product_error_sum) +
                 std::abs(addition_error_sum) +
                 std::abs(residual_head) + std::abs(correction)) +
                NT(8) * std::numeric_limits<NT>::denorm_min();
            if (!std::isfinite(Ac) || !std::isfinite(Bs) ||
                !std::isfinite(Ac_error) || !std::isfinite(Bs_error) ||
                !std::isfinite(sum) || !std::isfinite(sum_error) ||
                !std::isfinite(residual_head) ||
                !std::isfinite(residual_error) ||
                !std::isfinite(product_error_sum) ||
                !std::isfinite(addition_error_sum) ||
                !std::isfinite(correction) || !std::isfinite(residual) ||
                !std::isfinite(scale) ||
                !std::isfinite(absolute_tolerance) ||
                !std::isfinite(evaluated_tolerance) ||
                !std::isfinite(rounding_bound)) {
                arithmetic_ok = false;
                return false;
            }
            NT const abs_residual = std::abs(residual);
            // The compensated residual is still rounded when its EFT tails
            // are combined.  Confirm a tie only when a conservative bound on
            // that final rounding remains inside the absolute contact band.
            if (rounding_bound <= absolute_tolerance &&
                abs_residual <= absolute_tolerance - rounding_bound)
                return true;
            // Conversely, call the facet definitely off only when the whole
            // rounding interval lies beyond the wider evaluated-term band.
            if (abs_residual > evaluated_tolerance &&
                abs_residual - evaluated_tolerance > rounding_bound)
                return false;
            // Cancellation makes this interval unresolved at working
            // precision.  Do not guess simultaneous-contact semantics.
            arithmetic_ok = false;
            return false;
        }

        // At a genuine first exit the facet derivative
        // d/dtheta(A cos(theta)+B sin(theta)) is strictly positive.  Check
        // its sign from exact product tails after power-of-two scaling before
        // relying on the no-rehit-within-a-quarter-chunk invariant.
        inline bool facet_points_outward_at_direction(
            unsigned int fid, NT c, NT s, bool& arithmetic_ok) const
        {
            return facet_derivative_sign_at_direction(
                       fid, c, s, arithmetic_ok) > 0;
        }

        inline int facet_derivative_sign_at_direction(
            unsigned int fid, NT c, NT s, bool& arithmetic_ok) const
        {
            if constexpr (Dynamics::normalize_event_rows) {
                structural_facet_evaluation evaluation;
                if (!structural_facet_evaluation_at_direction(
                        fid, c, s, evaluation) ||
                    evaluation.derivative_sign == 0) {
                    arithmetic_ok = false;
                    return 0;
                }
                return evaluation.derivative_sign;
            }

            NT const A = facet_linear_value(fid, _alpha);
            NT const B = facet_linear_value(fid, _beta);
            NT const coefficient_scale = std::max(std::abs(A), std::abs(B));
            if (!std::isfinite(A) || !std::isfinite(B) ||
                !std::isfinite(c) || !std::isfinite(s) ||
                !std::isfinite(coefficient_scale) ||
                coefficient_scale <= NT(0)) {
                arithmetic_ok = false;
                return 0;
            }
            int coefficient_exponent = 0;
            std::frexp(coefficient_scale, &coefficient_exponent);
            NT const scaled_A = std::ldexp(A, -coefficient_exponent);
            NT const scaled_B = std::ldexp(B, -coefficient_exponent);
            NT derivative = NT(0);
            int derivative_sign = 0;
            if (!exact_product_difference(
                    scaled_B, c, scaled_A, s,
                    derivative, derivative_sign) ||
                !std::isfinite(derivative)) {
                arithmetic_ok = false;
                return 0;
            }
            if (derivative_sign == 0) {
                arithmetic_ok = false;
                return 0;
            }
            return derivative_sign;
        }

        inline bool contact_facets_are_disjoint(
            std::vector<unsigned int> const& facets) const
        {
            for (unsigned int i = 0; i < facets.size(); ++i) {
                unsigned int const fi = facets[i];
                unsigned int const i0 = _f_i0[fi], i1 = _f_i1[fi];
                for (unsigned int j = i + 1; j < facets.size(); ++j) {
                    unsigned int const fj = facets[j];
                    unsigned int const j0 = _f_i0[fj], j1 = _f_i1[fj];
                    if (i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1)
                        return false;
                }
            }
            return true;
        }

        inline bool contact_batch_is_disjoint() const
        {
            return contact_facets_are_disjoint(_contact_batch);
        }

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        typedef boost::multiprecision::number<
            boost::multiprecision::cpp_dec_float<100>> verifier_float;

        template <typename CoordinateVector>
        inline verifier_float verifier_facet_linear_value(
            unsigned int fid, CoordinateVector const& coordinates) const
        {
            unsigned int const i0 = _f_i0[fid], i1 = _f_i1[fid];
            verifier_float const x0(coordinates[i0]);
            if constexpr (Dynamics::normalize_event_rows) {
                if (_f_wall[fid])
                    return _f_s0[fid] < NT(0) ? -x0 : x0;
                // Reconstruct the unscaled structured cover equation in wide
                // precision, independent of the cached production row scale.
                return x0 - verifier_float(coordinates[i1]);
            } else {
                verifier_float const s0(_f_s0[fid]);
                if (_f_wall[fid]) return s0 * x0;
                return s0 * x0 +
                       verifier_float(_f_s1[fid]) *
                       verifier_float(coordinates[i1]);
            }
        }

        inline verifier_float verifier_facet_rhs(unsigned int fid) const
        {
            if constexpr (Dynamics::normalize_event_rows) {
                unsigned int const i0 = _f_i0[fid], i1 = _f_i1[fid];
                verifier_float const center0(
                    this->rounding_center_coordinate(i0));
                if (_f_wall[fid])
                    return _f_s0[fid] < NT(0)
                        ? center0 : verifier_float(1) - center0;
                return verifier_float(
                           this->rounding_center_coordinate(i1)) - center0;
            } else {
                return verifier_float(_f_C[fid]);
            }
        }

        // Independent verifier root.  The reference deliberately uses a
        // wide phase/offset solve and wide angle comparisons; it neither
        // reuses the production direction formula nor its rounded heap key.
        // This path exists only in verifier builds.
        inline bool full_scan_first_root(unsigned int fid,
                                         verifier_float& angle_out,
                                         bool& scan_ok)
        {
            NT const A_nt = facet_linear_value(fid, _alpha);
            NT const B_nt = facet_linear_value(fid, _beta);
            NT const C_nt = _f_C[fid];
            if (!std::isfinite(A_nt) || !std::isfinite(B_nt) ||
                !std::isfinite(C_nt)) {
                scan_ok = false;
                return false;
            }

            // After a validated outward hit and reflection, the same C>0
            // sinusoid cannot exit the facet again within this <=pi/2 chunk.
            // Production removes that row until the next chunk; the full scan
            // independently scans every other structured facet.
            if constexpr (Dynamics::normalize_event_rows)
                if (is_last_hit_facet(fid) && _cur_sin > NT(0)) return false;

            verifier_float const Aw = verifier_facet_linear_value(fid, _alpha);
            verifier_float const Bw = verifier_facet_linear_value(fid, _beta);
            verifier_float const Cw = verifier_facet_rhs(fid);
            verifier_float const radius_squared = Aw * Aw + Bw * Bw;
            verifier_float const radial_gap = radius_squared - Cw * Cw;
            if (radius_squared == 0) return false;
            if (radial_gap < 0) return false;
            if (radial_gap == 0) {
                // Tangency has no unambiguous reflection semantics here.
                scan_ok = false;
                return false;
            }

            using boost::multiprecision::abs;
            using boost::multiprecision::acos;
            using boost::multiprecision::atan;
            using boost::multiprecision::atan2;
            using boost::multiprecision::ceil;
            using boost::multiprecision::sqrt;
            verifier_float const current_angle = atan2(
                verifier_float(_cur_sin), verifier_float(_cur_cos));
            verifier_float const end_angle = atan2(
                verifier_float(_end_sin), verifier_float(_end_cos));
            verifier_float const pi = acos(verifier_float(-1));
            verifier_float const two_pi = verifier_float(2) * pi;
            verifier_float const forward_guard = is_last_hit_facet(fid)
                ? verifier_float(_angle_eps) : verifier_float(0);
            verifier_float const lower = current_angle + forward_guard;

            // Wide tan-half solve avoids subtracting phase and offset when a
            // valid first root is tiny compared with either (for example a
            // 1e200 modal coefficient and a 1e-200 trajectory window).
            verifier_float const qa = Aw + Cw;
            verifier_float const qb = -verifier_float(2) * Bw;
            verifier_float const qc = Cw - Aw;
            verifier_float const discriminant =
                qb * qb - verifier_float(4) * qa * qc;
            if (discriminant <= 0) {
                scan_ok = false;
                return false;
            }
            verifier_float const sqrt_discriminant = sqrt(discriminant);
            verifier_float half_angle_roots[2];
            unsigned int root_count = 0;
            if (qa == 0) {
                if (qb == 0) return false;
                half_angle_roots[root_count++] = -qc / qb;
            } else {
                verifier_float const q = -verifier_float(0.5) *
                    (qb + (qb < 0 ? -sqrt_discriminant
                                  : sqrt_discriminant));
                if (q == 0) {
                    half_angle_roots[root_count++] =
                        (-qb + sqrt_discriminant) /
                        (verifier_float(2) * qa);
                    half_angle_roots[root_count++] =
                        (-qb - sqrt_discriminant) /
                        (verifier_float(2) * qa);
                } else {
                    half_angle_roots[root_count++] = q / qa;
                    half_angle_roots[root_count++] = qc / q;
                }
            }
            bool found = false;
            verifier_float best = 0;
            for (unsigned int root_index = 0;
                 root_index < root_count; ++root_index) {
                verifier_float const base =
                    verifier_float(2) * atan(half_angle_roots[root_index]);
                verifier_float const cycles = ceil(
                    (lower - base) / two_pi);
                verifier_float candidate =
                    base + cycles * two_pi;
                if (!(candidate > lower)) candidate += two_pi;
                if (candidate > end_angle) continue;
                verifier_float const derivative =
                    -Aw * boost::multiprecision::sin(candidate) +
                     Bw * boost::multiprecision::cos(candidate);
                if (derivative < 0) continue;
                if (derivative == 0) {
                    scan_ok = false;
                    return false;
                }
                if (!found || candidate < best) {
                    found = true;
                    best = candidate;
                }
            }
            if (!found) return false;
            angle_out = best;
            return true;
        }

        // Correctness reference: scan every structured facet twice,
        // independently of heap state and incident-only invalidation.  The
        // first pass selects the earliest root; the second reconstructs its
        // complete pre-reflection contact set.
        inline bool full_scan_contact_batch(std::vector<unsigned int>& facets,
                                            verifier_float& cos_out,
                                            verifier_float& sin_out,
                                            bool& scan_ok)
        {
            facets.clear();
            scan_ok = true;
            verifier_float best_angle = 0;
            bool found_best = false;
            unsigned int best_fid = no_facet();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());

            for (unsigned int fid = 0; fid < F; ++fid) {
                verifier_float angle;
                if (full_scan_first_root(fid, angle, scan_ok) &&
                    (!found_best || angle < best_angle)) {
                    best_angle = angle;
                    best_fid = fid;
                    found_best = true;
                }
                if (!scan_ok) return false;
            }

            if (!found_best) return false;
            cos_out = boost::multiprecision::cos(best_angle);
            sin_out = boost::multiprecision::sin(best_angle);

            // Only identical represented wide roots form a reference batch.
            // A nonzero separation below verifier resolution is unresolved,
            // never guessed to be simultaneous.
            verifier_float const ambiguity_tolerance =
                verifier_float(1048576) *
                std::numeric_limits<verifier_float>::epsilon();
            for (unsigned int fid = 0; fid < F; ++fid) {
                verifier_float angle;
                if (!full_scan_first_root(fid, angle, scan_ok)) {
                    if (!scan_ok) return false;
                    continue;
                }
                using boost::multiprecision::abs;
                verifier_float const separation = abs(angle - best_angle);
                bool exact_common_root = fid == best_fid;
                if constexpr (Dynamics::normalize_event_rows) {
                    if (!exact_common_root) {
                        bool arithmetic_ok = true;
                        exact_common_root =
                            facets_have_exact_common_outward_root(
                                best_fid, fid,
                                cos_out.template convert_to<NT>(),
                                sin_out.template convert_to<NT>(),
                                arithmetic_ok);
                        if (!arithmetic_ok) {
                            scan_ok = false;
                            return false;
                        }
                    }
                }
                if (exact_common_root || separation == 0) {
                    facets.push_back(fid);
                } else if (separation <= ambiguity_tolerance) {
                    // Nonidentical wide roots are never guessed to be a tie.
                    // If their separation is beneath verifier resolution,
                    // report unresolved instead.
                    scan_ok = false;
                    return false;
                }
            }

            if (facets.empty()) {
                scan_ok = false;
                return false;
            }
            std::sort(facets.begin(), facets.end());
            return true;
        }
#endif

        // Advance once to the contact before reflecting either path.
        inline void accept_event_direction(unsigned int fid)
        {
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
                if constexpr (Dynamics::normalize_event_rows) {
                    NT const y = -_alpha(u) * sinV + _beta(u) * cosV;
                    NT const delta_y = -NT(2) * y;
                    _alpha(u) = std::fma(-delta_y, sinV, _alpha(u));
                    _beta(u) = std::fma(delta_y, cosV, _beta(u));
                } else {
                    NT const x = _alpha(u) * cosV + _beta(u) * sinV;
                    NT const y = -_alpha(u) * sinV + _beta(u) * cosV; // v/omega
                    _alpha(u) = x * cosV + y * sinV;
                    _beta(u)  = x * sinV - y * cosV;
                }
            } else {
                unsigned int const w = _f_i1[fid];
                if constexpr (Dynamics::normalize_event_rows) {
                    // Reflect in the matched diagonal mass metric.  The
                    // modal y values are velocity divided by omega.
                    NT const yu = -_alpha(u) * sinV + _beta(u) * cosV;
                    NT const yw = -_alpha(w) * sinV + _beta(w) * cosV;
                    NT reflected_yu = yu, reflected_yw = yw;
                    this->reflect_cover_eta(u, w, reflected_yu, reflected_yw);
                    NT const delta_yu = reflected_yu - yu;
                    NT const delta_yw = reflected_yw - yw;
                    _alpha(u) = std::fma(-delta_yu, sinV, _alpha(u));
                    _beta(u) = std::fma(delta_yu, cosV, _beta(u));
                    _alpha(w) = std::fma(-delta_yw, sinV, _alpha(w));
                    _beta(w) = std::fma(delta_yw, cosV, _beta(w));
                } else {
                    // Preserve the established identity-mass cover
                    // reflection: swap the two coordinate velocities.
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
        }

#if defined(__GNUC__) || defined(__clang__)
        __attribute__((cold, noinline))
#endif
        trajectory_status process_contact_batch(
            unsigned int first_fid, unsigned int& reflections,
            unsigned int& reflections_since_rebase, NT& theta_rem,
            bool& rebased)
        {
            set_last_hit_batch(_contact_batch);
            accept_event_direction(first_fid);
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            record_testing_contact_pre(first_fid, &_contact_batch);
#endif
            for (unsigned int const fid : _contact_batch) {
                VOLESTI_HMC_COUNT(n_reflections);
                this->count_reflection();
                reflect_event(fid);
                if constexpr (Dynamics::normalize_event_rows) {
                    bool arithmetic_ok = true;
                    int const derivative_sign =
                        facet_derivative_sign_at_direction(
                            fid, _cur_cos, _cur_sin, arithmetic_ok);
                    if (!arithmetic_ok || derivative_sign >= 0) {
                        _contact_batch.clear();
                        return trajectory_status::numerical_failure;
                    }
                }
                ++reflections;
                ++reflections_since_rebase;
            }
            if constexpr (Dynamics::normalize_event_rows) {
                if (!reflected_batch_incident_state_is_feasible()) {
                    _contact_batch.clear();
                    return trajectory_status::numerical_failure;
                }
            }
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            record_testing_contact_post();
#endif

            trajectory_status status = trajectory_status::success;
            if (reflections >= _rho) {
                status = trajectory_status::reflection_limit;
            } else if (_rebase_interval > 0 &&
                       reflections_since_rebase >= _rebase_interval &&
                       _sin_rem > _angle_eps) {
                VOLESTI_HMC_COUNT(n_rebase);
                materialize_global_state(_cur_cos, _cur_sin);
                theta_rem -= std::atan2(_cur_sin, _cur_cos);
                reflections_since_rebase = 0;
                rebased = true;
            } else {
                recompute_incident_facets_batch();
                if (_event_oracle_failed)
                    status = trajectory_status::numerical_failure;
            }
            _contact_batch.clear();
            return status;
        }

        template <typename HitFacetVisitor>
        inline bool reflected_incident_state_is_feasible_impl(
            HitFacetVisitor&& visit_hit_facets)
        {
            unsigned int count = 0;
            auto mark_vertex = [&](unsigned int vertex) {
                for (unsigned int idx = _inc_off[vertex];
                     idx < _inc_off[vertex + 1]; ++idx) {
                    unsigned int const fid = _inc_ids[idx];
                    if (!_affected_mask[fid]) {
                        _affected_mask[fid] = 1;
                        _affected_list[count++] = fid;
                    }
                }
            };
            visit_hit_facets(mark_vertex);

            bool feasible = true;
            for (unsigned int j = 0; j < count; ++j) {
                unsigned int const fid = _affected_list[j];
                structural_facet_evaluation evaluation;
                if (!structural_facet_evaluation_at_direction(
                        fid, _cur_cos, _cur_sin, evaluation) ||
                    (evaluation.residual_sign > 0 &&
                     !structural_residual_is_within_contact_roundoff(
                         evaluation))) {
                    feasible = false;
                }
                _affected_mask[fid] = 0;
            }
            return feasible;
        }

        inline bool reflected_incident_state_is_feasible(
            unsigned int hit_fid)
        {
            return reflected_incident_state_is_feasible_impl(
                [&](auto&& mark_vertex) {
                    mark_vertex(_f_i0[hit_fid]);
                    if (_f_i1[hit_fid] != _f_i0[hit_fid])
                        mark_vertex(_f_i1[hit_fid]);
                });
        }

        inline bool reflected_batch_incident_state_is_feasible()
        {
            return reflected_incident_state_is_feasible_impl(
                [&](auto&& mark_vertex) {
                    for (unsigned int const hit_fid : _contact_batch) {
                        mark_vertex(_f_i0[hit_fid]);
                        if (_f_i1[hit_fid] != _f_i0[hit_fid])
                            mark_vertex(_f_i1[hit_fid]);
                    }
                });
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
                if constexpr (Dynamics::normalize_event_rows) {
                    // After reflection the facet derivative points inward.
                    // With C>0 its next analytic hit is more than pi away, so
                    // it cannot reappear inside the current <=pi/2 chunk.
                    // Removing it also avoids a conditioned residual root at
                    // the just-accepted direction.
                    if (fid == hit_fid) _events.remove(fid);
                    else push_or_update_event<false>(fid);
                } else {
                    if (fid == hit_fid && _f_C[fid] == NT(0))
                        _events.remove(fid);
                    else
                        push_or_update_event<false>(fid);
                }
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
                if constexpr (Dynamics::normalize_event_rows) {
                    if (is_last_hit_facet(fid)) _events.remove(fid);
                    else push_or_update_event<true>(fid);
                } else {
                    if (is_last_hit_facet(fid) && _f_C[fid] == NT(0))
                        _events.remove(fid);
                    else
                        push_or_update_event<true>(fid);
                }
            }
        }

        inline void materialize_global_state(NT c, NT s)
        {
            _p.set_coeffs(_alpha * c + _beta * s);
            _v.set_coeffs((_beta * c - _alpha * s) * _omega);
        }

        // A time-zero contact with outward velocity needs multi-contact
        // semantics that this task deliberately does not define.  Detect it
        // before queue construction and fail closed; an inward boundary start
        // remains well-defined and may proceed.
        inline bool chunk_start_is_resolved() const
        {
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            NT const boundary_tolerance = NT(64) *
                std::numeric_limits<NT>::epsilon();
            for (unsigned int fid = 0; fid < F; ++fid) {
                if constexpr (Dynamics::normalize_event_rows) {
                    structural_facet_evaluation evaluation;
                    if (!structural_facet_evaluation_at_direction(
                            fid, NT(1), NT(0), evaluation) ||
                        (evaluation.residual_sign > 0 &&
                         !structural_residual_is_within_contact_roundoff(
                             evaluation)))
                        return false;
                    if (evaluation.residual_sign >= 0 &&
                        evaluation.derivative_sign > 0)
                        return false;
                    continue;
                }
                NT const value = facet_linear_value(fid, _alpha);
                NT const derivative = facet_linear_value(fid, _beta);
                if (!std::isfinite(value) || !std::isfinite(derivative))
                    return false;
                NT const residual = value - _f_C[fid];
                if (residual > NT(0)) return false;
                if (residual >= -boundary_tolerance && derivative > NT(0))
                    return false;
            }
            return true;
        }

#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
        inline void record_testing_contact_pre(
            unsigned int fid,
            std::vector<unsigned int> const* batch)
        {
            if (_testing_trace.captured_contact) return;
            _testing_trace.captured_contact = true;
            _testing_trace.hit_angle = std::atan2(_cur_sin, _cur_cos);
            if (batch)
                _testing_trace.contact_facets.assign(batch->begin(), batch->end());
            else
                _testing_trace.contact_facets.assign(1, fid);
            std::sort(_testing_trace.contact_facets.begin(),
                      _testing_trace.contact_facets.end());
            _testing_trace.collision_position.set_coeffs(
                _alpha * _cur_cos + _beta * _cur_sin);
            _testing_trace.pre_reflection_velocity.set_coeffs(
                (_beta * _cur_cos - _alpha * _cur_sin) * _omega);
        }

        inline void record_testing_contact_post()
        {
            if (!_testing_trace.captured_contact ||
                _testing_trace.captured_post_reflection)
                return;
            _testing_trace.captured_post_reflection = true;
            _testing_trace.post_reflection_velocity.set_coeffs(
                (_beta * _cur_cos - _alpha * _cur_sin) * _omega);
        }
#endif

#ifdef VOLESTI_VERIFY_ORDERPOLYTOPE_EVENT_QUEUE
        // Cross-check every queue outcome from the same pre-reflection state,
        // including no-event and unresolved shared-coordinate results.
        inline bool verify_contact_result_against_full_scan(
            Polytope const& P, contact_batch_result contact, unsigned int fid)
        {
            (void)P;
            verifier_float full_cos = 0, full_sin = 0;
            bool scan_ok = true;
            bool const found = full_scan_contact_batch(
                _verify_facets, full_cos, full_sin, scan_ok);

            bool rows_ok = false;
            if (contact == contact_batch_result::none) {
                rows_ok = !found;
            } else if (contact == contact_batch_result::singleton) {
                rows_ok = found && _verify_facets.size() == 1 &&
                          _verify_facets.front() == fid;
            } else if (contact == contact_batch_result::disjoint_batch) {
                std::sort(_contact_batch.begin(), _contact_batch.end());
                rows_ok = found && _verify_facets.size() > 1 &&
                          contact_facets_are_disjoint(_verify_facets) &&
                          _contact_batch == _verify_facets;
            } else if (contact ==
                       contact_batch_result::shared_coordinate_contact) {
                std::sort(_contact_batch.begin(), _contact_batch.end());
                rows_ok = found && _verify_facets.size() > 1 &&
                          !contact_facets_are_disjoint(_verify_facets) &&
                          _contact_batch == _verify_facets;
            }
            verifier_float direction_cross_w = 0, direction_dot_w = 1;
            bool direction_ok = contact == contact_batch_result::none;
            if (contact != contact_batch_result::none && found) {
                direction_cross_w =
                    full_cos * verifier_float(_ev_sin[fid]) -
                    full_sin * verifier_float(_ev_cos[fid]);
                direction_dot_w =
                    full_cos * verifier_float(_ev_cos[fid]) +
                    full_sin * verifier_float(_ev_sin[fid]);
                direction_ok =
                    boost::multiprecision::abs(direction_cross_w) <=
                        verifier_float(contact_angle_tol()) &&
                    direction_dot_w >=
                        verifier_float(1) - verifier_float(contact_angle_tol());
            }
            NT const direction_cross =
                direction_cross_w.template convert_to<NT>();
            NT const direction_dot =
                direction_dot_w.template convert_to<NT>();
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            _testing_trace.full_scan_checked = true;
            if (found) {
                ++_testing_trace.full_scan_contact_count;
                _testing_trace.max_full_scan_direction_error = std::max(
                    _testing_trace.max_full_scan_direction_error,
                    std::max(std::abs(direction_cross),
                             std::abs(NT(1) - direction_dot)));
            }
#endif
            bool const ok = scan_ok && rows_ok && direction_ok;
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
            _testing_trace.full_scan_agreed =
                _testing_trace.full_scan_agreed && ok;
#endif
            if (!ok) {
                std::fprintf(stderr,
                    "[verify] result=%d contacts=%zu full_contacts=%zu "
                    "found=%d scan_ok=%d "
                    "dir_cross=%.17g dir_dot=%.17g cur=(%.17g, %.17g) "
                    "end=(%.17g, %.17g)\n",
                    int(contact),
                    _contact_batch.empty() ? std::size_t(1) : _contact_batch.size(),
                    _verify_facets.size(), int(found), int(scan_ok),
                    double(direction_cross), double(direction_dot),
                    double(_cur_cos), double(_cur_sin),
                    double(_end_cos), double(_end_sin));
            }
            return ok;
        }
#endif

        static inline unsigned int no_facet()
        {
            return std::numeric_limits<unsigned int>::max();
        }

        inline bool is_last_hit_facet(unsigned int fid) const
        {
            return fid == _last_hit_fid ||
                   (_last_hit_fid == no_facet() && _last_hit_mask[fid] != 0);
        }

        inline void clear_last_hit_batch()
        {
            for (unsigned int const fid : _last_hit_facets)
                _last_hit_mask[fid] = 0;
            _last_hit_facets.clear();
        }

        inline void set_last_hit_facet(unsigned int fid)
        {
            // A stale batch list can remain allocated and masked: the scalar
            // id makes it inactive, and a future batch replacement clears it.
            // This keeps vector work out of every singleton reflection.
            _last_hit_fid = fid;
        }

        inline void set_last_hit_batch(std::vector<unsigned int> const& facets)
        {
            clear_last_hit_batch();
            _last_hit_fid = no_facet();
            _last_hit_facets.assign(facets.begin(), facets.end());
            for (unsigned int const fid : _last_hit_facets)
                _last_hit_mask[fid] = 1;
        }

        inline void save_last_hit_state()
        {
            _saved_last_hit_fid = _last_hit_fid;
            if (_last_hit_fid == no_facet())
                _saved_last_hit_facets.assign(_last_hit_facets.begin(),
                                              _last_hit_facets.end());
        }

        inline void restore_last_hit_state()
        {
            _contact_batch.clear();
            clear_last_hit_batch();
            _last_hit_fid = _saved_last_hit_fid;
            if (_last_hit_fid == no_facet()) {
                _last_hit_facets.assign(_saved_last_hit_facets.begin(),
                                        _saved_last_hit_facets.end());
                for (unsigned int const fid : _last_hit_facets)
                    _last_hit_mask[fid] = 1;
            }
        }

#if defined(__GNUC__) || defined(__clang__)
        __attribute__((cold, noinline, noreturn))
#else
        [[noreturn]]
#endif
        static void throw_trajectory_failure(
            trajectory_status status)
        {
            if (status == trajectory_status::shared_coordinate_contact)
                throw std::runtime_error(
                    "OrderPolytope exact HMC encountered an unresolved "
                    "shared-coordinate simultaneous contact");
            throw std::runtime_error(
                "OrderPolytope exact HMC encountered a numerical event failure");
        }

        // Quarter period in angle units.
        static inline NT chunk_cap() { return NT(1.57079632679489661923); }

        inline NT time_eps() const { return NT(1e-10); }
        inline NT key_tol() const { return NT(1e-12); }
        inline NT value_tol() const { return NT(1e-12); }
        // Operational floating-point coincidence threshold, not a certified
        // interval proof that two analytic roots are mathematically equal.
        inline NT contact_angle_tol() const {
            if constexpr (Dynamics::normalize_event_rows)
                return NT(256) * std::numeric_limits<NT>::epsilon();
            return NT(64) * std::numeric_limits<NT>::epsilon();
        }
        inline NT contact_angle_tight_tol() const {
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
        // The common singleton last hit stays scalar; the sparse-set mask is
        // activated only after a true multi-contact reflection.
        unsigned int _last_hit_fid, _saved_last_hit_fid;
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
#ifdef VOLESTI_ORDERPOLYTOPE_HMC_TESTING
        testing_trace _testing_trace;
#endif
    };

    parameters param;
};

// The production default remains a distinct compile-time spherical policy.
// Keep a public struct (rather than a type alias) so existing forward
// declarations of the policy remain source-compatible.
struct OrderPolytopeGaussianHamiltonianMonteCarloExactWalk
    : OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy<
          order_polytope_exact_hmc_detail::SphericalDynamics>
{
    using Base = OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy<
        order_polytope_exact_hmc_detail::SphericalDynamics>;
    using Base::Base;
};

// Explicit opt-in policy for matched M=D^{-1}.  It requires a
// DiagonalRoundingOrderPolytope carrying validated diagonal shape data.
struct OrderPolytopeDiagonalGaussianHamiltonianMonteCarloExactWalk
    : OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy<
          order_polytope_exact_hmc_detail::DiagonalDynamics>
{
    using Base = OrderPolytopeGaussianHamiltonianMonteCarloExactWalkPolicy<
        order_polytope_exact_hmc_detail::DiagonalDynamics>;
    using Base::Base;
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
