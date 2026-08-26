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

#include "convex_bodies/order_polytope_diagonal_rounding.hpp"
#include "sampling/sphere.hpp"

namespace order_polytope_hmc_detail {

// Stable 1-cos(theta) for event-direction ordering.
template <typename NT>
inline NT one_minus_cos_stable(NT c, NT s)
{
    return c > NT(0.5) ? s * s / (NT(1) + c) : NT(1) - c;
}

} // namespace order_polytope_hmc_detail

// Specialized exact Gaussian HMC walk for OrderPolytope.  Its sparse event
// queue recomputes only facets incident to reflected coordinates.  In angle
// space, p(theta)=alpha*cos(theta)+beta*sin(theta), and a facet is hit when
// A*cos(theta)+B*sin(theta)=C.  With D=sqrt(A^2+B^2-C^2), the two root
// directions are computed algebraically as
//
//   ((A*C+B*D)/R^2, (B*C-A*D)/R^2) and
//   ((A*C-B*D)/R^2, (B*C+A*D)/R^2), where R^2=A^2+B^2.
//
// Quarter-period chunks order these directions by 1-cos(theta); cross
// products test whether a stored direction is ahead of the current state.

namespace order_polytope_exact_hmc_detail {

// Exact sign, and optionally a representable value, of a floating-point sum.
template <typename NT, unsigned int ComponentCount>
inline bool exact_component_sum(
    NT const (&terms)[ComponentCount], int& sign_out,
    NT* value_out = nullptr)
{
    for (NT const term : terms)
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

    sign_out = 0;
    for (unsigned int i = expansion_size; i > 0; --i) {
        if (expansion[i - 1] > NT(0)) { sign_out = 1; break; }
        if (expansion[i - 1] < NT(0)) { sign_out = -1; break; }
    }
    if (value_out != nullptr) {
        NT value = NT(0);
        for (unsigned int i = 0; i < expansion_size; ++i)
            value += expansion[i];
        if (!std::isfinite(value)) return false;
        *value_out = value;
    }
    return true;
}

template <typename Polytope>
struct DynamicsStateBase
{
    typedef typename Polytope::PointType Point;

    explicit DynamicsStateBase(Polytope const& polytope)
        : _polytope(&polytope),
          _geometry_revision(polytope.geometry_revision())
    {
        if (!polytope.has_standard_order_facets())
            throw std::invalid_argument(
                "OrderPolytope exact HMC requires standard order facets");
    }

    inline bool body_is_compatible(Polytope const& polytope) const
    {
        return _polytope == &polytope &&
               polytope.has_standard_order_facets() &&
               _geometry_revision == polytope.geometry_revision();
    }

    inline bool velocity_is_valid(Point const& velocity) const
    {
        return velocity.dimension() == _polytope->dimension() &&
               velocity.getCoefficients().allFinite();
    }

protected:
    Polytope const* _polytope;
    unsigned long long _geometry_revision;
};

template <typename Polytope>
struct CenteredRoundingDynamicsState : DynamicsStateBase<Polytope>
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::VT VT;

    explicit CenteredRoundingDynamicsState(Polytope const& polytope)
        : DynamicsStateBase<Polytope>(polytope),
          _center(&polytope.center())
    {}

    inline bool position_is_feasible(Point const& centered) const
    {
        auto const& coordinates = centered.getCoefficients();
        for (unsigned int i = 0; i < this->_polytope->dimension(); ++i) {
            NT const center_i = (*_center)(i);
            NT const coordinate_i = coordinates(i);
            if (!std::isfinite(center_i) || !std::isfinite(coordinate_i) ||
                exact_sum_sign(center_i, coordinate_i) < 0 ||
                exact_sum_sign(center_i, coordinate_i, NT(-1)) > 0)
                return false;
        }
        for (unsigned int k = 0;
             k < this->_polytope->num_order_relations(); ++k) {
            auto const relation = this->_polytope->get_order_relation(k);
            NT const center_u = (*_center)(relation.first);
            NT const coordinate_u = coordinates(relation.first);
            NT const center_v = (*_center)(relation.second);
            NT const coordinate_v = coordinates(relation.second);
            if (exact_sum_sign(
                    center_u, coordinate_u, -center_v, -coordinate_v) > 0)
                return false;
        }
        return true;
    }

    inline bool leg_position_is_feasible(Point const& centered) const
    {
        NT const slack_tol = NT(1e-9);
        auto const& coordinates = centered.getCoefficients();
        unsigned int const n = this->_polytope->dimension();
        for (unsigned int i = 0; i < n; ++i) {
            NT const coordinate_i = (*_center)(i) + coordinates(i);
            if (!std::isfinite(coordinate_i) || coordinate_i < -slack_tol ||
                coordinate_i - NT(1) > slack_tol)
                return false;
        }
        for (unsigned int k = 0;
             k < this->_polytope->num_order_relations(); ++k) {
            auto const relation = this->_polytope->get_order_relation(k);
            NT const difference =
                (*_center)(relation.first) + coordinates(relation.first) -
                ((*_center)(relation.second) + coordinates(relation.second));
            if (!std::isfinite(difference) || difference > slack_tol)
                return false;
        }
        return true;
    }

    inline NT center_coordinate(unsigned int i) const
    {
        return (*_center)(i);
    }

protected:
    static inline int exact_sum_sign(
        NT a, NT b, NT c = NT(0), NT d = NT(0))
    {
        NT const terms[4] = {a, b, c, d};
        int sign = 0;
        order_polytope_exact_hmc_detail::exact_component_sum(terms, sign);
        return sign;
    }

    VT const* _center;
};

struct SphericalDynamics
{
    static constexpr bool normalize_event_rows = false;
    static constexpr bool globally_coupled_reflection = false;

    template <typename Polytope>
    struct State : DynamicsStateBase<Polytope>
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;

        explicit State(Polytope const& polytope)
            : DynamicsStateBase<Polytope>(polytope) {}

        template <typename RandomNumberGenerator>
        inline Point draw_velocity(unsigned int n, RandomNumberGenerator& rng) const
        {
            return GetDirection<Point>::apply(n, rng, false);
        }

        inline bool position_is_feasible(Point const& point) const
        {
            return point.dimension() == this->_polytope->dimension() &&
                   point.getCoefficients().allFinite() &&
                   this->_polytope->is_in(point, NT(0)) == -1;
        }
        inline bool leg_position_is_feasible(Point const& point) const
        {
            return position_is_feasible(point);
        }
    };
};

struct DiagonalDynamics
{
    static constexpr bool normalize_event_rows = true;
    static constexpr bool globally_coupled_reflection = false;

    template <typename Polytope>
    struct State : CenteredRoundingDynamicsState<Polytope>
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;

        explicit State(Polytope const& polytope)
            : CenteredRoundingDynamicsState<Polytope>(polytope),
              _shape_diag(&polytope.shape_diag()),
              _sqrt_shape_diag(&polytope.sqrt_shape_diag())
        {}

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
            // Form metric weights before multiplying delta to avoid overflow.
            eta_u -= NT(2) * (d_u / denom) * delta;
            eta_v += NT(2) * (d_v / denom) * delta;
        }

    private:
        VT const* _shape_diag;
        VT const* _sqrt_shape_diag;
    };
};

template <typename Dynamics>
struct WalkPolicy
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

    WalkPolicy(double L, unsigned int _rho)
        : param(L, true, _rho, true) {}

    WalkPolicy(double L)
        : param(L, true, 0, false) {}

    WalkPolicy()
        : param(0, false, 0, false) {}

    template <typename Polytope, typename RandomNumberGenerator>
    struct Walk : private Dynamics::template State<Polytope>
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Dynamics::template State<Polytope> DynamicsState;

        enum class trajectory_status {
            success,
            reflection_limit,
            unresolved_simultaneous_contact,
            // The represented root order cannot be certified.
            ambiguous_tie,
            numerical_failure
        };

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng)
            : Walk(P, p, a_i, rng, parameters(0, false, 0, false))
        {}

        Walk(Polytope &P, Point const& p, NT const& a_i, RandomNumberGenerator &rng,
             parameters const& params)
            : DynamicsState(P)
        {
            _omega = std::sqrt(NT(2) * a_i);
            _inv_omega = NT(1) / _omega;
            if (!std::isfinite(_omega) || _omega <= NT(0) ||
                !std::isfinite(_inv_omega) ||
                p.dimension() != P.dimension() ||
                !p.getCoefficients().allFinite()) {
                throw_trajectory_failure(trajectory_status::numerical_failure);
            }
            _angle_eps = _omega * time_eps();
            _Len = params.set_L ? params.m_L : default_trajectory_length();
            validate_trajectory_length(_Len);
            _rho = params.set_rho ? params.rho : 100 * P.dimension();
            _rebase_interval = params.rebase_interval;
            _last_hit_fid = no_facet();
            build_event_facets(P);
            initialize(P, p, rng);
            _strict_start_feasibility = false;
        }

        // a_i is ignored because omega is fixed when the walk is constructed;
        // annealing callers construct a fresh walk for each phase.
        inline void apply(Polytope const& P, Point& p, NT const&,
                          unsigned int const& walk_length, RandomNumberGenerator &rng)
        {
            if (!this->body_is_compatible(P)) {
                throw_trajectory_failure(trajectory_status::numerical_failure);
            }
            if (p.dimension() != P.dimension())
                throw std::invalid_argument(
                    "OrderPolytope exact HMC point dimension mismatch");

            unsigned int n = P.dimension();
            NT T;

            for (auto j=0u; j<walk_length; ++j)
            {
                try {
                    T = rng.sample_urdist() * _Len;
                    _v = this->draw_velocity(n, rng);
                } catch (...) {
                    p = _p;
                    throw;
                }
                clear_last_hit_batch();
                _last_hit_fid = no_facet();
                Point p0 = _p;

                trajectory_status const status = advance_event_queue(T);
                if (status != trajectory_status::success) {
                    // Roll back position and residual-root guards.
                    _p = p0;
                    reset_last_hit_state();
                    if (status == trajectory_status::ambiguous_tie) {
                        if constexpr (Dynamics::normalize_event_rows) {
                            p = _p;
                            throw_trajectory_failure(status);
                        }
                        // A spherical ambiguous tie is a self-loop.
                    } else if (status != trajectory_status::reflection_limit) {
                        p = _p;
                        throw_trajectory_failure(status);
                    }
                }
                p = _p;
            }
            p = _p;
        }

        inline void update_delta(NT L)
        {
            validate_trajectory_length(L);
            _Len = L;
        }

    private:

        enum class contact_batch_result {
            none,
            singleton,
            compatible_batch,
            unresolved_simultaneous_contact,
            ambiguous_tie
        };

        enum class root_order {
            first_before_candidate,
            equal,
            candidate_before_first,
            unresolved
        };

        // Indexed D-ary heap over the active events in the current chunk.
        class IndexedDHeap
        {
            static constexpr int D = 8;
            struct Entry { NT key; unsigned int id; };
            std::vector<Entry> _h;
            std::vector<int> _pos;

            int par(int i) const { return (i - 1) / D; }
            int child0(int i) const { return D * i + 1; }

            // Hole-based sifts write one position per level and scan each
            // contiguous child group without side-table gathers.
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

            void append_unordered(unsigned int id, NT key)
            {
                _pos[id] = static_cast<int>(_h.size());
                _h.push_back(Entry{key, id});
            }

            void heapify()
            {
                if (_h.size() < 2) return;
                for (int i = par(static_cast<int>(_h.size()) - 1);
                     i >= 0; --i)
                    sift_down(i);
            }

            template <typename Visitor>
            inline void for_each_key_leq(NT limit, Visitor const& visitor) const
            {
                for (Entry const& event : _h)
                    if (event.key <= limit) visitor(event.id);
            }

            void insert_or_update(unsigned int id, NT key)
            {
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
                id_out = _h[0].id;
                key_out = _h[0].key;
                _pos[id_out] = -1;
                if (_h.size() == 1) {
                    _h.pop_back();
                } else {
                    _h[0] = _h.back();
                    _pos[_h[0].id] = 0;
                    _h.pop_back();
                    // Replacing the minimum only requires sift_down.
                    sift_down(0);
                }
                return !_h.empty() && _h[0].key <= key_out + tie_tol ? 2u : 1u;
            }
        };

        inline NT default_trajectory_length()
        {
            // At a quarter period, free Gaussian dynamics reaches
            // x(T)=v(0)/omega.  Cap longer annealing legs at that angle.
            return std::min(NT(1), chunk_cap() * _inv_omega);
        }

        static inline void validate_trajectory_length(NT L)
        {
            if (!std::isfinite(L) || L < NT(0))
                throw std::invalid_argument(
                    "OrderPolytope exact HMC trajectory length must be finite and nonnegative");
        }

        inline bool consume_remaining_angle(NT& theta_rem, NT elapsed) const
        {
            if (!std::isfinite(theta_rem) || !std::isfinite(elapsed) ||
                elapsed <= NT(0))
                return false;
            NT const next = theta_rem - elapsed;
            if (!std::isfinite(next) || !(next < theta_rem))
                return false;
            theta_rem = std::max(NT(0), next);
            return true;
        }

        inline void build_event_facets(Polytope const& P)
        {
            unsigned int n = P.dimension();
            VT b_event = P.get_vec();
            NT const rel_scale = P.is_normalized() ? NT(1) / std::sqrt(NT(2)) : NT(1);

            _f_i0.clear(); _f_i1.clear(); _f_s0.clear(); _f_C.clear();

            auto add_facet = [&](unsigned int i0, unsigned int i1,
                                 NT s0, NT C) {
                if constexpr (Dynamics::normalize_event_rows) {
                    // Scale by an exact power of two so nonzero |C| is O(1).
                    // Scaling by 1/C would move shallow roots.
                    if (!std::isfinite(C) || C <= NT(0))
                        throw std::invalid_argument(
                            "rounded event row has nonpositive slack");
                    int row_exponent = 0;
                    std::frexp(std::abs(C), &row_exponent);
                    s0 = std::ldexp(s0, -row_exponent);
                    C = std::ldexp(C, -row_exponent);
                    if (!std::isfinite(s0))
                        throw std::invalid_argument(
                            "normalized event row has unusable scale");
                }
                _f_i0.push_back(i0); _f_i1.push_back(i1);
                _f_s0.push_back(s0); _f_C.push_back(C);
            };

            for (unsigned int i = 0; i < n; ++i)
                if (P.lower_bound_is_facet(i)) {
                    if constexpr (Dynamics::normalize_event_rows) {
                        NT const C = this->center_coordinate(i);
                        add_facet(i, i, NT(-1), C);
                    } else
                        add_facet(i, i, NT(-1), b_event(i));
                }

            for (unsigned int i = 0; i < n; ++i)
                if (P.upper_bound_is_facet(i)) {
                    if constexpr (Dynamics::normalize_event_rows) {
                        NT const C = NT(1) -
                                     this->center_coordinate(i);
                        add_facet(i, i, NT(1), C);
                    } else
                        add_facet(i, i, NT(1), b_event(n + i));
                }

            for (unsigned int k : P.cover_relation_indices()) {
                auto rel = P.get_order_relation(k);
                if constexpr (Dynamics::normalize_event_rows) {
                    NT const centered_offset =
                        this->center_coordinate(rel.second) -
                        this->center_coordinate(rel.first);
                    add_facet(rel.first, rel.second, NT(1), centered_offset);
                } else
                    add_facet(rel.first, rel.second, rel_scale,
                              b_event(2 * n + k));
            }

            unsigned int const F = static_cast<unsigned int>(_f_i0.size());

            // CSR incidence: walls occur once, covers under both endpoints.
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
            _last_hit_mask.assign(F, 0);
            _last_hit_facets.reserve(F);
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
            trajectory_status const status = advance_event_queue(T);
            if (status != trajectory_status::success) {
                _p = p0;
                reset_last_hit_state();
                if (status == trajectory_status::ambiguous_tie) {
                    if constexpr (Dynamics::normalize_event_rows) {
                        throw_trajectory_failure(status);
                    }
                } else if (status != trajectory_status::reflection_limit) {
                    throw_trajectory_failure(status);
                }
            }
        }

        inline trajectory_status advance_event_queue(NT T)
        {

            if (!std::isfinite(T) || T < NT(0)) {
                return trajectory_status::numerical_failure;
            }
            if (!this->velocity_is_valid(_v))
                return trajectory_status::numerical_failure;
            if (!(_strict_start_feasibility
                      ? this->position_is_feasible(_p)
                      : this->leg_position_is_feasible(_p)))
                return trajectory_status::numerical_failure;

            NT theta_rem = _omega * T;
            if (!std::isfinite(theta_rem))
                return trajectory_status::numerical_failure;
            unsigned int it = 0;
            unsigned int reflections_since_rebase = 0;
            _leg_abort_pending = false;

            while (theta_rem > NT(0))
            {
                NT const theta_chunk = std::min(theta_rem, chunk_cap());
                if (!begin_chunk(theta_chunk)) {
                    return trajectory_status::numerical_failure;
                }

                for (;;)
                {
                    if (_leg_abort_pending)
                        return trajectory_status::ambiguous_tie;
                    unsigned int fid = no_facet();
                    contact_batch_result const contact = pop_next_contact_batch(fid);
                    if (theta_rem == theta_chunk &&
                        contact != contact_batch_result::none &&
                        contact != contact_batch_result::ambiguous_tie &&
                        _ev_cos[fid] == _end_cos &&
                        _ev_sin[fid] == _end_sin) {
                        materialize_global_state(_end_cos, _end_sin);
                        if (!consume_remaining_angle(theta_rem, theta_chunk))
                            return trajectory_status::numerical_failure;
                        _contact_batch.clear();
                        break;
                    }
                    if (contact != contact_batch_result::singleton) {
                        if (contact == contact_batch_result::none) {
                            materialize_global_state(_end_cos, _end_sin);
                            if (!consume_remaining_angle(theta_rem,
                                                         theta_chunk))
                                return trajectory_status::numerical_failure;
                            // A contact guard expires after free flight, but
                            // survives a contact exactly at the chunk end.
                            if (_sin_rem > NT(0))
                                reset_last_hit_state();
                            break;
                        }
                        if (contact == contact_batch_result::ambiguous_tie)
                            return trajectory_status::ambiguous_tie;
                        if (contact ==
                            contact_batch_result::unresolved_simultaneous_contact)
                            return trajectory_status::
                                unresolved_simultaneous_contact;
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
                    reflect_event(fid);
                    ++it;
                    ++reflections_since_rebase;

                    if (it >= _rho)
                        return trajectory_status::reflection_limit;

                    if (_rebase_interval > 0 &&
                        reflections_since_rebase >= _rebase_interval &&
                        _sin_rem > _angle_eps) {
                        // Periodically materialize and rebase modal state to
                        // limit long-leg drift.
                        materialize_global_state(_cur_cos, _cur_sin);
                        NT const elapsed = std::atan2(_cur_sin, _cur_cos);
                        if (elapsed != NT(0) &&
                            !consume_remaining_angle(theta_rem, elapsed))
                            return trajectory_status::numerical_failure;
                        reflections_since_rebase = 0;
                        break;
                    }

                    recompute_incident_facets<false>(fid);
                    if (_event_oracle_failed) {
                        return trajectory_status::numerical_failure;
                    }
                }
            }

            if (!this->velocity_is_valid(_v))
                return trajectory_status::numerical_failure;
            if (!this->position_is_feasible(_p))
                return trajectory_status::numerical_failure;
            return trajectory_status::success;
        }

        // Initialize a quarter-period chunk from the current global state.
        inline bool begin_chunk(NT theta_chunk)
        {
            _alpha = _p.getCoefficients();
            _beta  = _v.getCoefficients() * _inv_omega;
            if (!_alpha.allFinite() || !_beta.allFinite())
                return false;
            if constexpr (Dynamics::normalize_event_rows)
                if (!chunk_start_is_resolved()) {
                    return false;
                }
            _end_cos = std::cos(theta_chunk);
            _end_sin = std::sin(theta_chunk);
            _key_end = (_end_sin * _end_sin) / (NT(1) + _end_cos);
            _cur_cos = NT(1);
            _cur_sin = NT(0);
            _sin_rem = _end_sin;
            _one_minus_cos_rem = _key_end;

            _event_oracle_failed = false;
            _events.clear();
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            bool const batch_last_hit =
                _last_hit_fid == no_facet() && !_last_hit_facets.empty();
            for (unsigned int fid = 0; fid < F; ++fid) {
                NT key, c, s;
                bool const active = batch_last_hit
                    ? next_event_for_facet<true>(fid, key, c, s)
                    : next_event_for_facet<false>(fid, key, c, s);
                if (active) {
                    _ev_cos[fid] = c;
                    _ev_sin[fid] = s;
                    _events.append_unordered(fid, key);
                }
            }
            _events.heapify();
            return !_event_oracle_failed;
        }

        inline bool facet_is_wall(unsigned int fid) const
        {
            return _f_i0[fid] == _f_i1[fid];
        }

        template <typename CoordinateVector>
        inline NT facet_linear_value(
            unsigned int fid, CoordinateVector const& coordinates) const
        {
            unsigned int const i0 = _f_i0[fid];
            if (facet_is_wall(fid))
                return _f_s0[fid] * coordinates[i0];
            return _f_s0[fid] * (coordinates[i0] - coordinates[_f_i1[fid]]);
        }

        // Compute scaled A^2+B^2-C^2 from FMA-split products and an exact
        // expansion, preserving representable shallow crossings.
        static inline bool exact_radial_gap(
            NT A, NT B, NT C, NT& radial_out, int& sign_out)
        {
            NT const product_A = A * A;
            NT const product_B = B * B;
            NT const product_C = C * C;
            NT const terms[6] = {
                product_A, std::fma(A, A, -product_A),
                product_B, std::fma(B, B, -product_B),
                -product_C, -std::fma(C, C, -product_C)};
            return order_polytope_exact_hmc_detail::exact_component_sum(
                terms, sign_out, &radial_out);
        }

        static inline bool exact_product_difference(
            NT a, NT b, NT c, NT d, NT& difference_out, int& sign_out)
        {
            NT const product_ab = a * b;
            NT const product_cd = c * d;
            NT const terms[4] = {
                product_ab, std::fma(a, b, -product_ab),
                -product_cd, -std::fma(c, d, -product_cd)};
            return order_polytope_exact_hmc_detail::exact_component_sum(
                terms, sign_out, &difference_out);
        }

        struct facet_evaluation
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

        // Evaluate cached facet equations with compensation;
        // rounded dynamics instead reconstructs the unscaled facet
        // and accumulates its terms as an exact expansion.
        inline bool evaluate_facet(
            unsigned int fid, NT c, NT s,
            facet_evaluation& evaluation) const
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
                       order_polytope_exact_hmc_detail::exact_component_sum(
                           residual_terms, evaluation.residual_sign,
                           &evaluation.residual) &&
                       order_polytope_exact_hmc_detail::exact_component_sum(
                           derivative_terms, evaluation.derivative_sign,
                           &evaluation.derivative);
            } else {
                NT residual_terms[16] = {};
                NT derivative_terms[12] = {};
                unsigned int residual_count = 0;
                unsigned int derivative_count = 0;
                unsigned int const i0 = _f_i0[fid];

                auto append_coordinate = [&](NT coefficient,
                                             unsigned int coordinate) {
                    residual_terms[residual_count++] = coefficient *
                        this->center_coordinate(coordinate);
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

                if (facet_is_wall(fid)) {
                    NT const coefficient = _f_s0[fid] < NT(0)
                        ? NT(-1) : NT(1);
                    append_coordinate(coefficient, i0);
                    if (coefficient > NT(0))
                        residual_terms[residual_count++] = NT(-1);
                } else {
                    append_coordinate(NT(1), i0);
                    append_coordinate(NT(-1), _f_i1[fid]);
                }

                for (unsigned int i = 0; i < residual_count; ++i)
                    evaluation.residual_scale += std::abs(residual_terms[i]);
                for (unsigned int i = 0; i < derivative_count; ++i)
                    evaluation.derivative_scale +=
                        std::abs(derivative_terms[i]);
                return std::isfinite(evaluation.residual_scale) &&
                       std::isfinite(evaluation.derivative_scale) &&
                       order_polytope_exact_hmc_detail::exact_component_sum(
                           residual_terms, evaluation.residual_sign,
                           &evaluation.residual) &&
                       order_polytope_exact_hmc_detail::exact_component_sum(
                           derivative_terms, evaluation.derivative_sign,
                           &evaluation.derivative);
            }
        }

        inline bool within_contact_roundoff(
            facet_evaluation const& evaluation) const
        {
            NT const scale = std::max(
                evaluation.residual_scale, evaluation.derivative_scale);
            NT const tolerance = contact_angle_tol() * scale;
            return std::abs(evaluation.residual) <= tolerance;
        }

        typedef boost::multiprecision::cpp_rational exact_rational;
        typedef boost::multiprecision::number<
            boost::multiprecision::cpp_dec_float<100>> wide_float;

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

        inline exact_facet_equation make_exact_facet_equation(
            unsigned int fid) const
        {
            if constexpr (!Dynamics::normalize_event_rows) {
                exact_facet_equation equation;
                equation.A = exact_rational_from_scalar(
                    facet_linear_value(fid, _alpha));
                equation.B = exact_rational_from_scalar(
                    facet_linear_value(fid, _beta));
                equation.C = exact_rational_from_scalar(_f_C[fid]);
                return equation;
            } else {
                unsigned int const i0 = _f_i0[fid];
                auto const alpha = [&](unsigned int i) {
                    return exact_rational_from_scalar(_alpha(i));
                };
                auto const beta = [&](unsigned int i) {
                    return exact_rational_from_scalar(_beta(i));
                };
                auto const center = [&](unsigned int i) {
                    return exact_rational_from_scalar(
                        this->center_coordinate(i));
                };

                exact_facet_equation equation;
                if (facet_is_wall(fid)) {
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
        }

        static inline bool exact_outward_roots_are_equal(
            exact_facet_equation const& first,
            exact_facet_equation const& candidate)
        {
            exact_rational const determinant =
                first.A * candidate.B - candidate.A * first.B;
            if (determinant == 0) {
                bool const proportional =
                    first.A * candidate.C == candidate.A * first.C &&
                    first.B * candidate.C == candidate.B * first.C;
                return proportional &&
                    first.A * candidate.A + first.B * candidate.B > 0;
            }

            exact_rational const common_cos =
                (first.C * candidate.B - candidate.C * first.B) /
                determinant;
            exact_rational const common_sin =
                (first.A * candidate.C - candidate.A * first.C) /
                determinant;
            if (common_cos * common_cos + common_sin * common_sin != 1)
                return false;
            return -first.A * common_sin + first.B * common_cos > 0 &&
                   -candidate.A * common_sin +
                       candidate.B * common_cos > 0;
        }

        static inline void exact_half_angle_signs(
            exact_facet_equation const& equation, NT separator,
            int& value_sign, int& derivative_sign)
        {
            exact_rational const t = exact_rational_from_scalar(separator);
            exact_rational const value =
                (equation.C + equation.A) * t * t -
                exact_rational(2) * equation.B * t +
                equation.C - equation.A;
            exact_rational const derivative =
                (equation.C + equation.A) * t - equation.B;
            value_sign = (value > 0) - (value < 0);
            derivative_sign = (derivative > 0) - (derivative < 0);
        }

        inline root_order compare_outward_roots(
            unsigned int first_fid, unsigned int candidate_fid) const
        {
            NT direction_cross = NT(0);
            int direction_sign = 0;
            if (!exact_product_difference(
                    _ev_cos[first_fid], _ev_sin[candidate_fid],
                    _ev_sin[first_fid], _ev_cos[candidate_fid],
                    direction_cross, direction_sign))
                return root_order::unresolved;
            exact_facet_equation const first =
                make_exact_facet_equation(first_fid);
            exact_facet_equation const candidate =
                make_exact_facet_equation(candidate_fid);
            if (exact_outward_roots_are_equal(first, candidate))
                return root_order::equal;
            if (std::abs(direction_cross) > contact_angle_tight_tol())
                return direction_sign > 0
                    ? root_order::first_before_candidate
                    : root_order::candidate_before_first;

            NT const first_denominator = NT(1) + _ev_cos[first_fid];
            NT const candidate_denominator = NT(1) + _ev_cos[candidate_fid];
            NT const first_t = _ev_sin[first_fid] / first_denominator;
            NT const candidate_t =
                _ev_sin[candidate_fid] / candidate_denominator;
            NT const midpoint_t = (first_t + candidate_t) / NT(2);
            if (first_denominator > NT(0) && candidate_denominator > NT(0) &&
                std::isfinite(midpoint_t)) {
                auto certify_at = [&](NT separator) {
                    int first_value = 0, first_derivative = 0;
                    int candidate_value = 0, candidate_derivative = 0;
                    exact_half_angle_signs(
                        first, separator, first_value, first_derivative);
                    exact_half_angle_signs(
                        candidate, separator,
                        candidate_value, candidate_derivative);
                    if (first_derivative < 0 && candidate_derivative < 0) {
                        if (first_value < 0 && candidate_value > 0)
                            return root_order::first_before_candidate;
                        if (candidate_value < 0 && first_value > 0)
                            return root_order::candidate_before_first;
                    }
                    return root_order::unresolved;
                };
                root_order order = certify_at(midpoint_t);
                if (order == root_order::unresolved)
                    order = certify_at(std::nextafter(
                        midpoint_t, -std::numeric_limits<NT>::infinity()));
                if (order == root_order::unresolved)
                    order = certify_at(std::nextafter(
                        midpoint_t, std::numeric_limits<NT>::infinity()));
                if (order != root_order::unresolved) return order;
            }

            return root_order::unresolved;
        }

        // Refine both roots of a shallow equation in wide precision.
        inline bool refine_root_pair(
            unsigned int fid, NT& c1, NT& s1, NT& c2, NT& s2) const
        {
            unsigned int const i0 = _f_i0[fid];
            wide_float A, B, C;
            if (facet_is_wall(fid)) {
                wide_float const alpha0(_alpha(i0));
                wide_float const beta0(_beta(i0));
                wide_float const center0(
                    this->center_coordinate(i0));
                if (_f_s0[fid] < NT(0)) {
                    A = -alpha0;
                    B = -beta0;
                    C = center0;
                } else {
                    A = alpha0;
                    B = beta0;
                    C = wide_float(1) - center0;
                }
            } else {
                unsigned int const i1 = _f_i1[fid];
                A = wide_float(_alpha(i0)) -
                    wide_float(_alpha(i1));
                B = wide_float(_beta(i0)) -
                    wide_float(_beta(i1));
                C = wide_float(
                        this->center_coordinate(i1)) -
                    wide_float(
                        this->center_coordinate(i0));
            }
            using boost::multiprecision::abs;
            using boost::multiprecision::sqrt;
            wide_float const radius_squared = A * A + B * B;
            wide_float const radial_gap =
                radius_squared - C * C;
            if (radial_gap <= 0) return false;
            wide_float const D = sqrt(radial_gap);
            wide_float const inv_radius_squared =
                wide_float(1) / radius_squared;
            wide_float const candidate_cos[2] = {
                (A * C + B * D) * inv_radius_squared,
                (A * C - B * D) * inv_radius_squared};
            wide_float const candidate_sin[2] = {
                (B * C - A * D) * inv_radius_squared,
                (B * C + A * D) * inv_radius_squared};
            NT* const root_cos[2] = {&c1, &c2};
            NT* const root_sin[2] = {&s1, &s2};
            for (unsigned int root = 0; root < 2; ++root) {
                wide_float const input_cos(*root_cos[root]);
                wide_float const input_sin(*root_sin[root]);
                wide_float const dot0 =
                    input_cos * candidate_cos[0] +
                    input_sin * candidate_sin[0];
                wide_float const dot1 =
                    input_cos * candidate_cos[1] +
                    input_sin * candidate_sin[1];
                unsigned int const selected = dot1 > dot0 ? 1u : 0u;
                wide_float const refined_cos =
                    candidate_cos[selected];
                wide_float const refined_sin =
                    candidate_sin[selected];
                wide_float const direction_cross =
                    input_cos * refined_sin - input_sin * refined_cos;
                if (abs(direction_cross) > wide_float("0.01"))
                    return false;
                NT c = refined_cos.template convert_to<NT>();
                NT s = refined_sin.template convert_to<NT>();
                NT const norm = std::hypot(c, s);
                if (!std::isfinite(norm) || norm <= NT(0))
                    return false;
                *root_cos[root] = c / norm;
                *root_sin[root] = s / norm;
            }
            return true;
        }

        inline bool normalize_feasible_root(
            unsigned int fid, NT& c, NT& s) const
        {
            NT const norm = std::hypot(c, s);
            if (!std::isfinite(norm) || norm <= NT(0))
                return false;
            c /= norm;
            s /= norm;
            facet_evaluation evaluation;
            bool const evaluated = evaluate_facet(
                fid, c, s, evaluation);
            return evaluated &&
                (evaluation.residual_sign <= 0 ||
                 within_contact_roundoff(evaluation));
        }

        // Earliest hit of facet fid in the window (theta_cur, theta_end],
        // as a unit direction plus its ordering key 1 - cos(theta*).
        template <bool BatchLastHit>
        inline bool next_event_for_facet(unsigned int fid, NT& key_out,
                                         NT& cos_out, NT& sin_out)
        {

            NT A = facet_linear_value(fid, _alpha);
            NT B = facet_linear_value(fid, _beta);
            NT C = _f_C[fid];

            if (!std::isfinite(A) || !std::isfinite(B) ||
                !std::isfinite(C)) {
                _event_oracle_failed = true;
                return false;
            }
            NT const coefficient_scale = std::max(
                std::abs(A), std::max(std::abs(B), std::abs(C)));
            if (coefficient_scale > NT(0)) {
                // Exact power-of-two scaling preserves the root equation.
                int coefficient_exponent = 0;
                std::frexp(coefficient_scale, &coefficient_exponent);
                NT const scaled_A = std::ldexp(A, -coefficient_exponent);
                NT const scaled_B = std::ldexp(B, -coefficient_exponent);
                NT const scaled_C = std::ldexp(C, -coefficient_exponent);
                if ((A != NT(0) && scaled_A == NT(0)) ||
                    (B != NT(0) && scaled_B == NT(0)) ||
                    (C != NT(0) && scaled_C == NT(0))) {
                    _event_oracle_failed = true;
                    return false;
                }
                A = scaled_A;
                B = scaled_B;
                C = scaled_C;
            }

            NT R2, scale, veps, disc, amplitude = NT(0);
            if constexpr (Dynamics::normalize_event_rows) {
                amplitude = std::hypot(A, B);
                scale = std::max(NT(1), amplitude);
                veps = value_tol();
                // hypot avoids overflow; the guards below reject unresolved
                // near tangencies.
                if (amplitude <= veps)
                    return false;
                NT const amplitude_guard = NT(8) *
                    std::numeric_limits<NT>::epsilon() * scale;
                if (std::abs(C) > amplitude) {
                    if (std::abs(C) - amplitude > amplitude_guard)
                        return false;
                    // This side of the near-tangent band is unresolved.
                    _event_oracle_failed = true;
                    return false;
                }
                if (amplitude - std::abs(C) <= amplitude_guard) {
                    _event_oracle_failed = true;
                    return false;
                }
            } else {
                R2 = std::fma(A, A, B * B);
                scale = std::max(NT(1),
                    std::abs(A) + std::abs(B));
                veps = value_tol() * scale;
                disc = std::fma(-C, C, R2);
                NT const disc_tol = veps * scale;
                if (disc < -disc_tol || R2 <= veps * veps)
                    return false;
                if (disc <= disc_tol) {
                    int radial_sign = 0;
                    if (!exact_radial_gap(A, B, C, disc, radial_sign)) {
                        _event_oracle_failed = true;
                        return false;
                    }
                    if (radial_sign <= 0)
                        return false;
                    if (!(disc > NT(0))) {
                        _event_oracle_failed = true;
                        return false;
                    }
                }
            }

            // Prune facets that cannot reach the boundary in this chunk.
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
                // Negative slack means a missed contact; abort rather than
                // accepting a later re-entry.
                _leg_abort_pending = true;
                return false;
            }
            if (slack > NT(0)) {
                // |v(theta+De)-v(theta)| is bounded by
                // |v|*(1-cos De)+|v'|*sin De; pad it against roundoff.
                NT const dval_cur = B * _cur_cos - A * _cur_sin;
                NT const reach = std::abs(val_cur) * _one_minus_cos_rem
                               + std::abs(dval_cur) * _sin_rem;
                NT const reach_bound = std::fma(
                    reach,
                    NT(1) + NT(64) * std::numeric_limits<NT>::epsilon(),
                    NT(64) * std::numeric_limits<NT>::epsilon() * scale);
                if (!std::isfinite(dval_cur) || !std::isfinite(reach) ||
                    !std::isfinite(reach_bound)) {
                    _event_oracle_failed = true;
                    return false;
                }
                if (slack > reach_bound)
                    return false;
            }

            NT c1, s1, c2, s2;
            bool needs_refinement = false;
            if constexpr (Dynamics::normalize_event_rows) {
                NT radial = NT(0);
                int radial_sign = 0;
                if (!exact_radial_gap(A, B, C, radial, radial_sign) ||
                    radial_sign <= 0 || !(radial > NT(0))) {
                    // The two-root radial gap is unresolved.
                    _event_oracle_failed = true;
                    return false;
                }
                NT const D = std::sqrt(radial);
                NT const R2 = std::fma(A, A, B * B);
                NT const normalized_derivative =
                    D / std::sqrt(R2);
                if (!std::isfinite(D) ||
                    !std::isfinite(R2) ||
                    !std::isfinite(normalized_derivative) ||
                    normalized_derivative <= contact_angle_tight_tol()) {
                    _event_oracle_failed = true;
                    return false;
                }
                // Direction error is about eps/u for normalized derivative u;
                // refine once it approaches the contact-angle tolerance.
                needs_refinement =
                    normalized_derivative <= NT(1e-3);
                NT const inv_R2 = NT(1) / R2;
                NT const AC = A * C;
                NT const BC = B * C;
                c1 = std::fma(B, D, AC) * inv_R2;
                s1 = std::fma(-A, D, BC) * inv_R2;
                c2 = std::fma(-B, D, AC) * inv_R2;
                s2 = std::fma(A, D, BC) * inv_R2;
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
                NT const D = std::sqrt(disc);
                NT const inv_R2 = NT(1) / R2;
                NT const AC = A * C, BC = B * C, AD = A * D, BD = B * D;
                c1 = (AC + BD) * inv_R2;
                s1 = (BC - AD) * inv_R2;
                c2 = (AC - BD) * inv_R2;
                s2 = (BC + AD) * inv_R2;
            }


            if constexpr (Dynamics::normalize_event_rows) {
                if (needs_refinement &&
                    !refine_root_pair(fid, c1, s1, c2, s2)) {
                    _event_oracle_failed = true;
                    return false;
                }
            }

            // A root is in-window when its cross product is forward and its
            // key precedes the chunk end.  Only the last-hit facet gets an
            // epsilon guard against its residual root.
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
                    if (!std::isfinite(current_dot)) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (current_dot < NT(0)) return;
                    NT end_cross = c * _end_sin - s * _end_cos;
                    if (!std::isfinite(end_cross)) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (end_cross < -contact_angle_tol()) return;
                    if (std::abs(end_cross) <= contact_angle_tol() &&
                        !normalize_feasible_root(
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
                        facet_evaluation end_evaluation;
                        if (!evaluate_facet(
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
                    NT const k = c < NT(0)
                        ? NT(1) - c
                        : (s * s) / (NT(1) + c);
                    if (!std::isfinite(k)) {
                        _event_oracle_failed = true;
                        return;
                    }
                    if (k <= limit && (!found || k < best_key)) {
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
                            facet_evaluation evaluation;
                            if (!evaluate_facet(
                                    fid, _cur_cos, _cur_sin, evaluation)) {
                                _event_oracle_failed = true;
                                return;
                            }
                            if (evaluation.residual_sign < 0 &&
                                evaluation.derivative_sign > 0)
                                _event_oracle_failed = true;
                        }
                    }
                } else {
                    // Spherical HMC reflects an outward zero-time root so a
                    // fresh wall start cannot escape; the last-hit guard still
                    // suppresses residual roots.
                    NT const current_dot = _cur_cos * c + _cur_sin * s;
                    NT const derivative = B * c - A * s;
                    if (fwd_eps == NT(0) &&
                        cross >= -_angle_eps &&
                        current_dot > NT(0) && derivative > NT(0)) {
                        NT const current_key =
                            order_polytope_hmc_detail::one_minus_cos_stable(
                                _cur_cos, _cur_sin);
                        if (!found || current_key < best_key) {
                            found = true;
                            best_key = current_key;
                            best_cos = _cur_cos;
                            best_sin = _cur_sin;
                        }
                    }
                }
            };
            consider(c1, s1);
            if constexpr (Dynamics::normalize_event_rows)
                consider(c2, s2);

            if (_event_oracle_failed) {
                return false;
            }
            if (!found) return false;

            if constexpr (Dynamics::normalize_event_rows) {
                // Use exact expansion validation only outside the fast band.
                NT const residual =
                    std::fma(A, best_cos, std::fma(B, best_sin, -C));
                NT const derivative = std::fma(B, best_cos, -A * best_sin);
                NT const accept_band =
                    contact_angle_tol() * std::max(NT(1), amplitude);
                bool const fast_ok = std::isfinite(residual) &&
                    std::isfinite(derivative) &&
                    std::abs(residual) <= accept_band && derivative > NT(0);
                if (!fast_ok) {
                    bool const contact = is_outward_contact(
                        fid, best_cos, best_sin);
                    if (!contact) {
                        _event_oracle_failed = true;
                        return false;
                    }
                }
            }

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

            // An event at or behind the current direction is unresolved unless
            // the policy explicitly recognizes an immediate outward contact.
            NT const cross = _cur_cos * _ev_sin[fid] - _cur_sin * _ev_cos[fid];
            if (cross <= NT(0)) {
                bool immediate_outward_contact = false;
                if constexpr (!Dynamics::normalize_event_rows) {
                    NT const dot = _cur_cos * _ev_cos[fid] +
                                   _cur_sin * _ev_sin[fid];
                    immediate_outward_contact =
                        !is_last_hit_facet(fid) &&
                        cross >= -_angle_eps && dot > NT(0) &&
                        is_current_outward_contact(fid);
                }
                if (!immediate_outward_contact) {
                    return contact_batch_result::ambiguous_tie;
                }
            }
            // Stored event directions are immutable, so pop does not revalidate.

            if (extracted == 1) {
                fid_out = fid;
                return contact_batch_result::singleton;
            }

            contact_batch_result const result = collect_contact_batch(fid, key);
            fid_out = fid;
            return result;
        }

        // Keep the coincident-key scan out of the singleton hot path.
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline))
#endif
        contact_batch_result collect_contact_batch(unsigned int& fid, NT key)
        {
            if constexpr (!Dynamics::normalize_event_rows) {
                bool possible_tie = false;
                NT const first_cos = _ev_cos[fid];
                NT const first_sin = _ev_sin[fid];
                _events.for_each_key_leq(
                    key + key_tol(), [&](unsigned int candidate) {
                        NT const cross = first_cos * _ev_sin[candidate] -
                                         first_sin * _ev_cos[candidate];
                        NT const dot = first_cos * _ev_cos[candidate] +
                                       first_sin * _ev_sin[candidate];
                        possible_tie = possible_tie ||
                            (std::abs(cross) <= contact_angle_tight_tol() &&
                             dot >= NT(1) - contact_angle_tight_tol());
                    });
                if (!possible_tie)
                    return contact_batch_result::singleton;
            }
            return collect_certified_contact_batch(fid, key);
        }

        contact_batch_result collect_certified_contact_batch(
            unsigned int& fid, NT key)
        {
            unsigned int const popped_fid = fid;
            auto const fail = [&]() {
                return contact_batch_result::ambiguous_tie;
            };
            _contact_batch.assign(1, popped_fid);
            _events.for_each_key_leq(key + key_tol(),
                [&](unsigned int candidate) {
                    _contact_batch.push_back(candidate);
                });

            unsigned int winner = popped_fid;
            for (unsigned int const candidate : _contact_batch) {
                if (candidate == winner) continue;
                root_order const order =
                    compare_outward_roots(winner, candidate);
                if (order == root_order::candidate_before_first) {
                    winner = candidate;
                } else if (order == root_order::unresolved) {
                    return fail();
                }
            }

            std::size_t const candidate_count = _contact_batch.size();
            std::size_t tie_count = 0;
            for (std::size_t i = 0; i < candidate_count; ++i) {
                unsigned int const candidate = _contact_batch[i];
                root_order const order = candidate == winner
                    ? root_order::equal
                    : compare_outward_roots(winner, candidate);
                if (order == root_order::equal) {
                    if (candidate == winner) {
                        _contact_batch[tie_count++] = candidate;
                        continue;
                    }
                    bool const contact = is_outward_contact(
                        candidate, _ev_cos[winner], _ev_sin[winner]);
                    if (!contact)
                        return fail();
                    _contact_batch[tie_count++] = candidate;
                } else if (order == root_order::first_before_candidate) {
                    NT represented_cross = NT(0);
                    int cross_sign = 0;
                    exact_product_difference(
                        _ev_cos[winner], _ev_sin[candidate],
                        _ev_sin[winner], _ev_cos[candidate],
                        represented_cross, cross_sign);
                    if (cross_sign <= 0)
                        return fail();
                    if (std::abs(represented_cross) <=
                        contact_angle_tight_tol()) {
                        facet_evaluation evaluation;
                        bool const safely_before =
                            evaluate_facet(
                                candidate, _ev_cos[winner], _ev_sin[winner],
                                evaluation) &&
                            evaluation.residual_sign < 0 &&
                            evaluation.derivative_sign > 0;
                        if (!safely_before) return fail();
                    }
                } else {
                    return fail();
                }
            }
            _contact_batch.resize(tie_count);

            if (winner != popped_fid)
                _events.insert_or_update(popped_fid, key);
            for (unsigned int const hit_fid : _contact_batch)
                if (hit_fid != popped_fid) _events.remove(hit_fid);
            fid = winner;

            if (_contact_batch.size() == 1) {
                _contact_batch.clear();
                return contact_batch_result::singleton;
            }
            if (!contact_batch_is_resolvable())
                return contact_batch_result::unresolved_simultaneous_contact;
            return contact_batch_result::compatible_batch;
        }

        inline bool facet_is_at_direction(unsigned int fid, NT c, NT s) const
        {
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
            // The inner band certifies contact; the outer band certifies
            // separation.  Values between them remain unresolved.
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
                !std::isfinite(rounding_bound))
                return false;
            NT const abs_residual = std::abs(residual);
            if (abs_residual <= absolute_tolerance - rounding_bound)
                return true;
            if (abs_residual - evaluated_tolerance > rounding_bound)
                return false;
            return false;
        }

        inline bool is_current_outward_contact(unsigned int fid) const
        {
            return facet_is_at_direction(fid, _cur_cos, _cur_sin) &&
                   facet_derivative_sign(fid, _cur_cos, _cur_sin) > 0;
        }

        inline int facet_derivative_sign(unsigned int fid, NT c, NT s) const
        {
            NT const A = facet_linear_value(fid, _alpha);
            NT const B = facet_linear_value(fid, _beta);
            NT const coefficient_scale = std::max(std::abs(A), std::abs(B));
            int coefficient_exponent = 0;
            std::frexp(coefficient_scale, &coefficient_exponent);
            NT const scaled_A = std::ldexp(A, -coefficient_exponent);
            NT const scaled_B = std::ldexp(B, -coefficient_exponent);
            NT derivative = NT(0);
            int derivative_sign = 0;
            if (!exact_product_difference(
                    scaled_B, c, scaled_A, s,
                    derivative, derivative_sign))
                return 0;
            return derivative_sign;
        }

        inline bool is_outward_contact(
            unsigned int fid, NT c, NT s) const
        {
            facet_evaluation evaluation;
            if (!evaluate_facet(
                    fid, c, s, evaluation) ||
                evaluation.derivative_sign <= 0 ||
                (evaluation.residual_sign > 0 &&
                 !within_contact_roundoff(
                     evaluation)))
                return false;
            return true;
        }

        inline bool contact_facets_are_compatible(
            std::vector<unsigned int> const& facets) const
        {
            for (unsigned int i = 0; i < facets.size(); ++i) {
                unsigned int const fi = facets[i];
                unsigned int const i0 = _f_i0[fi], i1 = _f_i1[fi];
                for (unsigned int j = i + 1; j < facets.size(); ++j) {
                    unsigned int const fj = facets[j];
                    unsigned int const j0 = _f_i0[fj], j1 = _f_i1[fj];
                    if constexpr (Dynamics::globally_coupled_reflection) {
                        if (!this->contact_normals_are_orthogonal(
                                i0, i1, j0, j1))
                            return false;
                    } else if (i0 == j0 || i0 == j1 ||
                               i1 == j0 || i1 == j1) {
                        return false;
                    }
                }
            }
            return true;
        }

        inline bool contact_batch_is_resolvable() const
        {
            return contact_facets_are_compatible(_contact_batch);
        }


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

            if constexpr (Dynamics::globally_coupled_reflection) {
                this->reflect_eta(
                    u, _f_i1[fid], cosV, sinV, _alpha, _beta);
            } else {
                if (facet_is_wall(fid)) {
                    if constexpr (Dynamics::normalize_event_rows) {
                        NT const y = -_alpha(u) * sinV + _beta(u) * cosV;
                        NT const delta_y = -NT(2) * y;
                        _alpha(u) = std::fma(-delta_y, sinV, _alpha(u));
                        _beta(u) = std::fma(delta_y, cosV, _beta(u));
                    } else {
                        NT const x = _alpha(u) * cosV + _beta(u) * sinV;
                        NT const y = -_alpha(u) * sinV + _beta(u) * cosV;
                        _alpha(u) = x * cosV + y * sinV;
                        _beta(u)  = x * sinV - y * cosV;
                    }
                } else {
                    unsigned int const w = _f_i1[fid];
                    if constexpr (Dynamics::normalize_event_rows) {
                        NT const yu = -_alpha(u) * sinV + _beta(u) * cosV;
                        NT const yw = -_alpha(w) * sinV + _beta(w) * cosV;
                        NT reflected_yu = yu, reflected_yw = yw;
                        this->reflect_cover_eta(
                            u, w, reflected_yu, reflected_yw);
                        NT const delta_yu = reflected_yu - yu;
                        NT const delta_yw = reflected_yw - yw;
                        _alpha(u) = std::fma(-delta_yu, sinV, _alpha(u));
                        _beta(u) = std::fma(delta_yu, cosV, _beta(u));
                        _alpha(w) = std::fma(-delta_yw, sinV, _alpha(w));
                        _beta(w) = std::fma(delta_yw, cosV, _beta(w));
                    } else {
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
            for (unsigned int const fid : _contact_batch) {
                reflect_event(fid);
                ++reflections;
                ++reflections_since_rebase;
            }

            trajectory_status status = trajectory_status::success;
            if (reflections >= _rho) {
                status = trajectory_status::reflection_limit;
            } else if (_rebase_interval > 0 &&
                       reflections_since_rebase >= _rebase_interval &&
                       _sin_rem > _angle_eps) {
                materialize_global_state(_cur_cos, _cur_sin);
                NT const elapsed = std::atan2(_cur_sin, _cur_cos);
                if (elapsed != NT(0) &&
                    !consume_remaining_angle(theta_rem, elapsed)) {
                    status = trajectory_status::numerical_failure;
                } else {
                    reflections_since_rebase = 0;
                    rebased = true;
                }
            } else {
                recompute_incident_facets<true>();
                if (_event_oracle_failed)
                    status = trajectory_status::numerical_failure;
            }
            _contact_batch.clear();
            return status;
        }

        template <bool BatchLastHit>
        inline void recompute_incident_facets(
            unsigned int singleton_fid = no_facet())
        {
            if constexpr (Dynamics::globally_coupled_reflection) {
                _events.clear();
                for (unsigned int fid = 0; fid < _f_i0.size(); ++fid) {
                    bool const last_hit = BatchLastHit
                        ? is_last_hit_facet(fid)
                        : fid == singleton_fid;
                    if (last_hit) continue;
                    NT key, c, s;
                    if (next_event_for_facet<BatchLastHit>(
                            fid, key, c, s)) {
                        _ev_cos[fid] = c;
                        _ev_sin[fid] = s;
                        _events.append_unordered(fid, key);
                    }
                }
                _events.heapify();
                return;
            }

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

            if constexpr (BatchLastHit) {
                for (unsigned int const hit_fid : _contact_batch) {
                    mark_vertex(_f_i0[hit_fid]);
                    if (!facet_is_wall(hit_fid))
                        mark_vertex(_f_i1[hit_fid]);
                }
            } else {
                mark_vertex(_f_i0[singleton_fid]);
                if (!facet_is_wall(singleton_fid))
                    mark_vertex(_f_i1[singleton_fid]);
            }

            for (unsigned int j = 0; j < count; ++j) {
                unsigned int const fid = _affected_list[j];
                _affected_mask[fid] = 0;
                bool const last_hit = is_last_hit_facet(fid);
                if constexpr (Dynamics::normalize_event_rows) {
                    // A positive-offset hit cannot exit again in this chunk;
                    // leaving it out also suppresses its residual root.
                    if (!last_hit) push_or_update_event<BatchLastHit>(fid);
                } else {
                    bool const suppress_last_hit =
                        last_hit && _f_C[fid] == NT(0);
                    if (!suppress_last_hit)
                        push_or_update_event<BatchLastHit>(fid);
                }
            }
        }

        inline void materialize_global_state(NT c, NT s)
        {
            _p.set_coeffs(_alpha * c + _beta * s);
            _v.set_coeffs((_beta * c - _alpha * s) * _omega);
        }

        // Reject outward contacts at theta=0; inward boundary starts are valid.
        inline bool chunk_start_is_resolved() const
        {
            unsigned int const F = static_cast<unsigned int>(_f_i0.size());
            NT const boundary_tolerance = NT(64) *
                std::numeric_limits<NT>::epsilon();
            for (unsigned int fid = 0; fid < F; ++fid) {
                NT const value = facet_linear_value(fid, _alpha);
                NT const derivative = facet_linear_value(fid, _beta);
                NT const residual = value - _f_C[fid];
                // Allow row-scaled accumulated reflection roundoff.
                NT const residual_tol = value_tol() * std::max(
                    NT(1), std::max(std::abs(value), std::abs(_f_C[fid])));
                if (residual > residual_tol) return false;
                if (residual >= -boundary_tolerance && derivative > NT(0))
                    return false;
            }
            return true;
        }

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
            // A scalar id deactivates stale batch masks without vector work.
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

        inline void reset_last_hit_state()
        {
            _contact_batch.clear();
            clear_last_hit_batch();
            _last_hit_fid = no_facet();
        }

#if defined(__GNUC__) || defined(__clang__)
        __attribute__((cold, noinline, noreturn))
#else
        [[noreturn]]
#endif
        static void throw_trajectory_failure(
            trajectory_status status)
        {
            if (status ==
                trajectory_status::unresolved_simultaneous_contact)
                throw std::runtime_error(
                    "OrderPolytope exact HMC encountered an unresolved "
                    "simultaneous contact");
            if (status == trajectory_status::ambiguous_tie)
                throw std::runtime_error(
                    "OrderPolytope exact HMC encountered an ambiguous "
                    "simultaneous-event tie");
            throw std::runtime_error(
                "OrderPolytope exact HMC encountered a numerical event failure");
        }

        // Quarter period in angle units.
        static inline NT chunk_cap() { return NT(1.57079632679489661923); }

        inline NT time_eps() const { return NT(1e-10); }
        inline NT key_tol() const { return NT(1e-12); }
        inline NT value_tol() const { return NT(1e-12); }
        // Floating-point tie tolerance, not an exact certificate.
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
        bool _leg_abort_pending = false;
        // The constructor validates exactly; later legs allow wall roundoff.
        bool _strict_start_feasibility = true;
        bool _event_oracle_failed = false;
        // The common singleton last hit stays scalar; the sparse-set mask is
        // activated only after a true multi-contact reflection.
        unsigned int _last_hit_fid;
        std::vector<unsigned char> _last_hit_mask;
        std::vector<unsigned int> _last_hit_facets;

        // Facet data, structure-of-arrays.  Equal indices identify walls;
        // cover rows have the derived second coefficient -s0.
        std::vector<unsigned int> _f_i0, _f_i1;
        std::vector<NT> _f_s0, _f_C;

        // CSR lists of facets incident to each coordinate.
        std::vector<unsigned int> _inc_off, _inc_ids;

        // Pending events: unit direction per facet plus the indexed heap.
        std::vector<NT> _ev_cos, _ev_sin;
        IndexedDHeap _events;
        std::vector<unsigned int> _contact_batch;
        std::vector<unsigned char> _affected_mask;
        std::vector<unsigned int> _affected_list;
    };

    parameters param;
};

} // namespace order_polytope_exact_hmc_detail

struct OrderPolytopeExactHMCWalk
    : order_polytope_exact_hmc_detail::WalkPolicy<
          order_polytope_exact_hmc_detail::SphericalDynamics>
{
    using Base = order_polytope_exact_hmc_detail::WalkPolicy<
        order_polytope_exact_hmc_detail::SphericalDynamics>;
    using Base::Base;
};

struct OrderPolytopeDiagonalExactHMCWalk
    : order_polytope_exact_hmc_detail::WalkPolicy<
          order_polytope_exact_hmc_detail::DiagonalDynamics>
{
    using Base = order_polytope_exact_hmc_detail::WalkPolicy<
        order_polytope_exact_hmc_detail::DiagonalDynamics>;
    using Base::Base;
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GAUSSIAN_EXACT_HMC_WALK_HPP
