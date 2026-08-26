// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

// Internal modal dynamics for an arbitrary Gaussian precision A and momentum
// mass M. With M = L L^T, diagonalizing L^{-1} A L^{-T} gives independent
// harmonic modes. True facets remain two-entry sparse rows; their modal
// coefficients are derived on demand instead of being stored in an n-by-F
// matrix.

namespace order_gaussian_hmc_detail {

template <typename NT>
inline NT round_up(NT value)
{
    if (!std::isfinite(value))
        throw std::runtime_error("modal HMC: non-finite interval bound");
    NT const result = std::nextafter(
        value, std::numeric_limits<NT>::infinity());
    if (!std::isfinite(result))
        throw std::runtime_error("modal HMC: interval bound overflow");
    return result;
}

template <typename NT>
inline NT round_down(NT value)
{
    if (!std::isfinite(value))
        throw std::runtime_error("modal HMC: non-finite interval bound");
    NT const result = std::nextafter(
        value, -std::numeric_limits<NT>::infinity());
    if (!std::isfinite(result))
        throw std::runtime_error("modal HMC: interval bound overflow");
    return result;
}

template <typename NT>
inline NT nonnegative_product_up(NT lhs, NT rhs)
{
    return round_up(lhs * rhs);
}

template <typename NT>
inline NT nonnegative_sum_up(NT lhs, NT rhs)
{
    return round_up(lhs + rhs);
}

template <typename NT>
inline bool certified_nonpositive(NT value, NT tolerance)
{
    return value <= -tolerance;
}

template <typename NT>
inline bool certified_positive(NT value, NT tolerance)
{
    return value > tolerance;
}

template <typename NT>
inline NT quadratic_value_upper(NT midpoint_value, NT value_tolerance,
                                NT midpoint_derivative,
                                NT derivative_tolerance,
                                NT curvature_bound, NT half_width)
{
    NT const slope_bound = nonnegative_sum_up(
        std::abs(midpoint_derivative), derivative_tolerance);
    NT const slope_margin = nonnegative_product_up(slope_bound, half_width);
    NT const square_width = nonnegative_product_up(half_width, half_width);
    NT const curvature_margin = nonnegative_product_up(
        NT(0.5), nonnegative_product_up(curvature_bound, square_width));
    return round_up(
        round_up(round_up(midpoint_value + value_tolerance) + slope_margin) +
        curvature_margin);
}

template <typename Polytope>
class ModalTrajectory
{
public:
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::VT VT;
    typedef typename Polytope::MT MT;

    ModalTrajectory(Polytope const& polytope, MT const& precision,
                    MT const& mass, VT const& center)
        : _center(center)
    {
        unsigned int const n = polytope.dimension();
        if (!polytope.has_standard_order_facets())
            throw std::invalid_argument(
                "modal HMC requires standard OrderPolytope facets");
        if (n == 0)
            throw std::invalid_argument(
                "modal HMC: zero-dimensional trajectories are unsupported");
        if (precision.rows() != static_cast<Eigen::Index>(n) ||
            precision.cols() != precision.rows() ||
            mass.rows() != static_cast<Eigen::Index>(n) ||
            mass.cols() != mass.rows())
            throw std::invalid_argument("modal HMC: A/M dimension mismatch");
        if (!precision.allFinite() || !mass.allFinite())
            throw std::invalid_argument("modal HMC: A and M must be finite");
        if (!(precision.array() == precision.transpose().array()).all() ||
            !(mass.array() == mass.transpose().array()).all())
            throw std::invalid_argument(
                "modal HMC: A and M must be exactly symmetric");
        if (_center.size() == 0) _center = VT::Zero(n);
        if (_center.size() != static_cast<Eigen::Index>(n) ||
            !_center.allFinite())
            throw std::invalid_argument(
                "modal HMC: center must be finite and dimension-compatible");

        _mass_cholesky.compute(mass);
        if (_mass_cholesky.info() != Eigen::Success)
            throw std::invalid_argument("modal HMC: M is not SPD");
        MT const L = _mass_cholesky.matrixL();

        MT B = L.template triangularView<Eigen::Lower>().solve(precision);
        B = L.template triangularView<Eigen::Lower>()
                .solve(MT(B.transpose()));
        B = NT(0.5) * (B + MT(B.transpose()));
        if (!B.allFinite())
            throw std::invalid_argument(
                "modal HMC: non-finite generalized eigensystem");

        Eigen::SelfAdjointEigenSolver<MT> eigensolver(B);
        if (eigensolver.info() != Eigen::Success ||
            !eigensolver.eigenvalues().allFinite() ||
            !(eigensolver.eigenvalues().minCoeff() > NT(0)))
            throw std::invalid_argument("modal HMC: A is not SPD");

        _modal_basis =
            L.transpose().template triangularView<Eigen::Upper>().solve(
                MT(eigensolver.eigenvectors()));
        _projection =
            MT(eigensolver.eigenvectors().transpose()) * L.transpose();
        _frequency.resize(n);
        for (unsigned int k = 0; k < n; ++k)
            _frequency(k) = std::sqrt(eigensolver.eigenvalues()(k));
        if (!_modal_basis.allFinite() || !_projection.allFinite())
            throw std::invalid_argument(
                "modal HMC: modal transformation is unusable");

        build_facets(polytope);
        _initial_mode = VT::Zero(n);
        _initial_mode_velocity = VT::Zero(n);
    }

    void set_state(VT const& position, VT const& velocity)
    {
        if (position.size() != static_cast<Eigen::Index>(dimension()) ||
            velocity.size() != static_cast<Eigen::Index>(dimension()) ||
            !position.allFinite() || !velocity.allFinite())
            throw std::invalid_argument(
                "modal HMC: state must be finite and dimension-compatible");
        _initial_mode.noalias() = _projection * (position - _center);
        _initial_mode_velocity.noalias() = _projection * velocity;
        if (!_initial_mode.allFinite() ||
            !_initial_mode_velocity.allFinite())
            throw std::runtime_error("modal HMC: non-finite projected state");
    }

    unsigned int dimension() const
    {
        return static_cast<unsigned int>(_center.size());
    }

    unsigned int num_facets() const
    {
        return static_cast<unsigned int>(_facets.size());
    }

    VT position(NT time) const
    {
        validate_time(time);
        VT mode(dimension());
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const frequency = _frequency(k);
            mode(k) = _initial_mode(k) * std::cos(frequency * time) +
                _initial_mode_velocity(k) / frequency *
                    std::sin(frequency * time);
        }
        VT const result = _center + _modal_basis * mode;
        if (!result.allFinite())
            throw std::runtime_error("modal HMC: non-finite position");
        return result;
    }

    VT velocity(NT time) const
    {
        validate_time(time);
        VT mode_velocity(dimension());
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const frequency = _frequency(k);
            mode_velocity(k) = -_initial_mode(k) * frequency *
                    std::sin(frequency * time) +
                _initial_mode_velocity(k) * std::cos(frequency * time);
        }
        VT const result = _modal_basis * mode_velocity;
        if (!result.allFinite())
            throw std::runtime_error("modal HMC: non-finite velocity");
        return result;
    }

    // g_f(t) = n_f^T x(t) - b_f; g_f <= 0 is feasible.
    NT facet_value(unsigned int facet, NT time) const
    {
        validate_time(time);
        NT value = -_facets.at(facet).offset;
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const frequency = _frequency(k);
            value += modal_coefficient(facet, k) *
                (_initial_mode(k) * std::cos(frequency * time) +
                 _initial_mode_velocity(k) / frequency *
                    std::sin(frequency * time));
        }
        if (!std::isfinite(value))
            throw std::runtime_error("modal HMC: non-finite facet value");
        return value;
    }

    NT facet_derivative(unsigned int facet, NT time) const
    {
        validate_time(time);
        NT value = NT(0);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const frequency = _frequency(k);
            value += modal_coefficient(facet, k) *
                (-_initial_mode(k) * frequency *
                    std::sin(frequency * time) +
                 _initial_mode_velocity(k) * std::cos(frequency * time));
        }
        if (!std::isfinite(value))
            throw std::runtime_error(
                "modal HMC: non-finite facet derivative");
        return value;
    }

    std::pair<NT, NT> facet_derivative_bounds(unsigned int facet) const
    {
        NT first = NT(0);
        NT second = NT(0);
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const amplitude = std::hypot(
                _frequency(k) * _initial_mode(k),
                _initial_mode_velocity(k));
            NT const term = nonnegative_product_up(
                std::abs(modal_coefficient(facet, k)), round_up(amplitude));
            first = nonnegative_sum_up(first, term);
            second = nonnegative_sum_up(
                second, nonnegative_product_up(_frequency(k), term));
        }
        return std::make_pair(first, second);
    }

    NT facet_value_envelope(unsigned int facet) const
    {
        NT envelope = round_up(std::abs(_facets.at(facet).offset));
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const scaled_velocity = round_up(
                std::abs(_initial_mode_velocity(k)) / _frequency(k));
            NT const amplitude = round_up(std::hypot(
                std::abs(_initial_mode(k)), scaled_velocity));
            NT const coefficient = round_up(
                std::abs(modal_coefficient(facet, k)));
            envelope = nonnegative_sum_up(
                envelope, nonnegative_product_up(coefficient, amplitude));
        }
        return envelope;
    }

    NT facet_initial_value_envelope(unsigned int facet) const
    {
        NT envelope = round_up(std::abs(_facets.at(facet).offset));
        for (unsigned int k = 0; k < dimension(); ++k) {
            NT const coefficient = round_up(
                std::abs(modal_coefficient(facet, k)));
            envelope = nonnegative_sum_up(
                envelope, nonnegative_product_up(
                    coefficient, std::abs(_initial_mode(k))));
        }
        return envelope;
    }

    template <typename Vector>
    NT normal_dot(unsigned int facet, Vector const& vector) const
    {
        Facet const& row = _facets.at(facet);
        NT result = row.first_scale * vector(row.first_index);
        if (row.second_index != row.first_index)
            result += row.second_scale * vector(row.second_index);
        return result;
    }

    VT inverse_mass_normal(unsigned int facet) const
    {
        Facet const& row = _facets.at(facet);
        VT normal = VT::Zero(dimension());
        normal(row.first_index) = row.first_scale;
        if (row.second_index != row.first_index)
            normal(row.second_index) = row.second_scale;
        VT const result = _mass_cholesky.solve(normal);
        if (!result.allFinite())
            throw std::runtime_error(
                "modal HMC: non-finite inverse-mass normal");
        return result;
    }

    VT velocity_from_standard_normal(VT const& standard_normal) const
    {
        if (standard_normal.size() != static_cast<Eigen::Index>(dimension()) ||
            !standard_normal.allFinite())
            throw std::invalid_argument(
                "modal HMC: invalid standard-normal velocity input");
        VT const result = _mass_cholesky.matrixU().solve(standard_normal);
        if (!result.allFinite())
            throw std::runtime_error(
                "modal HMC: non-finite velocity refresh");
        return result;
    }

private:
    struct Facet
    {
        unsigned int first_index;
        unsigned int second_index;
        NT first_scale;
        NT second_scale;
        NT offset;
    };

    static void validate_time(NT time)
    {
        if (!std::isfinite(time))
            throw std::invalid_argument("modal HMC: time must be finite");
    }

    NT modal_coefficient(unsigned int facet, unsigned int mode) const
    {
        Facet const& row = _facets.at(facet);
        NT result = row.first_scale * _modal_basis(row.first_index, mode);
        if (row.second_index != row.first_index)
            result += row.second_scale * _modal_basis(row.second_index, mode);
        return result;
    }

    void add_facet(unsigned int first_index, unsigned int second_index,
                   NT first_scale, NT second_scale, NT rhs)
    {
        NT center_dot = first_scale * _center(first_index);
        if (second_index != first_index)
            center_dot += second_scale * _center(second_index);
        NT const offset = rhs - center_dot;
        if (!std::isfinite(offset))
            throw std::invalid_argument(
                "modal HMC: non-finite facet representation");
        _facets.push_back(Facet{first_index, second_index,
                                first_scale, second_scale, offset});
    }

    void build_facets(Polytope const& polytope)
    {
        unsigned int const n = polytope.dimension();
        VT const rhs = polytope.get_vec();
        NT const relation_scale = polytope.is_normalized()
            ? NT(1) / std::sqrt(NT(2)) : NT(1);

        _facets.reserve(polytope.num_true_facets());
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.lower_bound_is_facet(i))
                add_facet(i, i, NT(-1), NT(0), rhs(i));
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.upper_bound_is_facet(i))
                add_facet(i, i, NT(1), NT(0), rhs(n + i));
        for (unsigned int relation_index : polytope.cover_relation_indices()) {
            auto const relation = polytope.get_order_relation(relation_index);
            add_facet(relation.first, relation.second,
                      relation_scale, -relation_scale,
                      rhs(2 * n + relation_index));
        }
    }

    Eigen::LLT<MT> _mass_cholesky;
    MT _modal_basis;
    MT _projection;
    VT _center;
    VT _frequency;
    VT _initial_mode;
    VT _initial_mode_velocity;
    std::vector<Facet> _facets;
};

template <typename NT>
struct OracleOptions
{
    NT time_tolerance = NT(1e-10);
    NT value_tolerance = NT(1e-12);
    unsigned int bisection_steps = 100;
    unsigned long long max_interval_steps = 1000000ULL;
};

template <typename NT>
struct FirstHit
{
    NT time;
    int facet;
    std::vector<int> tied_facets;
};

// Full true-facet scan. An interval is excluded only by an outward-inflated
// derivative/curvature bound. Every unresolved interval is bisected in time
// order; exhaustion and representational stagnation fail closed by throwing.
template <typename Polytope>
FirstHit<typename Polytope::PointType::FT>
first_hit(ModalTrajectory<Polytope> const& trajectory,
          typename Polytope::PointType::FT time_horizon,
          int last_hit_facet,
          OracleOptions<typename Polytope::PointType::FT> const& options =
              OracleOptions<typename Polytope::PointType::FT>())
{
    typedef typename Polytope::PointType::FT NT;
    FirstHit<NT> best{std::numeric_limits<NT>::max(), -1, {}};
    std::vector<std::pair<int, NT>> detected_hits;

    if (!std::isfinite(time_horizon))
        throw std::invalid_argument(
            "modal oracle: time horizon must be finite");
    if (!std::isfinite(options.time_tolerance) ||
        options.time_tolerance < NT(0) ||
        !std::isfinite(options.value_tolerance) ||
        options.value_tolerance < NT(0) ||
        options.bisection_steps == 0 ||
        options.max_interval_steps == 0)
        throw std::invalid_argument("modal oracle: invalid options");
    if (!(time_horizon > NT(0))) {
        best.time = time_horizon;
        return best;
    }

    auto record_hit = [&](int facet, NT root) {
        detected_hits.emplace_back(facet, root);
        if (root < best.time) {
            best.time = root;
            best.facet = facet;
        }
    };

    auto evaluation_margin = [&](NT envelope) {
        NT const operation_count = NT(8) * NT(trajectory.dimension()) + NT(16);
        NT const roundoff = round_up(
            operation_count * std::numeric_limits<NT>::epsilon());
        if (!(roundoff < NT(0.5)))
            throw std::runtime_error(
                "modal oracle: numerical error bound is not representable");
        NT const gamma = round_up(roundoff / (NT(1) - roundoff));
        NT const requested = round_up(options.value_tolerance);
        return nonnegative_sum_up(
            requested, nonnegative_product_up(gamma, envelope));
    };

    for (unsigned int facet = 0; facet < trajectory.num_facets(); ++facet) {
        NT const value_tolerance = evaluation_margin(
            trajectory.facet_value_envelope(facet));

        NT const initial_value_raw = trajectory.facet_value(facet, NT(0));
        NT const initial_value_tolerance = evaluation_margin(
            trajectory.facet_initial_value_envelope(facet));
        if (initial_value_raw > initial_value_tolerance)
            throw std::runtime_error(
                "modal oracle: initial point lies outside the polytope");

        auto const derivative_bounds =
            trajectory.facet_derivative_bounds(facet);
        NT const derivative_bound = derivative_bounds.first;
        NT const curvature_bound = derivative_bounds.second;
        NT const derivative_tolerance =
            evaluation_margin(derivative_bound);
        NT const initial_derivative =
            trajectory.facet_derivative(facet, NT(0));
        NT const initial_value = initial_value_raw >= -initial_value_tolerance
            ? NT(0) : initial_value_raw;

        if (initial_value_raw >= -initial_value_tolerance &&
            initial_derivative > derivative_tolerance) {
            if (static_cast<int>(facet) == last_hit_facet)
                throw std::runtime_error(
                    "modal oracle: reflected facet still moves outward");
            record_hit(static_cast<int>(facet), NT(0));
            continue;
        }

        auto refine_root = [&](NT lower, NT upper) {
            NT lower_value = trajectory.facet_value(facet, lower);
            for (unsigned int step = 0;
                 step < options.bisection_steps; ++step) {
                NT const midpoint = lower + (upper - lower) / NT(2);
                if (!(midpoint > lower) || !(midpoint < upper)) break;
                NT const midpoint_value =
                    trajectory.facet_value(facet, midpoint);
                if (midpoint_value > NT(0)) upper = midpoint;
                else {
                    lower = midpoint;
                    lower_value = midpoint_value;
                }
            }
            if (lower_value > NT(0) ||
                !(std::abs(lower_value) <= value_tolerance) ||
                !(upper - lower <= options.time_tolerance))
                throw std::runtime_error(
                    "modal oracle: root refinement did not converge");
            return lower;
        };

        unsigned long long interval_steps = 0;
        auto search_interval = [&](auto&& self, NT lower, NT lower_value,
                                   bool lower_is_feasible, NT upper,
                                   NT upper_value,
                                   bool upper_is_feasible) -> NT {
            if (interval_steps++ >= options.max_interval_steps)
                throw std::runtime_error(
                    "modal oracle: interval step limit exceeded");
            NT const midpoint = lower + (upper - lower) / NT(2);
            if (!(midpoint > lower) || !(midpoint < upper))
                throw std::runtime_error(
                    "modal oracle: interval isolation made no progress");

            NT const midpoint_value =
                trajectory.facet_value(facet, midpoint);
            NT const midpoint_derivative =
                trajectory.facet_derivative(facet, midpoint);
            NT const half_width = round_up(std::max(
                midpoint - lower, upper - midpoint));
            NT derivative_radius = nonnegative_product_up(
                curvature_bound, half_width);
            derivative_radius = nonnegative_sum_up(
                derivative_radius, derivative_tolerance);
            NT const derivative_lower = round_down(
                midpoint_derivative - derivative_radius);
            NT const derivative_upper = round_up(
                midpoint_derivative + derivative_radius);

            NT const linear_margin = nonnegative_product_up(
                derivative_bound, half_width);
            NT const linear_upper = round_up(
                round_up(midpoint_value + value_tolerance) + linear_margin);
            NT const quadratic_upper = quadratic_value_upper(
                midpoint_value, value_tolerance, midpoint_derivative,
                derivative_tolerance, curvature_bound, half_width);
            NT const value_upper = std::min(linear_upper, quadratic_upper);

            if ((derivative_upper < NT(0) && lower_is_feasible) ||
                (derivative_lower > NT(0) && upper_is_feasible) ||
                value_upper < NT(0))
                return NT(-1);
            if (derivative_lower > NT(0) &&
                lower_is_feasible &&
                certified_positive(upper_value, value_tolerance))
                return refine_root(lower, upper);

            NT const left = self(self, lower, lower_value,
                                 lower_is_feasible, midpoint, midpoint_value,
                                 certified_nonpositive(
                                     midpoint_value, value_tolerance));
            if (left >= NT(0)) return left;
            return self(self, midpoint, midpoint_value, true,
                        upper, upper_value, upper_is_feasible);
        };

        NT const horizon = std::min(
            time_horizon, best.time + options.time_tolerance);
        if (horizon > NT(0)) {
            NT const horizon_value = trajectory.facet_value(facet, horizon);
            NT const root = search_interval(
                search_interval, NT(0), initial_value, true,
                horizon, horizon_value,
                certified_nonpositive(horizon_value, value_tolerance));
            if (root >= NT(0)) {
                if (!(trajectory.facet_derivative(facet, root) >
                      derivative_tolerance))
                    throw std::runtime_error(
                        "modal oracle: unresolved outward root");
                record_hit(static_cast<int>(facet), root);
            }
        }
    }

    if (best.facet < 0) {
        best.time = time_horizon;
    } else {
        for (auto const& hit : detected_hits)
            if (std::abs(hit.second - best.time) <=
                options.time_tolerance)
                best.tied_facets.push_back(hit.first);
    }
    return best;
}

} // namespace order_gaussian_hmc_detail

#endif // RANDOM_WALKS_ORDER_POLYTOPE_HMC_MODAL_HPP
