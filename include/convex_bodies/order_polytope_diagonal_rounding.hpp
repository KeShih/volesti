// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP
#define ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "convex_bodies/orderpolytope.h"

enum class OrderPolytopeDiagonalCoolingStage {
    Schedule,
    Ratio
};

template <typename NT>
struct OrderPolytopeDiagonalCoolingDiagnostics {
    double metric_setup_microseconds = 0.0;
    NT initial_a0 = std::numeric_limits<NT>::quiet_NaN();
    unsigned int cooling_phase_count = 0;
    unsigned long long schedule_trajectories = 0;
    unsigned long long ratio_trajectories = 0;
    unsigned long long reflection_count = 0;
    unsigned long long event_recomputation_count = 0;
    double cooling_microseconds = 0.0;
    OrderPolytopeDiagonalCoolingStage active_stage =
        OrderPolytopeDiagonalCoolingStage::Schedule;

    unsigned long long total_trajectories() const
    {
        return schedule_trajectories + ratio_trajectories;
    }

    double microseconds_per_trajectory() const
    {
        unsigned long long const total = total_trajectories();
        return total == 0 ? 0.0 : cooling_microseconds / static_cast<double>(total);
    }
};

template <typename NT>
struct OrderPolytopeDiagonalRoundingDiagnostics {
    double metric_setup_microseconds = 0.0;
    double total_estimator_microseconds = 0.0;
    std::vector<OrderPolytopeDiagonalCoolingDiagnostics<NT> > residual_cooling;

    unsigned long long schedule_trajectories() const
    {
        unsigned long long total = 0;
        for (auto const& diagnostic : residual_cooling)
            total += diagnostic.schedule_trajectories;
        return total;
    }

    unsigned long long ratio_trajectories() const
    {
        unsigned long long total = 0;
        for (auto const& diagnostic : residual_cooling)
            total += diagnostic.ratio_trajectories;
        return total;
    }

    unsigned long long total_trajectories() const
    {
        return schedule_trajectories() + ratio_trajectories();
    }

    unsigned long long reflection_count() const
    {
        unsigned long long total = 0;
        for (auto const& diagnostic : residual_cooling)
            total += diagnostic.reflection_count;
        return total;
    }

    unsigned long long event_recomputation_count() const
    {
        unsigned long long total = 0;
        for (auto const& diagnostic : residual_cooling)
            total += diagnostic.event_recomputation_count;
        return total;
    }

    double cooling_microseconds() const
    {
        double total = 0.0;
        for (auto const& diagnostic : residual_cooling)
            total += diagnostic.cooling_microseconds;
        return total;
    }

    double microseconds_per_trajectory() const
    {
        unsigned long long const total = total_trajectories();
        return total == 0 ? 0.0 : cooling_microseconds() / static_cast<double>(total);
    }
};

// User-provided diagonal shape for implicit rounding in the original
// OrderPolytope coordinates.  At cooling phase a, the target is
// exp(-a (x-c)^T D^{-1} (x-c)); its untruncated covariance is D/(2a).
template <typename Point>
class OrderPolytopeDiagonalRounding {
public:
    typedef typename Point::FT NT;
    typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;

    OrderPolytopeDiagonalRounding(Poset const& poset,
                                  Point const& center,
                                  VT const& shape_diag,
                                  bool require_strict_feasibility = true)
        : OrderPolytopeDiagonalRounding(
              OrderPolytope<Point>(poset), center, shape_diag,
              require_strict_feasibility)
    {}

    OrderPolytopeDiagonalRounding(OrderPolytope<Point> const& polytope,
                                  Point const& center,
                                  VT const& shape_diag,
                                  bool require_strict_feasibility = true)
        : _center(center), _shape_diag(shape_diag)
    {
        auto const start = std::chrono::steady_clock::now();
        validate_and_precompute(polytope, require_strict_feasibility);
        auto const end = std::chrono::steady_clock::now();
        _setup_microseconds =
            std::chrono::duration<double, std::micro>(end - start).count();
    }

    unsigned int dimension() const
    {
        return static_cast<unsigned int>(_shape_diag.size());
    }

    Point const& center() const { return _center; }
    VT const& shape_diag() const { return _shape_diag; }
    VT const& inv_shape_diag() const { return _inv_shape_diag; }
    VT const& sqrt_shape_diag() const { return _sqrt_shape_diag; }
    NT log_det_shape() const { return _log_det_shape; }
    double setup_microseconds() const { return _setup_microseconds; }
    std::vector<NT> const& facet_metric_distances() const
    {
        return _facet_metric_distances;
    }

    NT quadratic_statistic(Point const& x) const
    {
        if (x.dimension() != dimension())
            throw std::invalid_argument(
                "diagonal rounding point dimension does not match shape dimension");
        NT q = NT(0);
        for (unsigned int i = 0; i < dimension(); ++i) {
            NT const delta = x[i] - _center[i];
            q += delta * delta * _inv_shape_diag(i);
        }
        if (!std::isfinite(q))
            throw std::runtime_error(
                "diagonal rounding quadratic statistic is non-finite");
        return q;
    }

    NT centered_quadratic_statistic(Point const& centered_x) const
    {
        if (centered_x.dimension() != dimension())
            throw std::invalid_argument(
                "diagonal rounding point dimension does not match shape dimension");
        NT q = NT(0);
        for (unsigned int i = 0; i < dimension(); ++i) {
            NT const value = centered_x[i];
            q += value * value * _inv_shape_diag(i);
        }
        if (!std::isfinite(q))
            throw std::runtime_error(
                "diagonal rounding quadratic statistic is non-finite");
        return q;
    }

    NT gaussian_log_normalizer(NT a) const
    {
        if (!std::isfinite(a) || a <= NT(0))
            throw std::invalid_argument(
                "diagonal rounding Gaussian parameter must be finite and positive");
        static NT const pi = std::acos(NT(-1));
        NT const value = NT(dimension()) / NT(2) *
                             (std::log(pi) - std::log(a))
                       + NT(0.5) * _log_det_shape;
        if (!std::isfinite(value))
            throw std::runtime_error(
                "diagonal rounding Gaussian normalizer is non-finite");
        return value;
    }

    OrderPolytopeDiagonalRounding restrict_to(
        Poset const& residual,
        std::vector<unsigned int> const& original_vertices) const
    {
        if (residual.num_elem() != original_vertices.size())
            throw std::invalid_argument(
                "residual vertex map dimension does not match residual poset");
        VT center_coeffs(residual.num_elem());
        VT shape(residual.num_elem());
        for (unsigned int i = 0; i < residual.num_elem(); ++i) {
            unsigned int const original = original_vertices[i];
            if (original >= dimension())
                throw std::invalid_argument(
                    "residual vertex map contains an out-of-range index");
            center_coeffs(i) = _center[original];
            shape(i) = _shape_diag(original);
        }
        return OrderPolytopeDiagonalRounding(
            residual, Point(center_coeffs), shape, true);
    }

    static void reflect_cover_velocity(VT& velocity, unsigned int u,
                                       unsigned int v, VT const& shape_diag)
    {
        if (u >= static_cast<unsigned int>(velocity.size()) ||
            v >= static_cast<unsigned int>(velocity.size()) ||
            shape_diag.size() != velocity.size())
            throw std::invalid_argument(
                "diagonal rounding reflection dimension mismatch");
        NT const d_u = shape_diag(u), d_v = shape_diag(v);
        if (!std::isfinite(d_u) || !std::isfinite(d_v) ||
            d_u <= NT(0) || d_v <= NT(0))
            throw std::invalid_argument(
                "diagonal rounding reflection requires positive finite shape entries");
        NT const denom = d_u + d_v;
        if (!std::isfinite(denom) || denom <= NT(0))
            throw std::runtime_error(
                "diagonal rounding reflection denominator is invalid");
        NT const delta = velocity(u) - velocity(v);
        velocity(u) -= NT(2) * (d_u / denom) * delta;
        velocity(v) += NT(2) * (d_v / denom) * delta;
        if (!std::isfinite(velocity(u)) || !std::isfinite(velocity(v)))
            throw std::runtime_error(
                "diagonal rounding reflection produced non-finite velocity");
    }

private:
    void validate_and_precompute(OrderPolytope<Point> const& polytope,
                                 bool require_strict_feasibility)
    {
        unsigned int const n = polytope.dimension();
        if (_center.dimension() != n ||
            static_cast<unsigned int>(_shape_diag.size()) != n)
            throw std::invalid_argument(
                "diagonal rounding center/shape dimension mismatch");
        if (!_center.getCoefficients().allFinite())
            throw std::invalid_argument(
                "diagonal rounding center contains NaN or infinity");

        _inv_shape_diag.resize(n);
        _sqrt_shape_diag.resize(n);
        _log_det_shape = NT(0);
        for (unsigned int i = 0; i < n; ++i) {
            NT const d = _shape_diag(i);
            if (!std::isfinite(d) || d <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding shape entries must be finite and positive");
            NT const inv = NT(1) / d;
            NT const root = std::sqrt(d);
            NT const log_d = std::log(d);
            if (!std::isfinite(inv) || !std::isfinite(root) ||
                root <= NT(0) || !std::isfinite(log_d))
                throw std::invalid_argument(
                    "diagonal rounding shape produces unusable derived data");
            _inv_shape_diag(i) = inv;
            _sqrt_shape_diag(i) = root;
            _log_det_shape += log_d;
        }
        if (!std::isfinite(_log_det_shape))
            throw std::invalid_argument(
                "diagonal rounding log determinant is non-finite");

        VT const b = polytope.get_vec();
        NT const relation_scale = polytope.is_normalized()
            ? NT(1) / std::sqrt(NT(2)) : NT(1);
        _facet_metric_distances.clear();
        _facet_metric_distances.reserve(polytope.num_true_facets());

        auto add_distance = [&](NT slack, NT metric_normal_norm) {
            if (!std::isfinite(slack) || !std::isfinite(metric_normal_norm) ||
                metric_normal_norm <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding facet metric data is invalid");
            if (require_strict_feasibility && slack <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding center is not strictly feasible");
            NT const distance = slack / metric_normal_norm;
            if (!std::isfinite(distance) ||
                (require_strict_feasibility && distance <= NT(0)))
                throw std::invalid_argument(
                    "diagonal rounding facet distance is unusable");
            NT const squared_distance = distance * distance;
            NT const inverse_squared_distance = NT(1) / squared_distance;
            if (!std::isfinite(squared_distance) || squared_distance <= NT(0) ||
                !std::isfinite(inverse_squared_distance))
                throw std::invalid_argument(
                    "diagonal rounding facet distance has unusable scale");
            _facet_metric_distances.push_back(distance);
        };

        for (unsigned int i = 0; i < n; ++i)
            if (polytope.lower_bound_is_facet(i))
                add_distance(b(i) + _center[i], _sqrt_shape_diag(i));
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.upper_bound_is_facet(i))
                add_distance(b(n + i) - _center[i], _sqrt_shape_diag(i));
        for (unsigned int k = 0; k < polytope.num_order_relations(); ++k) {
            auto const relation = polytope.get_order_relation(k);
            NT const shape_sum = _shape_diag(relation.first)
                               + _shape_diag(relation.second);
            if (!std::isfinite(shape_sum) || shape_sum <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding cover shape sum is unusable");
            NT const slack = b(2 * n + k)
                - relation_scale * (_center[relation.first] - _center[relation.second]);
            add_distance(slack, relation_scale * std::sqrt(shape_sum));
        }
        if (n > 0 && _facet_metric_distances.empty())
            throw std::invalid_argument(
                "diagonal rounding requires at least one true facet");
    }

    Point _center;
    VT _shape_diag, _inv_shape_diag, _sqrt_shape_diag;
    NT _log_det_shape = NT(0);
    double _setup_microseconds = 0.0;
    std::vector<NT> _facet_metric_distances;
};

// A translated OrderPolytope carrying only diagonal data needed by the
// compile-time matched-mass HMC backend.  No dense affine transform is made.
template <typename Point, bool EnableDiagnostics = false>
class DiagonalRoundingOrderPolytope : private OrderPolytope<Point> {
public:
    typedef OrderPolytope<Point> Base;
    typedef typename Base::PointType PointType;
    typedef typename Point::FT NT;
    typedef typename Base::VT VT;
    typedef typename Base::MT MT;
    static constexpr bool rounding_diagnostics_enabled = EnableDiagnostics;

    using Base::dimension;
    using Base::get_order_relation;
    using Base::get_vec;
    using Base::is_in;
    using Base::is_normalized;
    using Base::lower_bound_is_facet;
    using Base::num_order_relations;
    using Base::num_true_facets;
    using Base::upper_bound_is_facet;

    DiagonalRoundingOrderPolytope(
        OrderPolytope<Point> const& polytope,
        OrderPolytopeDiagonalRounding<Point> const& metric,
        OrderPolytopeDiagonalCoolingDiagnostics<NT>* diagnostics = nullptr)
        : Base(polytope),
          _center(metric.center().getCoefficients()),
          _shape_diag(metric.shape_diag()),
          _sqrt_shape_diag(metric.sqrt_shape_diag()),
          _diagnostics(diagnostics)
    {
        if constexpr (EnableDiagnostics) {
            if (!_diagnostics)
                throw std::invalid_argument(
                    "diagonal rounding diagnostics-enabled body requires a sink");
        } else {
            _diagnostics = nullptr;
        }
        if (this->dimension() != metric.dimension())
            throw std::invalid_argument(
                "diagonal rounding polytope/metric dimension mismatch");
        // Revalidate against the actual body supplied to the walk.  A metric
        // constructed for a different same-dimensional poset must not reuse
        // stale true-facet distances or a center on an omitted cover facet.
        OrderPolytopeDiagonalRounding<Point> checked_metric(
            polytope, metric.center(), metric.shape_diag(), true);
        _facet_metric_distances = checked_metric.facet_metric_distances();
        if constexpr (EnableDiagnostics)
            _diagnostics->metric_setup_microseconds =
                checked_metric.setup_microseconds();
        this->normalize();
        this->shift(metric.center().getCoefficients());
        Point origin(this->dimension());
        if (this->is_in(origin) == 0)
            throw std::runtime_error(
                "diagonal rounding translated origin is infeasible");
    }

    DiagonalRoundingOrderPolytope(
        DiagonalRoundingOrderPolytope const&) = delete;
    DiagonalRoundingOrderPolytope& operator=(
        DiagonalRoundingOrderPolytope const&) = delete;
    DiagonalRoundingOrderPolytope(
        DiagonalRoundingOrderPolytope&&) = delete;
    DiagonalRoundingOrderPolytope& operator=(
        DiagonalRoundingOrderPolytope&&) = delete;

    VT const& rounding_shape_diag() const { return _shape_diag; }
    VT const& rounding_center() const { return _center; }
    VT const& rounding_sqrt_shape_diag() const { return _sqrt_shape_diag; }
    std::vector<NT> const& rounding_facet_metric_distances() const
    {
        return _facet_metric_distances;
    }
    OrderPolytopeDiagonalCoolingDiagnostics<NT>* rounding_diagnostics() const
    {
        return _diagnostics;
    }

private:
    // Cached metric distances and event rows are bound to the translated body.
    // Prevent ordinary callers from invalidating them after construction.
    using Base::ComputeInnerBall;
    using Base::linear_transformIt;
    using Base::normalize;
    using Base::shift;

    VT _center, _shape_diag, _sqrt_shape_diag;
    std::vector<NT> _facet_metric_distances;
    OrderPolytopeDiagonalCoolingDiagnostics<NT>* _diagnostics;
};

#endif // ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP
