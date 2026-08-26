// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP
#define ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include "convex_bodies/orderpolytope.h"

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
                                  VT const& shape_diag)
        : OrderPolytopeDiagonalRounding(
              OrderPolytope<Point>(poset), center, shape_diag)
    {}

    OrderPolytopeDiagonalRounding(OrderPolytope<Point> const& polytope,
                                  Point const& center,
                                  VT const& shape_diag)
        : _center(center), _shape_diag(shape_diag)
    {
        validate_and_precompute(polytope);
    }

    unsigned int dimension() const
    {
        return static_cast<unsigned int>(_shape_diag.size());
    }

    Point const& center() const { return _center; }
    VT const& shape_diag() const { return _shape_diag; }
    VT const& sqrt_shape_diag() const { return _sqrt_shape_diag; }
    std::vector<NT> const& facet_metric_distances() const
    {
        return _facet_metric_distances;
    }

    NT centered_quadratic_statistic(Point const& centered_x) const
    {
        if (centered_x.dimension() != dimension())
            throw std::invalid_argument(
                "diagonal rounding point dimension does not match shape dimension");
        NT q = NT(0);
        for (unsigned int i = 0; i < dimension(); ++i)
            q += centered_x[i] * centered_x[i] * _inv_shape_diag(i);
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
        return value;
    }

    OrderPolytopeDiagonalRounding restrict_to(
        Poset const& residual,
        std::vector<unsigned int> const& original_vertices) const
    {
        return restrict_to(
            OrderPolytope<Point>(residual), original_vertices);
    }

    OrderPolytopeDiagonalRounding restrict_to(
        OrderPolytope<Point> const& residual,
        std::vector<unsigned int> const& original_vertices) const
    {
        if (residual.dimension() != original_vertices.size())
            throw std::invalid_argument(
                "residual vertex map dimension does not match residual polytope");
        VT center_coeffs(residual.dimension());
        VT shape(residual.dimension());
        for (unsigned int i = 0; i < residual.dimension(); ++i) {
            unsigned int const original = original_vertices[i];
            if (original >= dimension())
                throw std::invalid_argument(
                    "residual vertex map contains an out-of-range index");
            center_coeffs(i) = _center[original];
            shape(i) = _shape_diag(original);
        }
        return OrderPolytopeDiagonalRounding(
            residual, Point(center_coeffs), shape);
    }

private:
    void validate_and_precompute(OrderPolytope<Point> const& polytope)
    {
        unsigned int const n = polytope.dimension();
        if (!polytope.has_standard_order_facets())
            throw std::invalid_argument(
                "diagonal rounding requires standard OrderPolytope facets");
        if (_center.dimension() != n ||
            static_cast<unsigned int>(_shape_diag.size()) != n)
            throw std::invalid_argument(
                "diagonal rounding center/shape dimension mismatch");
        if (!_center.getCoefficients().allFinite())
            throw std::invalid_argument(
                "diagonal rounding center contains NaN or infinity");

        _inv_shape_diag.resize(n);
        _sqrt_shape_diag.resize(n);
        for (unsigned int i = 0; i < n; ++i) {
            NT const d = _shape_diag(i);
            if (!std::isfinite(d) || d <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding shape entries must be finite and positive");
            NT const inv = NT(1) / d;
            NT const root = std::sqrt(d);
            NT const log_d = std::log(d);
            if (!std::isfinite(inv))
                throw std::invalid_argument(
                    "diagonal rounding shape produces unusable derived data");
            _inv_shape_diag(i) = inv;
            _sqrt_shape_diag(i) = root;
            _log_det_shape += log_d;
        }
        VT const b = polytope.get_vec();
        NT const relation_scale = polytope.is_normalized()
            ? NT(1) / std::sqrt(NT(2)) : NT(1);
        _facet_metric_distances.reserve(polytope.num_true_facets());

        auto add_distance = [&](NT slack, NT metric_normal_norm) {
            NT const distance = slack / metric_normal_norm;
            if (!std::isfinite(distance) || distance <= NT(0))
                throw std::invalid_argument(
                    "diagonal rounding facet distance is unusable");
            _facet_metric_distances.push_back(distance);
        };

        for (unsigned int i = 0; i < n; ++i)
            if (polytope.lower_bound_is_facet(i))
                add_distance(b(i) + _center[i], _sqrt_shape_diag(i));
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.upper_bound_is_facet(i))
                add_distance(b(n + i) - _center[i], _sqrt_shape_diag(i));
        for (unsigned int k : polytope.cover_relation_indices()) {
            auto const relation = polytope.get_order_relation(k);
            NT const shape_sum = _shape_diag(relation.first)
                               + _shape_diag(relation.second);
            NT const slack = b(2 * n + k)
                - relation_scale * (_center[relation.first] - _center[relation.second]);
            add_distance(slack, relation_scale * std::sqrt(shape_sum));
        }
    }

    Point _center;
    VT _shape_diag, _inv_shape_diag, _sqrt_shape_diag;
    NT _log_det_shape = NT(0);
    std::vector<NT> _facet_metric_distances;
};

namespace order_polytope_rounding_detail {

template <typename Point>
class DiagonalBody : private OrderPolytope<Point> {
public:
    typedef OrderPolytope<Point> Base;
    typedef typename Base::PointType PointType;
    typedef typename Point::FT NT;
    typedef typename Base::VT VT;
    typedef typename Base::MT MT;
    using Base::cover_relation_indices;
    using Base::dimension;
    using Base::get_order_relation;
    using Base::get_vec;
    using Base::geometry_revision;
    using Base::has_standard_order_facets;
    using Base::is_in;
    using Base::is_normalized;
    using Base::lower_bound_is_facet;
    using Base::num_order_relations;
    using Base::upper_bound_is_facet;

    DiagonalBody(
        OrderPolytope<Point> const& polytope,
        OrderPolytopeDiagonalRounding<Point> const& metric)
        : Base(polytope),
          _center(metric.center().getCoefficients()),
          _shape_diag(metric.shape_diag()),
          _sqrt_shape_diag(metric.sqrt_shape_diag())
    {
        VT const source_offsets = polytope.get_vec();
        unsigned int const n = polytope.dimension();
        for (unsigned int i = 0; i < n; ++i)
            if (source_offsets(i) != NT(0) ||
                source_offsets(n + i) != NT(1))
                throw std::invalid_argument(
                    "diagonal rounding walk requires a canonical unshifted OrderPolytope");
        // Revalidate the center and facet distances against the walk body.
        OrderPolytopeDiagonalRounding<Point> checked_metric(
            polytope, metric.center(), metric.shape_diag());
        _facet_metric_distances = checked_metric.facet_metric_distances();
        this->normalize();
        this->shift(_center);
    }

    DiagonalBody(DiagonalBody const&) = delete;
    DiagonalBody& operator=(DiagonalBody const&) = delete;

    VT const& shape_diag() const { return _shape_diag; }
    VT const& center() const { return _center; }
    VT const& sqrt_shape_diag() const { return _sqrt_shape_diag; }
    std::vector<NT> const& facet_metric_distances() const
    {
        return _facet_metric_distances;
    }
private:
    VT _center, _shape_diag, _sqrt_shape_diag;
    std::vector<NT> _facet_metric_distances;
};

} // namespace order_polytope_rounding_detail

#endif // ORDER_POLYTOPE_DIAGONAL_ROUNDING_HPP
