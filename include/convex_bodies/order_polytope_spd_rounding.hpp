// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ORDER_POLYTOPE_SPD_ROUNDING_HPP
#define ORDER_POLYTOPE_SPD_ROUNDING_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>

#include "convex_bodies/orderpolytope.h"

// Dense covariance-shape S for implicit rounding in the original
// OrderPolytope coordinates.  At cooling phase a, the target is
// exp(-a (x-c)^T S^{-1} (x-c)); its untruncated covariance is S/(2a).
template <typename Point>
class OrderPolytopeSPDRounding {
public:
    typedef typename Point::FT NT;
    typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
    typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;

    OrderPolytopeSPDRounding(Poset const& poset, Point const& center,
                             MT const& shape)
        : OrderPolytopeSPDRounding(
              OrderPolytope<Point>(poset), center, shape)
    {}

    OrderPolytopeSPDRounding(OrderPolytope<Point> const& polytope,
                             Point const& center, MT const& shape)
        : _center(center), _shape(shape)
    {
        unsigned int const n = polytope.dimension();
        if (_center.dimension() != n ||
            _shape.rows() != static_cast<Eigen::Index>(n) ||
            _shape.cols() != _shape.rows() ||
            !_center.getCoefficients().allFinite() || !_shape.allFinite() ||
            !(_shape.array() == _shape.transpose().array()).all())
            throw std::invalid_argument(
                "SPD rounding requires a finite center and symmetric shape "
                "of the body dimension");

        _diagonal = true;
        for (Eigen::Index i = 0; i < _shape.rows(); ++i)
            for (Eigen::Index j = 0; j < _shape.cols(); ++j)
                if (i != j && _shape(i, j) != NT(0)) _diagonal = false;

        _shape_cholesky = MT::Zero(n, n);
        if (_diagonal) {
            for (unsigned int i = 0; i < n; ++i) {
                if (_shape(i, i) <= NT(0))
                    throw std::invalid_argument(
                        "SPD rounding shape must be positive definite");
                _shape_cholesky(i, i) = std::sqrt(_shape(i, i));
            }
        } else if (n != 0) {
            Eigen::LLT<MT> factorization(_shape);
            if (factorization.info() != Eigen::Success)
                throw std::invalid_argument(
                    "SPD rounding shape must be positive definite");
            _shape_cholesky = factorization.matrixL();
        }
        for (unsigned int i = 0; i < n; ++i) {
            NT const diagonal = _shape_cholesky(i, i);
            if (!std::isfinite(diagonal) || diagonal <= NT(0))
                throw std::invalid_argument(
                    "SPD rounding Cholesky factor is unusable");
            _log_det_shape += NT(2) * std::log(diagonal);
        }
        _facet_metric_distances = metric_distances(polytope);
    }

    unsigned int dimension() const
    {
        return static_cast<unsigned int>(_shape.rows());
    }

    Point const& center() const { return _center; }
    MT const& shape() const { return _shape; }
    MT const& shape_cholesky() const { return _shape_cholesky; }
    std::vector<NT> const& facet_metric_distances() const
    {
        return _facet_metric_distances;
    }

    bool is_diagonal() const
    {
        return _diagonal;
    }

    void validate_for(Poset const& poset) const
    {
        if (poset.num_elem() != dimension())
            throw std::invalid_argument(
                "SPD rounding dimension does not match the poset");
        for (unsigned int i = 0; i < dimension(); ++i)
            if (!(_center[i] > NT(0) && _center[i] < NT(1)))
                throw std::invalid_argument(
                    "SPD rounding center must be strictly feasible");
        for (unsigned int k = 0; k < poset.num_relations(); ++k) {
            auto const relation = poset.get_relation(k);
            if (!(_center[relation.first] < _center[relation.second]))
                throw std::invalid_argument(
                    "SPD rounding center must be strictly feasible");
        }
    }

    NT centered_quadratic_statistic(Point const& centered_x) const
    {
        if (centered_x.dimension() != dimension())
            throw std::invalid_argument(
                "SPD rounding point dimension does not match shape dimension");
        VT whitened = _shape_cholesky.template triangularView<Eigen::Lower>()
                          .solve(centered_x.getCoefficients());
        NT const q = whitened.squaredNorm();
        if (!std::isfinite(q))
            throw std::runtime_error(
                "SPD rounding quadratic statistic is non-finite");
        return q;
    }

    NT gaussian_log_normalizer(NT a) const
    {
        if (!std::isfinite(a) || a <= NT(0))
            throw std::invalid_argument(
                "SPD rounding Gaussian parameter must be finite and positive");
        static NT const pi = std::acos(NT(-1));
        NT const value = NT(dimension()) / NT(2) *
                             (std::log(pi) - std::log(a))
                       + NT(0.5) * _log_det_shape;
        return value;
    }

    std::vector<NT> metric_distances(
        OrderPolytope<Point> const& polytope) const
    {
        if (!polytope.has_standard_order_facets())
            throw std::invalid_argument(
                "SPD rounding requires standard OrderPolytope facets");
        unsigned int const n = polytope.dimension();
        if (n != dimension())
            throw std::invalid_argument(
                "SPD rounding body dimension does not match shape dimension");
        std::vector<NT> distances;
        distances.reserve(polytope.num_true_facets());
        VT const b = polytope.get_vec();
        NT const relation_scale = polytope.is_normalized()
            ? NT(1) / std::sqrt(NT(2)) : NT(1);

        auto add_distance = [&](NT slack, NT metric_normal_norm) {
            NT const distance = slack / metric_normal_norm;
            if (!std::isfinite(distance) || distance <= NT(0))
                throw std::invalid_argument(
                    "SPD rounding center must be strictly feasible");
            distances.push_back(distance);
        };
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.lower_bound_is_facet(i))
                add_distance(b(i) + _center[i],
                             _shape_cholesky.row(i).norm());
        for (unsigned int i = 0; i < n; ++i)
            if (polytope.upper_bound_is_facet(i))
                add_distance(b(n + i) - _center[i],
                             _shape_cholesky.row(i).norm());
        for (unsigned int k : polytope.cover_relation_indices()) {
            auto const relation = polytope.get_order_relation(k);
            NT const slack = b(2 * n + k) - relation_scale *
                (_center[relation.first] - _center[relation.second]);
            NT const norm = relation_scale *
                (_shape_cholesky.row(relation.first) -
                 _shape_cholesky.row(relation.second)).norm();
            add_distance(slack, norm);
        }
        return distances;
    }

    OrderPolytopeSPDRounding restrict_to(
        Poset const& residual,
        std::vector<unsigned int> const& original_vertices) const
    {
        return restrict_to(OrderPolytope<Point>(residual), original_vertices);
    }

    OrderPolytopeSPDRounding restrict_to(
        OrderPolytope<Point> const& residual,
        std::vector<unsigned int> const& original_vertices) const
    {
        if (residual.dimension() != original_vertices.size())
            throw std::invalid_argument(
                "residual vertex map dimension does not match residual polytope");
        unsigned int const r = residual.dimension();
        VT center_coeffs(r);
        MT shape(r, r);
        for (unsigned int const original : original_vertices)
            if (original >= dimension())
                throw std::invalid_argument(
                    "residual vertex map contains an out-of-range index");
        for (unsigned int i = 0; i < r; ++i) {
            unsigned int const original_i = original_vertices[i];
            center_coeffs(i) = _center[original_i];
            for (unsigned int j = 0; j < r; ++j)
                shape(i, j) = _shape(original_i, original_vertices[j]);
        }
        return OrderPolytopeSPDRounding(
            residual, Point(center_coeffs), shape);
    }

private:
    Point _center;
    MT _shape, _shape_cholesky;
    NT _log_det_shape = NT(0);
    bool _diagonal = false;
    std::vector<NT> _facet_metric_distances;
};

namespace order_polytope_rounding_detail {

template <typename Point>
class SPDBody : private OrderPolytope<Point> {
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

    SPDBody(
        OrderPolytope<Point> const& polytope,
        OrderPolytopeSPDRounding<Point> const& metric)
        : Base(polytope),
          _center(metric.center().getCoefficients()),
          _shape(metric.shape()),
          _shape_cholesky(metric.shape_cholesky()),
          _facet_metric_distances(metric.metric_distances(polytope))
    {
        VT const offsets = polytope.get_vec();
        unsigned int const n = polytope.dimension();
        for (unsigned int i = 0; i < n; ++i)
            if (offsets(i) != NT(0) || offsets(n + i) != NT(1))
                throw std::invalid_argument(
                    "SPD rounding walk requires a canonical unshifted OrderPolytope");
        this->normalize();
        this->shift(_center);
    }

    SPDBody(SPDBody const&) = delete;
    SPDBody& operator=(SPDBody const&) = delete;

    VT const& center() const { return _center; }
    MT const& shape() const { return _shape; }
    MT const& shape_cholesky() const { return _shape_cholesky; }
    std::vector<NT> const& facet_metric_distances() const
    {
        return _facet_metric_distances;
    }

private:
    VT _center;
    MT _shape, _shape_cholesky;
    std::vector<NT> _facet_metric_distances;
};

} // namespace order_polytope_rounding_detail

#endif // ORDER_POLYTOPE_SPD_ROUNDING_HPP
