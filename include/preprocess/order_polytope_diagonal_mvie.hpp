// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ORDER_POLYTOPE_DIAGONAL_MVIE_HPP
#define ORDER_POLYTOPE_DIAGONAL_MVIE_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>

#include "convex_bodies/order_polytope_diagonal_rounding.hpp"

namespace order_mvie_detail {

template <typename NT>
using Vector = Eigen::Matrix<NT, Eigen::Dynamic, 1>;

template <typename NT>
using Matrix = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>;

struct Facets {
    unsigned int dimension;
    std::vector<unsigned char> lower, upper;
    Poset::RV covers;
    unsigned int barrier_parameter;
};

template <typename Point>
Facets true_facets(OrderPolytope<Point> const& polytope)
{
    typedef typename Point::FT NT;
    unsigned int const n = polytope.dimension();
    typename OrderPolytope<Point>::VT const b = polytope.get_vec();
    bool canonical = polytope.has_standard_order_facets();
    for (unsigned int i = 0; i < n; ++i)
        canonical = canonical && b(i) == NT(0);
    if (!canonical)
        throw std::invalid_argument(
            "diagonal MVIE requires a canonical unshifted OrderPolytope");

    Facets facets{n, std::vector<unsigned char>(n),
                  std::vector<unsigned char>(n), {}, 0};
    for (unsigned int i = 0; i < n; ++i) {
        facets.lower[i] = polytope.lower_bound_is_facet(i);
        facets.upper[i] = polytope.upper_bound_is_facet(i);
        facets.barrier_parameter += facets.lower[i] + facets.upper[i];
    }
    facets.covers.reserve(polytope.cover_relation_indices().size());
    for (unsigned int relation_index : polytope.cover_relation_indices())
        facets.covers.push_back(
            polytope.get_order_relation(relation_index));
    facets.barrier_parameter += 2 * facets.covers.size();
    return facets;
}

template <typename NT>
NT objective(Facets const& facets, Vector<NT> const& x, NT mu)
{
    unsigned int const n = facets.dimension;
    NT value = NT(0);
    for (unsigned int i = 0; i < n; ++i) {
        NT const c = x(i), axis = x(n + i);
        if (!(axis > NT(0))) return std::numeric_limits<NT>::infinity();
        value -= std::log(axis);
        if (facets.lower[i]) {
            NT const slack = c - axis;
            if (!(slack > NT(0)))
                return std::numeric_limits<NT>::infinity();
            value -= mu * std::log(slack);
        }
        if (facets.upper[i]) {
            NT const slack = NT(1) - c - axis;
            if (!(slack > NT(0)))
                return std::numeric_limits<NT>::infinity();
            value -= mu * std::log(slack);
        }
    }
    for (Poset::RT const& cover : facets.covers) {
        NT const delta = x(cover.second) - x(cover.first);
        NT const radius = std::hypot(
            x(n + cover.first), x(n + cover.second));
        NT const minus = delta - radius;
        if (!(minus > NT(0)))
            return std::numeric_limits<NT>::infinity();
        value -= mu * (std::log(minus) + std::log(delta + radius));
    }
    return value;
}

template <typename NT>
void derivatives(Facets const& facets, Vector<NT> const& x, NT mu,
                 Vector<NT>& gradient, Matrix<NT>& hessian)
{
    unsigned int const n = facets.dimension;
    gradient = Vector<NT>::Zero(2 * n);
    hessian = Matrix<NT>::Zero(2 * n, 2 * n);

    for (unsigned int i = 0; i < n; ++i) {
        unsigned int const ci = i, ai = n + i;
        NT const c = x(ci), axis = x(ai);
        gradient(ai) -= NT(1) / axis;
        hessian(ai, ai) += NT(1) / (axis * axis);

        auto add_linear_barrier = [&](NT slack, NT dc, NT da) {
            NT const first = mu / slack, second = first / slack;
            gradient(ci) -= first * dc;
            gradient(ai) -= first * da;
            hessian(ci, ci) += second * dc * dc;
            hessian(ai, ai) += second * da * da;
            hessian(ci, ai) += second * dc * da;
            hessian(ai, ci) += second * dc * da;
        };
        if (facets.lower[i])
            add_linear_barrier(c - axis, NT(1), NT(-1));
        if (facets.upper[i])
            add_linear_barrier(NT(1) - c - axis, NT(-1), NT(-1));
    }

    for (Poset::RT const& cover : facets.covers) {
        unsigned int const u = cover.first, v = cover.second;
        unsigned int const au = n + u, av = n + v;
        NT const delta = x(v) - x(u);
        NT const axis_u = x(au), axis_v = x(av);
        NT const radius = std::hypot(axis_u, axis_v);
        NT const minus = delta - radius, plus = delta + radius;
        NT const q = minus * plus;

        unsigned int const indices[4] = {u, v, au, av};
        NT const dq[4] = {
            -NT(2) * delta, NT(2) * delta,
            -NT(2) * axis_u, -NT(2) * axis_v};
        NT const first = mu / q, second = first / q;
        for (unsigned int i = 0; i < 4; ++i) {
            gradient(indices[i]) -= first * dq[i];
            for (unsigned int j = 0; j < 4; ++j)
                hessian(indices[i], indices[j]) +=
                    second * dq[i] * dq[j];
        }
        hessian(u, u) -= NT(2) * first;
        hessian(v, v) -= NT(2) * first;
        hessian(u, v) += NT(2) * first;
        hessian(v, u) += NT(2) * first;
        hessian(au, au) += NT(2) * first;
        hessian(av, av) += NT(2) * first;
    }
}

template <typename NT>
Vector<NT> strictly_feasible_start(Facets const& facets)
{
    unsigned int const n = facets.dimension;
    Poset::RV covers = facets.covers;
    Poset poset(n, covers);
    std::vector<unsigned int> const order = poset.topologically_sorted_list();
    Vector<NT> x(2 * n);
    for (unsigned int rank = 0; rank < n; ++rank)
        x(order[rank]) = NT(rank + 1) / NT(n + 1);
    x.tail(n).setConstant(NT(1) / (NT(4) * NT(n + 1)));
    return x;
}

template <typename NT>
Vector<NT> solve(Facets const& facets)
{
    unsigned int const n = facets.dimension;
    Vector<NT> x = strictly_feasible_start<NT>(facets), gradient, direction;
    Matrix<NT> hessian;
    NT mu = NT(1);

    for (unsigned int outer = 0; outer < 12; ++outer) {
        bool centered = false;
        for (unsigned int inner = 0; inner < 80; ++inner) {
            NT const value = objective(facets, x, mu);
            derivatives(facets, x, mu, gradient, hessian);
            Vector<NT> const scale =
                hessian.diagonal().array().sqrt().inverse().matrix();
            if (!scale.allFinite())
                throw std::runtime_error("diagonal MVIE Newton system failed");
            Matrix<NT> const scaled_hessian =
                scale.asDiagonal() * hessian * scale.asDiagonal();
            Eigen::LDLT<Matrix<NT> > factorization(scaled_hessian);
            if (factorization.info() != Eigen::Success ||
                !factorization.vectorD().allFinite() ||
                !(factorization.vectorD().minCoeff() > NT(0)))
                throw std::runtime_error("diagonal MVIE Newton system failed");
            direction = scale.asDiagonal() * factorization.solve(
                (-scale.array() * gradient.array()).matrix());
            NT const decrement2 = -gradient.dot(direction);
            if (!std::isfinite(decrement2) || decrement2 < NT(0))
                throw std::runtime_error("diagonal MVIE Newton step failed");
            if (decrement2 / NT(2) <= NT(1e-9)) {
                centered = true;
                break;
            }
            NT step = NT(1);
            while (step > NT(1e-16)) {
                NT const candidate = objective(
                    facets, (x + step * direction).eval(), mu);
                if (std::isfinite(candidate) &&
                    candidate <= value + NT(0.01) * step *
                                             gradient.dot(direction))
                    break;
                step *= NT(0.5);
            }
            if (!(step > NT(1e-16)))
                throw std::runtime_error("diagonal MVIE line search failed");
            x += step * direction;
        }
        if (!centered)
            throw std::runtime_error("diagonal MVIE did not converge");
        NT const barrier_gap = facets.barrier_parameter * mu;
        if (barrier_gap / NT(n) <= NT(1e-6)) return x;
        mu *= NT(0.1);
    }
    throw std::runtime_error("diagonal MVIE did not converge");
}

} // namespace order_mvie_detail

template <typename Point>
OrderPolytopeDiagonalRounding<Point>
max_inscribed_diagonal_ellipsoid(OrderPolytope<Point> const& polytope)
{
    typedef typename Point::FT NT;
    typedef typename OrderPolytope<Point>::VT VT;
    unsigned int const n = polytope.dimension();
    if (n == 0)
        return OrderPolytopeDiagonalRounding<Point>(
            polytope, Point(0), VT(0));
    auto const facets =
        order_mvie_detail::true_facets(polytope);
    VT const solution =
        order_mvie_detail::solve<NT>(facets);
    VT const center = solution.head(n), axes = solution.tail(n);
    return OrderPolytopeDiagonalRounding<Point>(
        polytope, Point(center), axes.array().square().matrix());
}

template <typename Point>
OrderPolytopeDiagonalRounding<Point>
diagonal_mvie_rounding(
    OrderPolytope<Point> const& polytope)
{
    typedef typename Point::FT NT;
    OrderPolytopeDiagonalRounding<Point> const ellipsoid =
        max_inscribed_diagonal_ellipsoid(polytope);
    typename OrderPolytope<Point>::VT shape = ellipsoid.shape_diag();
    if (shape.size() != 0) {
        NT const scale = std::exp(
            shape.array().log().sum() / NT(shape.size()));
        shape /= scale;
    }
    return OrderPolytopeDiagonalRounding<Point>(
        polytope, ellipsoid.center(), shape);
}

#endif // ORDER_POLYTOPE_DIAGONAL_MVIE_HPP
