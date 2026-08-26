// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_SPD_GAUSSIAN_EXACT_HMC_WALK_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_SPD_GAUSSIAN_EXACT_HMC_WALK_HPP

#include "convex_bodies/order_polytope_spd_rounding.hpp"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"

namespace order_polytope_exact_hmc_detail {

struct SPDDynamics
{
    static constexpr bool normalize_event_rows = true;
    static constexpr bool globally_coupled_reflection = true;

    template <typename Polytope>
    struct State : CenteredRoundingDynamicsState<Polytope>
    {
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;

        explicit State(Polytope const& polytope)
            : CenteredRoundingDynamicsState<Polytope>(polytope),
              _shape(&polytope.shape()),
              _shape_cholesky(&polytope.shape_cholesky())
        {}

        template <typename RandomNumberGenerator>
        inline Point draw_velocity(
            unsigned int n, RandomNumberGenerator& rng) const
        {
            Point standard = GetDirection<Point>::apply(n, rng, false);
            Point velocity(n);
            velocity.set_coeffs(
                (*_shape_cholesky) * standard.getCoefficients());
            return velocity;
        }

        template <typename Vector>
        inline void reflect_eta(unsigned int u, unsigned int w, NT c, NT s,
                                Vector& alpha, Vector& beta) const
        {
            bool const wall = u == w;
            NT const eta_u = -alpha(u) * s + beta(u) * c;
            NT const numerator = wall ? eta_u :
                eta_u - (-alpha(w) * s + beta(w) * c);
            NT const denominator = wall
                ? _shape_cholesky->row(u).squaredNorm()
                : (_shape_cholesky->row(u) -
                   _shape_cholesky->row(w)).squaredNorm();
            NT const scale = -NT(2) * numerator / denominator;
            for (Eigen::Index i = 0; i < alpha.size(); ++i) {
                NT const shape_normal = wall ? (*_shape)(i, u) :
                    (*_shape)(i, u) - (*_shape)(i, w);
                NT const delta = scale * shape_normal;
                alpha(i) = std::fma(-delta, s, alpha(i));
                beta(i) = std::fma(delta, c, beta(i));
            }
        }

        inline bool contact_normals_are_orthogonal(
            unsigned int i0, unsigned int i1,
            unsigned int j0, unsigned int j1) const
        {
            NT const terms[4] = {
                (*_shape)(i0, j0),
                j0 == j1 ? NT(0) : -(*_shape)(i0, j1),
                i0 == i1 ? NT(0) : -(*_shape)(i1, j0),
                i0 == i1 || j0 == j1 ? NT(0) : (*_shape)(i1, j1)};
            int sign = 0;
            return exact_component_sum(terms, sign) && sign == 0;
        }

    private:
        MT const* _shape;
        MT const* _shape_cholesky;
    };
};

} // namespace order_polytope_exact_hmc_detail

struct OrderPolytopeSPDExactHMCWalk
    : order_polytope_exact_hmc_detail::WalkPolicy<
          order_polytope_exact_hmc_detail::SPDDynamics>
{
    using Base = order_polytope_exact_hmc_detail::WalkPolicy<
        order_polytope_exact_hmc_detail::SPDDynamics>;
    using Base::Base;
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_SPD_GAUSSIAN_EXACT_HMC_WALK_HPP
