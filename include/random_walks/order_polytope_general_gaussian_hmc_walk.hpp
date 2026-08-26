// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef RANDOM_WALKS_ORDER_POLYTOPE_GENERAL_GAUSSIAN_HMC_WALK_HPP
#define RANDOM_WALKS_ORDER_POLYTOPE_GENERAL_GAUSSIAN_HMC_WALK_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Eigen>

#include "random_walks/order_polytope_hmc_modal.hpp"
#include "sampling/sphere.hpp"

// Reflected HMC for
//
//   pi(x) proportional to exp(-1/2 (x-c)^T A (x-c)) 1{x in P},
//
// where both the precision A and momentum mass M are arbitrary SPD matrices.
// Multi-frequency trajectories are analytic in modal coordinates. Boundary
// isolation uses outward-inflated floating bounds and fails closed; this is a
// safeguarded floating earliest-hit walk, not an exact-arithmetic sampler.
struct OrderPolytopeGeneralGaussianHMCWalk
{
    template <typename NT>
    struct parameters
    {
        typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
        typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;

        parameters(MT const& precision, MT const& mass,
                   VT const& gaussian_center)
            : A(precision), M(mass), center(gaussian_center)
        {}

        MT A;
        MT M;
        VT center;
    };

    template <typename Polytope, typename RandomNumberGenerator>
    class Walk
    {
    public:
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;
        typedef order_gaussian_hmc_detail::
            ModalTrajectory<Polytope> Trajectory;

        Walk(Polytope& polytope, Point const& start,
             parameters<NT> const& params, RandomNumberGenerator& /*rng*/)
            : _body(&polytope),
              _geometry_revision(polytope.geometry_revision()),
              _trajectory(polytope, params.A, params.M, params.center),
              _trajectory_length(NT(1))
        {
            unsigned int const n = polytope.dimension();
            if (start.dimension() != n ||
                !start.getCoefficients().allFinite())
                throw std::invalid_argument(
                    "general Gaussian HMC: invalid start point");
            _max_reflections = 100U * n;
            if (polytope.is_in(start, NT(0)) != -1)
                throw std::invalid_argument(
                    "general Gaussian HMC: start point is outside the body");
        }

        void apply(Polytope const& polytope, Point& point,
                   unsigned int walk_length,
                   RandomNumberGenerator& rng)
        {
            if (_body != &polytope ||
                _geometry_revision != polytope.geometry_revision() ||
                !polytope.has_standard_order_facets() ||
                point.dimension() != polytope.dimension() ||
                !point.getCoefficients().allFinite())
                throw std::invalid_argument(
                    "general Gaussian HMC: body or point mismatch");
            if (polytope.is_in(point, NT(0)) != -1)
                throw std::invalid_argument(
                    "general Gaussian HMC: point is outside the body");

            for (unsigned int step = 0; step < walk_length; ++step) {
                Point const standard_normal = GetDirection<Point>::apply(
                    _body->dimension(), rng, false);
                VT const velocity =
                    _trajectory.velocity_from_standard_normal(
                        standard_normal.getCoefficients());
                NT const time =
                    NT(rng.sample_urdist()) * _trajectory_length;
                if (!advance(point, velocity, time))
                    throw std::runtime_error(
                        "general Gaussian HMC failed to complete a trajectory");
            }
        }

        void update_delta(NT trajectory_length)
        {
            if (!std::isfinite(trajectory_length) ||
                trajectory_length < NT(0))
                throw std::invalid_argument(
                    "general Gaussian HMC: trajectory length must be finite "
                    "and nonnegative");
            _trajectory_length = trajectory_length;
        }

    private:
        bool advance(Point& point, VT velocity, NT time)
        {
            VT position = point.getCoefficients();
            NT remaining = time;
            unsigned int reflection_count = 0;
            int last_hit_facet = -1;
            while (remaining > NT(0)) {
                _trajectory.set_state(position, velocity);
                auto const hit =
                    order_gaussian_hmc_detail::first_hit(
                        _trajectory, remaining, last_hit_facet);

                if (hit.tied_facets.size() > 1) {
                    // Sequential metric reflections commute only when all
                    // contact normals are pairwise M^{-1}-orthogonal.
                    for (unsigned int i = 0;
                         i < hit.tied_facets.size(); ++i) {
                        int const first = hit.tied_facets[i];
                        for (unsigned int j = i + 1;
                             j < hit.tied_facets.size(); ++j) {
                            int const second = hit.tied_facets[j];
                            VT const inverse_mass_second =
                                _trajectory.inverse_mass_normal(
                                    static_cast<unsigned int>(second));
                            NT const coupling = _trajectory.normal_dot(
                                static_cast<unsigned int>(first),
                                inverse_mass_second);
                            if (coupling != NT(0))
                                return false;
                        }
                    }
                }

                if (hit.facet < 0) {
                    position = _trajectory.position(remaining);
                    remaining = NT(0);
                    break;
                }
                if (reflection_count >= _max_reflections)
                    return false;

                position = _trajectory.position(hit.time);
                velocity = _trajectory.velocity(hit.time);
                unsigned int const facet =
                    static_cast<unsigned int>(hit.facet);
                VT const inverse_mass_normal =
                    _trajectory.inverse_mass_normal(facet);
                NT const denominator = _trajectory.normal_dot(
                    facet, inverse_mass_normal);
                NT const normal_velocity =
                    _trajectory.normal_dot(facet, velocity);
                if (!std::isfinite(denominator) || denominator <= NT(0) ||
                    !std::isfinite(normal_velocity))
                    return false;
                velocity.noalias() -=
                    (NT(2) * normal_velocity / denominator) *
                    inverse_mass_normal;
                if (!velocity.allFinite()) return false;

                last_hit_facet = hit.facet;
                remaining -= hit.time;
                ++reflection_count;
            }

            Point endpoint(position);
            if (_body->is_in(endpoint, NT(0)) != -1)
                return false;
            point = endpoint;
            return true;
        }

        Polytope const* _body;
        unsigned long long _geometry_revision;
        Trajectory _trajectory;
        NT _trajectory_length;
        unsigned int _max_reflections;
    };
};

#endif // RANDOM_WALKS_ORDER_POLYTOPE_GENERAL_GAUSSIAN_HMC_WALK_HPP
