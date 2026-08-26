// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_ORDER_POLYTOPE_GENERAL_MASS_COOLING_HPP
#define VOLUME_ORDER_POLYTOPE_GENERAL_MASS_COOLING_HPP

#include <stdexcept>

#include <Eigen/Eigen>

#include "convex_bodies/orderpolytope.h"
#include "random_walks/order_polytope_general_gaussian_hmc_walk.hpp"
#include "volume/volume_cooling_gaussians_log.hpp"

namespace order_mass_cooling_detail {

static constexpr unsigned long long default_max_samples = 10000000ULL;

// The generic cooling driver copies, normalizes, and translates its body.
// This wrapper carries the user's fixed momentum mass through those operations;
// translation does not change the coordinate basis or the mass matrix.
template <typename Point>
class Body : public OrderPolytope<Point>
{
public:
    typedef OrderPolytope<Point> Base;
    typedef typename Base::MT MT;

    Body(Base const& body, MT const& mass)
        : Base(body), _mass(mass)
    {
        if (!body.has_standard_order_facets())
            throw std::invalid_argument(
                "general-mass cooling requires standard OrderPolytope facets");
        unsigned int const n = body.dimension();
        if (_mass.rows() != static_cast<Eigen::Index>(n) ||
            _mass.cols() != _mass.rows())
            throw std::invalid_argument(
                "general-mass cooling: mass dimension mismatch");
        if (!_mass.allFinite() ||
            !(_mass.array() == _mass.transpose().array()).all())
            throw std::invalid_argument(
                "general-mass cooling: mass must be finite and exactly "
                "symmetric");
        if (n != 0) {
            Eigen::LLT<MT> factorization(_mass);
            if (factorization.info() != Eigen::Success)
                throw std::invalid_argument(
                    "general-mass cooling: mass is not SPD");
        }
    }

    MT const& general_mass() const { return _mass; }

private:
    MT _mass;
};

struct WalkPolicy
{
    template <typename Polytope, typename RandomNumberGenerator>
    class Walk
    {
    public:
        typedef typename Polytope::PointType Point;
        typedef typename Point::FT NT;
        typedef typename Polytope::VT VT;
        typedef typename Polytope::MT MT;
        typedef OrderPolytopeGeneralGaussianHMCWalk Policy;
        typedef typename Policy::template Walk<
            Polytope, RandomNumberGenerator> InnerWalk;

        Walk(Polytope& polytope, Point const& start, NT current_a,
             RandomNumberGenerator& rng)
            : _inner(polytope, start,
                     make_parameters(polytope, current_a), rng)
        {}

        void apply(Polytope const& polytope, Point& point,
                   NT const& /*current_a*/, unsigned int walk_length,
                   RandomNumberGenerator& rng)
        {
            _inner.apply(polytope, point, walk_length, rng);
        }

    private:
        static typename Policy::template parameters<NT>
        make_parameters(Polytope const& polytope, NT current_a)
        {
            unsigned int const n = polytope.dimension();
            MT const precision =
                NT(2) * current_a * MT::Identity(n, n);
            return typename Policy::template parameters<NT>(
                precision, polytope.general_mass(), VT::Zero(n));
        }

        InnerWalk _inner;
    };
};

} // namespace order_mass_cooling_detail

// Standard spherical Gaussian cooling (A_i = 2 a_i I) using one fixed,
// caller-supplied SPD momentum mass M at every stage. M changes only the HMC
// dynamics; it does not change the cooling densities or volume normalizer.
// As with volume_cooling_gaussians_log, success reports completion within the
// numerical/sample guards, not a deterministic accuracy certificate.
template <typename Point, typename RandomNumberGenerator>
GaussianCoolingLogVolume<typename Point::FT>
volume_cooling_gaussians_log_mass(
    OrderPolytope<Point> const& input_polytope,
    RandomNumberGenerator& rng,
    typename OrderPolytope<Point>::MT const& mass,
    double error = 0.1,
    unsigned int walk_length = 1,
    unsigned long long max_samples =
        order_mass_cooling_detail::default_max_samples)
{
    typedef order_mass_cooling_detail::
        Body<Point> Polytope;
    Polytope body(input_polytope, mass);
    return volume_cooling_gaussians_log<
        order_mass_cooling_detail::WalkPolicy>(
            body, rng, error, walk_length, max_samples);
}

#endif // VOLUME_ORDER_POLYTOPE_GENERAL_MASS_COOLING_HPP
