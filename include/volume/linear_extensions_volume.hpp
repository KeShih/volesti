// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis

// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
#define VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include "convex_bodies/orderpolytope.h"
#include "misc/order_polytope_volume_reduction.h"
#include "misc/poset.h"
#include "preprocess/order_polytope_diagonal_mvie.hpp"
#include "random_walks/order_polytope_gaussian_hmc_exact_walk.hpp"
#include "volume/order_polytope_diagonal_cooling.hpp"
#include "volume/order_polytope_general_mass_cooling.hpp"
#include "volume/order_polytope_spd_cooling.hpp"
#include "volume/volume_cooling_gaussians_log.hpp"


struct LinearExtensionOptions
{
    double error = 0.1;
    unsigned int walk_length = 1;
    unsigned int repetitions = 1;
    OrderPolytopeVolumeReductionOptions reduction;
};

template <typename NT>
struct LinearExtensionEstimate {
    NT volume_estimate = NT(0);
    NT log_volume = NT(0);
    NT se_log_volume = NT(0);
    NT log_extensions = NT(0);
    unsigned int repetitions = 1;
    bool reduction_exact = false;
    unsigned int residual_count = 0;
};

namespace linear_extensions_detail {

inline void validate_options(LinearExtensionOptions const& options)
{
    if (!std::isfinite(options.error) || options.error <= 0.0 ||
        options.walk_length == 0 || options.repetitions == 0)
        throw std::invalid_argument(
            "linear-extension options require positive error, walk length, "
            "and repetitions");
}

template <typename NT, typename ResidualEstimator>
LinearExtensionEstimate<NT> estimate(
    Poset const& poset,
    LinearExtensionOptions const& options,
    ResidualEstimator&& estimate_residual)
{
    validate_options(options);
    ReducedOrderPolytopeVolumeProblem const reduced =
        reduce_order_polytope_volume_problem(poset, options.reduction);
    LinearExtensionEstimate<NT> out;
    out.repetitions = options.repetitions;
    out.reduction_exact = reduced.is_exact();
    out.residual_count = static_cast<unsigned int>(
        reduced.residual_posets.size());

    double const residual_error = out.residual_count > 1
        ? options.error / std::sqrt(double(out.residual_count))
        : options.error;

    std::vector<NT> log_vols;
    log_vols.reserve(out.repetitions);
    for (unsigned int repetition = 0;
         repetition < out.repetitions; ++repetition) {
        NT log_volume = static_cast<NT>(reduced.log_volume_offset);
        for (unsigned int residual_index = 0;
             residual_index < out.residual_count; ++residual_index) {
            GaussianCoolingLogVolume<NT> const residual = estimate_residual(
                residual_index,
                reduced.residual_posets[residual_index],
                reduced.residual_vertices[residual_index], residual_error);
            if (!residual.success)
                throw std::runtime_error(
                    "linear-extension volume estimation failed");
            log_volume += residual.log_volume;
        }
        if (!std::isfinite(log_volume))
            throw std::runtime_error(
                "linear-extension log volume is non-finite");
        log_vols.push_back(log_volume);
    }

    NT cv2;
    if (!volume_cooling_gaussians_log_detail::log_weight_moments(
            log_vols, out.log_volume, cv2))
        throw std::runtime_error(
            "linear-extension repetition aggregation failed");
    out.se_log_volume = out.repetitions > 1
        ? std::sqrt(cv2 / NT(out.repetitions - 1)) : NT(0);
    out.volume_estimate = std::exp(out.log_volume);
    out.log_extensions = out.log_volume +
        std::lgamma(NT(poset.num_elem()) + NT(1));
    return out;
}

} // namespace linear_extensions_detail

template <typename RandomNumberGenerator, typename Point>
LinearExtensionEstimate<typename Point::FT> estimate_log_linear_extensions(
    Poset const& poset,
    RandomNumberGenerator& rng,
    LinearExtensionOptions const& options = LinearExtensionOptions())
{
    return linear_extensions_detail::estimate<typename Point::FT>(
        poset, options,
        [&](unsigned int, Poset const& residual,
            std::vector<unsigned int> const&,
            double residual_error) {
            OrderPolytope<Point> body =
                OrderPolytope<Point>::from_reduced_relations(residual);
            return volume_cooling_gaussians_log<
                OrderPolytopeExactHMCWalk>(
                    body, rng, residual_error, options.walk_length);
        });
}

template <typename RandomNumberGenerator, typename Point>
LinearExtensionEstimate<typename Point::FT>
estimate_log_linear_extensions(
    Poset const& poset,
    RandomNumberGenerator& rng,
    OrderPolytopeDiagonalRounding<Point> const& metric,
    LinearExtensionOptions const& options = LinearExtensionOptions())
{
    OrderPolytopeDiagonalRounding<Point> const checked_metric(
        poset, metric.center(), metric.shape_diag());
    return linear_extensions_detail::estimate<typename Point::FT>(
        poset, options,
        [&](unsigned int, Poset const& residual,
            std::vector<unsigned int> const& original_vertices,
            double residual_error) {
            OrderPolytope<Point> body =
                OrderPolytope<Point>::from_reduced_relations(residual);
            OrderPolytopeDiagonalRounding<Point> const residual_metric =
                checked_metric.restrict_to(body, original_vertices);
            return volume_cooling_gaussians_log(
                body, rng, residual_metric, residual_error,
                options.walk_length);
        });
}

template <typename RandomNumberGenerator, typename Point>
LinearExtensionEstimate<typename Point::FT>
estimate_log_linear_extensions_diagonal_mvie(
    Poset const& poset,
    RandomNumberGenerator& rng,
    LinearExtensionOptions const& options = LinearExtensionOptions())
{
    std::vector<OrderPolytopeDiagonalRounding<Point> > residual_metrics;
    return linear_extensions_detail::estimate<typename Point::FT>(
        poset, options,
        [&](unsigned int residual_index, Poset const& residual,
            std::vector<unsigned int> const&, double residual_error) {
            OrderPolytope<Point> body =
                OrderPolytope<Point>::from_reduced_relations(residual);
            if (residual_index == residual_metrics.size())
                residual_metrics.push_back(
                    diagonal_mvie_rounding(body));
            return volume_cooling_gaussians_log(
                body, rng, residual_metrics[residual_index], residual_error,
                options.walk_length);
        });
}

template <typename RandomNumberGenerator, typename Point>
LinearExtensionEstimate<typename Point::FT>
estimate_log_linear_extensions(
    Poset const& poset,
    RandomNumberGenerator& rng,
    OrderPolytopeSPDRounding<Point> const& metric,
    LinearExtensionOptions const& options = LinearExtensionOptions())
{
    if (metric.is_diagonal()) {
        typename OrderPolytope<Point>::VT const shape_diag =
            metric.shape().diagonal();
        OrderPolytopeDiagonalRounding<Point> const diagonal_metric(
            poset, metric.center(), shape_diag);
        return estimate_log_linear_extensions(
            poset, rng, diagonal_metric, options);
    }

    metric.validate_for(poset);
    return linear_extensions_detail::estimate<typename Point::FT>(
        poset, options,
        [&](unsigned int, Poset const& residual,
            std::vector<unsigned int> const& original_vertices,
            double residual_error) {
            OrderPolytope<Point> body =
                OrderPolytope<Point>::from_reduced_relations(residual);
            bool identity_map = original_vertices.size() == metric.dimension();
            if (identity_map)
                return
                    volume_cooling_gaussians_log(
                        body, rng, metric, residual_error,
                        options.walk_length);
            OrderPolytopeSPDRounding<Point> const residual_metric =
                metric.restrict_to(body, original_vertices);
            return
                volume_cooling_gaussians_log(
                    body, rng, residual_metric, residual_error,
                    options.walk_length);
        });
}

template <typename RandomNumberGenerator, typename Point>
LinearExtensionEstimate<typename Point::FT>
estimate_log_linear_extensions_mass(
    Poset const& poset,
    RandomNumberGenerator& rng,
    typename OrderPolytope<Point>::MT const& mass,
    LinearExtensionOptions const& options = LinearExtensionOptions())
{
    typedef typename OrderPolytope<Point>::MT MT;
    unsigned int const n = poset.num_elem();
    if (mass.rows() != static_cast<Eigen::Index>(n) ||
        mass.cols() != mass.rows() || !mass.allFinite() ||
        !(mass.array() == mass.transpose().array()).all())
        throw std::invalid_argument(
            "general-mass counting requires a finite symmetric mass matrix");
    if (n != 0) {
        Eigen::LLT<MT> factorization(mass);
        if (factorization.info() != Eigen::Success)
            throw std::invalid_argument(
                "general-mass counting requires an SPD mass matrix");
    }

    return linear_extensions_detail::estimate<typename Point::FT>(
        poset, options,
        [&](unsigned int, Poset const& residual,
            std::vector<unsigned int> const& original_vertices,
            double residual_error) {
            unsigned int const residual_n = residual.num_elem();
            MT residual_mass(residual_n, residual_n);
            for (unsigned int i = 0; i < residual_n; ++i)
                for (unsigned int j = 0; j < residual_n; ++j)
                    residual_mass(i, j) =
                        mass(original_vertices[i], original_vertices[j]);
            OrderPolytope<Point> body =
                OrderPolytope<Point>::from_reduced_relations(residual);
            return volume_cooling_gaussians_log_mass(
                body, rng, residual_mass, residual_error,
                options.walk_length);
        });
}

#endif // VOLUME_LINEAR_EXTENSIONS_VOLUME_HPP
