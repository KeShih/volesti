// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2021- Vaibhav Thakkar

// Contributed by Vaibhav Thakkar, as part of Google Summer of Code 2021 program.
// Contributed and/or modified by Ke Shi, as part of Google Summer of Code 2026 program.

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include <fstream>
#include <iostream>
#include <string>


#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/orderpolytope.h"
#include "convex_bodies/hpolytope.h"

#include "generators/order_polytope_generator.h"

#include "misc/poset.h"
#include "misc/misc.h"
#include "root_finders/trigonometric_equation_solvers.hpp"

#include "random_walks/random_walks.hpp"
#include "sampling/sampling.hpp"



template <typename NT>
void call_test_reflection() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Create Poset, 4 elements, a0 <= a1, a0 <= a2, a1 <= a3
    RV poset_data{{0, 1}, {0, 2}, {1, 3}};
    Poset poset(4, poset_data);

    // Initialize order polytope from the poset
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();

    // no need to explicitly normalize the Polytope's matrix, it is handled inside the function itself
    std::cout << "compute reflection of an incident ray with the facet number 2d (the first relation facet)" << std::endl;
    Point ray = Point::all_ones(OP.dimension());
    ray.set_coord(0, 1.5);

    Point expected_reflected_ray = Point::all_ones(OP.dimension());
    expected_reflected_ray.set_coord(1, 1.5);

    OP.compute_reflection(ray, Point(), 2*OP.dimension());
    CHECK( (expected_reflected_ray == ray) );
}


template <typename NT>
void call_test_line_intersect() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Create Poset, 3 elements, no relations (as easy to verify manually)
    RV poset_data{};
    Poset poset(3, poset_data);

    // Initialize order polytope from the poset
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();

    // intersection of the order polytope with ray from (0.5, 0.5, 0.5) and parallel to x-axis
    Point start_point(OP.dimension(), std::vector<double>(OP.dimension(), 0.5));
    Point expected_intersection(OP.dimension(), std::vector<double>(OP.dimension(), 0.5));
    expected_intersection.set_coord(0, 1.0);

    Point direction = expected_intersection - start_point;
    std::pair<double, double> curr_res = OP.line_intersect(start_point, direction, true);
    Point intersect_point = start_point + curr_res.first * direction;

    CHECK( (intersect_point == expected_intersection) );
}

template <typename NT>
void call_test_vec_mult() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Create Poset, 4 elements, a0 <= a3, a1 <= a3, a2 <= a3
    RV poset_data{{0, 3}, {1, 3}, {2, 3}};
    Poset poset(4, poset_data);

    // Initialize order polytope from the poset
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();

    // multiply by all 1-vector (Ax)
    VT x = Eigen::MatrixXd::Constant(d, 1, 1.0);                            // d x 1 vector
    VT expected_res_vector = -Eigen::MatrixXd::Constant(m, 1, 1.0);         // m x 1 vector
    expected_res_vector.block(d, 0, d, 1) = Eigen::MatrixXd::Constant(d, 1, 1.0);
    expected_res_vector.block(2*d, 0, m - 2*d, 1) = Eigen::MatrixXd::Zero(m - 2*d, 1);

    VT Ax = OP.vec_mult(x);
    CHECK((expected_res_vector - Ax).norm() == 0);

    // multiply by all 1-vector (A^t x)
    x = Eigen::MatrixXd::Constant(m, 1, 1.0);                        // m x 1 vector
    expected_res_vector = Eigen::MatrixXd::Constant(d, 1, 1.0);      // d x 1 vector (entries = (1, 1, 1, -3))
    expected_res_vector(3, 0) = -3.0;

    VT At_x = OP.vec_mult(x, true);
    CHECK((expected_res_vector - At_x).norm() == 0);
}


template <typename NT>
void call_test_basics() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Create Poset, 4 elements, a0 <= a1, a0 <= a2, a1 <= a3
    RV poset_data{{0, 1}, {0, 2}, {1, 3}};
    Poset poset(4, poset_data);
    CHECK(poset.num_elem() == 4);
    CHECK(poset.num_relations() == 3);


    // Initialize order polytope from the poset
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();
    CHECK(d == 4);
    CHECK(m == 2*4 + 3);


    VT expected_dist_vector = Eigen::MatrixXd::Zero(m, 1);
    expected_dist_vector.block(d, 0, d, 1) = Eigen::MatrixXd::Constant(d, 1, 1.0);
    VT ret_dists_vector = Eigen::Map< VT >(OP.get_dists(0.0).data(), m);
    CHECK( (expected_dist_vector - ret_dists_vector).norm() == 0 );

    CHECK(OP.is_in(Point(4, {0.0, 0.5, 1.0, 1.0})) == -1);
    CHECK(OP.is_in(Point(4, {1.0, 0.5, 1.0, 1.0})) == 0);   // a0 <= a1 violated
    CHECK(OP.is_in(Point(4, {0.5, 0.5, 0.0, 1.0})) == 0);   // a0 <= a2 violated
    CHECK(OP.is_in(Point(4, {-0.1, 0.5, 1.0, 1.0})) == 0);  // a0 >= 0 violated
    CHECK(OP.is_in(Point(4, {1.0, 0.5, 1.0, 1.1})) == 0);   // a3 <= 1 violated

    // Create a random Order Polytope of dimension 10 with up to 30 facets as an Hpolytope class
    // (transitive reduction may remove redundant relations, so m <= 30)
    HPolytope<Point> HP = random_orderpoly<HPolytope<Point>, NT>(10, 30);

    d = HP.dimension();
    m = HP.num_of_hyperplanes();

    CHECK(d == 10);
    CHECK(m <= 30);
    CHECK(m >= 2 * 10);  // at least the box constraints

}


template <typename NT>
void call_test_trig_helper() {
    const NT pi = std::acos(NT(-1));
    const NT tol = NT(1e-10);

    // Case 1: x(t) = cos(t), boundary 0 => A=1, B=0, C=0, omega=1
    {
        auto [t, hit] = first_trigonometric_solution(NT(1), NT(0), NT(0), NT(1), NT(0));
        CHECK(hit);
        CHECK(std::abs(t - pi / 2) < tol);
    }

    // Case 2: x(t) = sin(t), boundary 1 => A=0, B=1, C=1, omega=1
    {
        auto [t, hit] = first_trigonometric_solution(NT(0), NT(1), NT(1), NT(1), NT(0));
        CHECK(hit);
        CHECK(std::abs(t - pi / 2) < tol);
    }

    // Case 3: x(t) = 0.5*cos(t) + 0.5*sin(t), check residual
    {
        NT a = NT(0.5);
        NT b = NT(0.5);
        NT c = NT(0.3);
        NT omega = NT(1);
        auto [t, hit] = first_trigonometric_solution(a, b, c, omega, NT(0));
        CHECK(hit);
        NT residual = a * std::cos(omega * t) + b * std::sin(omega * t) - c;
        CHECK(std::abs(residual) < tol);
    }

    // Case 4: no solution when |C| > sqrt(A^2+B^2)
    {
        auto [t, hit] = first_trigonometric_solution(NT(1), NT(0), NT(2), NT(1), NT(0));
        CHECK(!hit);
        CHECK(t == std::numeric_limits<NT>::max());
    }

    // Case 5: cos(t) = 1 has root at t=0; helper returns it (no re-hit filtering)
    {
        auto [t, hit] = first_trigonometric_solution(NT(1), NT(0), NT(1), NT(1), NT(0));
        CHECK(hit);
        CHECK(std::abs(t) < tol);
    }

    // Case 6: caller skips t=0 by passing min_time just past the root
    {
        auto [t, hit] = first_trigonometric_solution(NT(1), NT(0), NT(1), NT(1), NT(1e-10));
        CHECK(hit);
        CHECK(std::abs(t - 2 * pi) < tol);
    }

    // Case 7: near-tangent case, |C| just barely exceeds the radius should return no hit
    {
        auto [t, hit] = first_trigonometric_solution(
            NT(1), NT(0), NT(1) + NT(1e-6), NT(1), NT(0));
        CHECK(!hit);
    }
}


template <typename NT>
void call_test_trig_lower_bound_hit() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    RV poset_data{};
    Poset poset(2, poset_data);
    OrderPolytope<Point> OP(poset);

    Point r(2, {NT(0.5), NT(0.5)});
    Point v(2, {NT(-1), NT(0)});
    NT omega = NT(1);
    int prev = -1;

    auto result = OP.trigonometric_positive_intersect(r, v, omega, prev);
    NT t = result.first;
    int facet = result.second;

    CHECK(facet == 0);
    CHECK(t > NT(0));

    NT residual = -r[0] * std::cos(omega * t) + (-v[0] / omega) * std::sin(omega * t) - NT(0);
    CHECK(std::abs(residual) < NT(1e-8));
}


template <typename NT>
void call_test_trig_upper_bound_hit() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    RV poset_data{};
    Poset poset(2, poset_data);
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension();

    Point r(2, {NT(0.5), NT(0.5)});
    Point v(2, {NT(1), NT(0)});
    NT omega = NT(1);
    int prev = -1;

    auto result = OP.trigonometric_positive_intersect(r, v, omega, prev);
    NT t = result.first;
    int facet = result.second;

    CHECK(facet == static_cast<int>(d));
    CHECK(t > NT(0));

    NT val_at_t = r[0] * std::cos(omega * t) + (v[0] / omega) * std::sin(omega * t);
    CHECK(std::abs(val_at_t - NT(1)) < NT(1e-8));
}


template <typename NT>
void call_test_trig_order_facet_hit() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}};
    Poset poset(2, poset_data);
    OrderPolytope<Point> OP(poset);
    unsigned int d = OP.dimension();

    Point r(2, {NT(0.25), NT(0.75)});
    Point v(2, {NT(1), NT(-1)});
    NT omega = NT(1);
    int prev = -1;

    auto result = OP.trigonometric_positive_intersect(r, v, omega, prev);
    NT t = result.first;
    int facet = result.second;

    CHECK(facet == static_cast<int>(2 * d));
    CHECK(t > NT(0));

    NT x0 = r[0] * std::cos(omega * t) + (v[0] / omega) * std::sin(omega * t);
    NT x1 = r[1] * std::cos(omega * t) + (v[1] / omega) * std::sin(omega * t);
    CHECK(std::abs(x0 - x1) < NT(1e-8));
}


template <typename NT>
void compare_op_hp_trig(OrderPolytope<typename Cartesian<NT>::Point> const& OP,
                        typename Cartesian<NT>::Point const& r,
                        typename Cartesian<NT>::Point const& v,
                        NT omega)
{
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef HPolytope<Point> HP_t;

    HP_t HP(OP.dimension(), OP.get_dense_mat(), OP.get_vec());
    int prev_op = -1, prev_hp = -1;
    auto res_op = OP.trigonometric_positive_intersect(r, v, omega, prev_op);
    auto res_hp = HP.trigonometric_positive_intersect(r, v, omega, prev_hp);

    NT t_op = res_op.first;
    NT t_hp = res_hp.first;
    bool op_hit = (t_op < std::numeric_limits<NT>::max());
    bool hp_hit = (t_hp < std::numeric_limits<NT>::max());

    CHECK(op_hit == hp_hit);
    if (op_hit && hp_hit)
    {
        NT scale = std::max(NT(1), std::max(std::abs(t_op), std::abs(t_hp)));
        CHECK(std::abs(t_op - t_hp) / scale < NT(1e-8));
    }
}


template <typename NT>
void call_test_trig_shifted() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}, {1, 2}};
    Poset poset(3, poset_data);
    OrderPolytope<Point> OP(poset);

    VT c(3);
    c << NT(0.1), NT(0.2), NT(0.3);
    OP.shift(c);

    Point r(3, {NT(0.2), NT(0.3), NT(0.4)});
    Point v(3, {NT(1), NT(-0.5), NT(0.3)});
    NT omega = NT(1.5);

    compare_op_hp_trig<NT>(OP, r, v, omega);
}


template <typename NT>
void call_test_trig_normalized() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}, {1, 2}};
    Poset poset(3, poset_data);
    OrderPolytope<Point> OP(poset);

    OP.normalize();

    Point r(3, {NT(0.1), NT(0.4), NT(0.8)});
    Point v(3, {NT(0.5), NT(-0.3), NT(0.7)});
    NT omega = NT(2.0);

    compare_op_hp_trig<NT>(OP, r, v, omega);
}


template <typename NT>
void call_test_trig_repeated_facet() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    const NT pi = std::acos(NT(-1));

    {
        RV data1{};
        Poset poset(1, data1);
        OrderPolytope<Point> OP(poset);

        Point r(1, {NT(0)});
        Point v(1, {NT(0.3)});
        NT omega = NT(1);
        int prev = 0;

        auto res = OP.trigonometric_positive_intersect(r, v, omega, prev);
        CHECK(res.first > NT(1e-10));
        CHECK(std::abs(res.first - pi) < NT(1e-8));
    }

    {
        RV data2{};
        Poset poset(2, data2);
        OrderPolytope<Point> OP(poset);

        Point r(2, {NT(0), NT(0)});
        Point v(2, {NT(0), NT(0.3)});
        NT omega = NT(1);
        int prev = 0;

        auto res = OP.trigonometric_positive_intersect(r, v, omega, prev);
        CHECK(res.second == 1);
        CHECK(std::abs(res.first - pi) < NT(1e-8));
    }
}


template <typename NT>
void call_test_trig_shifted_normalized() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename OrderPolytope<Point>::VT VT;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}, {1, 2}};
    Poset poset(3, poset_data);
    OrderPolytope<Point> OP(poset);

    VT c(3);
    c << NT(0.05), NT(0.15), NT(0.25);
    OP.shift(c);
    OP.normalize();

    Point r(3, {NT(0.05), NT(0.35), NT(0.55)});
    Point v(3, {NT(0.8), NT(-0.6), NT(0.1)});
    NT omega = NT(1.5);

    int prev = -1;
    auto res = OP.trigonometric_positive_intersect(r, v, omega, prev);
    CHECK(res.second >= static_cast<int>(2 * OP.dimension()));

    compare_op_hp_trig<NT>(OP, r, v, omega);
}


template <typename NT>
void call_test_true_box_facets() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}, {0, 2}, {2, 3}};
    Poset poset(4, poset_data);
    OrderPolytope<Point> OP(poset);

    CHECK(OP.lower_bound_is_facet(0));
    CHECK_FALSE(OP.lower_bound_is_facet(1));
    CHECK_FALSE(OP.lower_bound_is_facet(2));
    CHECK_FALSE(OP.lower_bound_is_facet(3));

    CHECK_FALSE(OP.upper_bound_is_facet(0));
    CHECK(OP.upper_bound_is_facet(1));
    CHECK_FALSE(OP.upper_bound_is_facet(2));
    CHECK(OP.upper_bound_is_facet(3));
}


template <typename NT>
void call_test_true_relation_facets() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef typename Poset::RV RV;

    // Poset with redundant edge {0,2}: constructor reduces to covers {0,1},{1,2}
    RV poset_data{{0, 1}, {1, 2}, {0, 2}};
    Poset poset(3, poset_data);
    OrderPolytope<Point> OP(poset);

    // Only 2 cover relations survive; redundant {0,2} is gone
    CHECK(OP.num_order_relations() == 2);
    // Facets: lb(0) + ub(2) + 2 covers = 4
    CHECK(OP.num_true_facets() == 4);
    // Hyperplanes: 2*3 + 2 = 8 (smaller matrix than with redundant row)
    CHECK(OP.num_of_hyperplanes() == 8);

    // Opting out of the reduction keeps the input relations verbatim
    OrderPolytope<Point> OP_raw(poset, false);
    CHECK(OP_raw.num_order_relations() == 3);
    CHECK(OP_raw.num_of_hyperplanes() == 9);
}


TEST_CASE("basics") {
    call_test_basics<double>();
}

TEST_CASE("line_intersect") {
    call_test_line_intersect<double>();
}

TEST_CASE("reflection") {
    call_test_reflection<double>();
}

TEST_CASE("vec_mult") {
    call_test_vec_mult<double>();
}

TEST_CASE("trig_helper") {
    call_test_trig_helper<double>();
}

TEST_CASE("trig_lower_bound_hit") {
    call_test_trig_lower_bound_hit<double>();
}

TEST_CASE("trig_upper_bound_hit") {
    call_test_trig_upper_bound_hit<double>();
}

TEST_CASE("trig_order_facet_hit") {
    call_test_trig_order_facet_hit<double>();
}

TEST_CASE("trig_shifted") {
    call_test_trig_shifted<double>();
}

TEST_CASE("trig_normalized") {
    call_test_trig_normalized<double>();
}

TEST_CASE("trig_repeated_facet") {
    call_test_trig_repeated_facet<double>();
}

TEST_CASE("trig_shifted_normalized") {
    call_test_trig_shifted_normalized<double>();
}

TEST_CASE("true_box_facets") {
    call_test_true_box_facets<double>();
}

TEST_CASE("true_relation_facets") {
    call_test_true_relation_facets<double>();
}


template <typename NT>
void call_test_order_polytope_ghmc() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 42> RNGType;

    RV poset_data{{0, 1}};
    Poset poset(2, poset_data);
    OP_t OP(poset);
    unsigned int d = OP.dimension();

    Point start(OP.inner_point());

    unsigned int walk_len = 1, rnum = 20, nburns = 2;
    NT a = NT(1);
    RNGType rng(d);
    std::list<Point> randPoints;

    gaussian_sampling<OrderPolytopeGaussianHamiltonianMonteCarloExactWalk>(
        randPoints, OP, rng, walk_len, rnum, a, start, nburns);

    CHECK(randPoints.size() == rnum);
    for (auto const& p : randPoints) {
        CHECK(OP.is_in(p, NT(1e-7)) == -1);
    }
}


template <typename NT>
void call_test_order_polytope_ghmc_hp_compare() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef HPolytope<Point> HP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 7> RNGType;

    RV poset_data{{0, 1}, {1, 2}};
    Poset poset(3, poset_data);
    OP_t OP(poset);
    unsigned int d = OP.dimension();

    HP_t HP(d, OP.get_dense_mat(), OP.get_vec());
    HP.ComputeInnerBall();

    Point start(OP.inner_point());
    unsigned int walk_len = 1, rnum = 20, nburns = 2;
    NT a = NT(1);

    RNGType rng_op(d);
    std::list<Point> op_points;
    gaussian_sampling<OrderPolytopeGaussianHamiltonianMonteCarloExactWalk>(
        op_points, OP, rng_op, walk_len, rnum, a, start, nburns);

    RNGType rng_hp(d);
    std::list<Point> hp_points;
    gaussian_sampling<GaussianHamiltonianMonteCarloExactWalk>(
        hp_points, HP, rng_hp, walk_len, rnum, a, start, nburns);

    CHECK(op_points.size() == rnum);
    CHECK(hp_points.size() == rnum);

    for (auto const& p : op_points) {
        CHECK(OP.is_in(p, NT(1e-7)) == -1);
    }
    for (auto const& p : hp_points) {
        CHECK(HP.is_in(p, NT(1e-7)) == -1);
    }
}


template <typename NT>
void call_test_coord_intersect_normalized_shifted() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef HPolytope<Point> HP_t;
    typedef typename OP_t::VT VT;
    typedef typename Poset::RV RV;

    RV poset_data{{0, 1}, {1, 2}};
    Poset poset(3, poset_data);
    OP_t OP(poset);
    HP_t HP(OP.dimension(), OP.get_dense_mat(), OP.get_vec());

    OP.normalize();
    HP.normalize();

    VT center(OP.dimension());
    center << NT(0.2), NT(0.5), NT(0.8);
    OP.shift(center);
    HP.shift(center);

    Point p(OP.dimension());
    VT op_lamdas = VT::Zero(OP.num_of_hyperplanes());
    VT hp_lamdas = VT::Zero(HP.num_of_hyperplanes());

    unsigned int first_coord = 0;
    auto op_first = OP.line_intersect_coord(p, first_coord, op_lamdas);
    auto hp_first = HP.line_intersect_coord(p, first_coord, hp_lamdas);
    CHECK(op_first.first == doctest::Approx(hp_first.first));
    CHECK(op_first.second == doctest::Approx(hp_first.second));

    Point p_prev = p;
    NT step = (op_first.first + op_first.second) / NT(3);
    p.set_coord(first_coord, step);

    unsigned int second_coord = 1;
    auto op_second = OP.line_intersect_coord(
        p, p_prev, second_coord, first_coord, op_lamdas);
    auto hp_second = HP.line_intersect_coord(
        p, p_prev, second_coord, first_coord, hp_lamdas);

    CHECK(op_second.first == doctest::Approx(hp_second.first));
    CHECK(op_second.second == doctest::Approx(hp_second.second));
}


TEST_CASE("order_polytope_ghmc") {
    call_test_order_polytope_ghmc<double>();
}

TEST_CASE("order_polytope_ghmc_hp_compare") {
    call_test_order_polytope_ghmc_hp_compare<double>();
}

template <typename NT>
class ScriptedGaussianRNG {
public:
    ScriptedGaussianRNG(std::vector<NT> normal_values,
                        std::vector<NT> uniform_values)
        : normals(std::move(normal_values)), uniforms(std::move(uniform_values))
    {}

    NT sample_ndist() {
        REQUIRE(normal_index < normals.size());
        return normals[normal_index++];
    }

    NT sample_urdist() {
        REQUIRE(uniform_index < uniforms.size());
        return uniforms[uniform_index++];
    }

private:
    std::vector<NT> normals, uniforms;
    unsigned int normal_index = 0, uniform_index = 0;
};

template <typename NT>
struct ScriptedOrderHmcFixture {
    typedef Cartesian<NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename Poset::RV RV;
    typedef ScriptedGaussianRNG<NT> RNG;
    typedef typename OrderPolytopeGaussianHamiltonianMonteCarloExactWalk::
        template Walk<OP_t, RNG> Walk;
    typedef typename Walk::trajectory_status Status;

    struct Outcome {
        explicit Outcome(Point const& start)
            : point(start), status(Status::success), feasible(false) {}

        Point point;
        Status status;
        bool feasible;
        std::string failure;
    };

    static Outcome run(unsigned int n, RV relations,
                       std::vector<NT> start,
                       std::vector<NT> normals,
                       std::vector<NT> uniforms,
                       unsigned int rho = 100,
                       unsigned int walk_length = 1)
    {
        OP_t OP(Poset(n, relations));
        Point p(n, start);
        RNG rng(std::move(normals), std::move(uniforms));
        typename OrderPolytopeGaussianHamiltonianMonteCarloExactWalk::parameters
            params(NT(1), true, rho, true);
        Outcome outcome(p);

        try {
            Walk walk(OP, p, NT(1), rng, params);
            walk.apply(OP, p, NT(1), walk_length, rng);
            outcome.status = walk.last_trajectory_status();
        } catch (std::runtime_error const& error) {
            outcome.failure = error.what();
        }

        outcome.point = p;
        outcome.feasible = OP.is_in(p, NT(1e-10)) == -1;
        return outcome;
    }
};

template <typename NT>
void call_test_order_polytope_ghmc_single_event_equivalence() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels;
    auto const outcome = Fixture::run(
        1, rels, {NT(0.5)}, {NT(0), NT(2)}, {NT(0), NT(0.75)});

    CHECK(outcome.failure.empty());
    CHECK(outcome.status == Fixture::Status::success);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] ==
          doctest::Approx(NT(0.08838611995487355)).epsilon(1e-12));
}

template <typename NT>
void call_test_order_polytope_ghmc_simultaneous_disjoint_batch() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels;
    // Constructor: zero velocity and zero time.  apply(): equal coordinate
    // velocities and T=0.75, producing an exact simultaneous hit of the two
    // disjoint upper walls.
    auto const outcome = Fixture::run(
        2, rels, {NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(2), NT(2)}, {NT(0), NT(0.75)});

    CHECK(outcome.failure.empty());
    CHECK(outcome.status == Fixture::Status::success);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(outcome.point[1]).epsilon(1e-13));
    CHECK(outcome.point[0] ==
          doctest::Approx(NT(0.08838611995487355)).epsilon(1e-12));
}

template <typename NT>
void call_test_order_polytope_ghmc_simultaneous_disjoint_relations() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels{{0, 1}, {2, 3}};
    // The two independent order differences have identical trajectories and
    // reach zero together.  Their identity-mass reflections swap velocities
    // on disjoint coordinate pairs, so the joint operation is unambiguous.
    auto const outcome = Fixture::run(
        4, rels, {NT(0.25), NT(0.75), NT(0.25), NT(0.75)},
        {NT(0), NT(0), NT(0), NT(0), NT(1), NT(-1), NT(1), NT(-1)},
        {NT(0), NT(0.75)});

    CHECK(outcome.failure.empty());
    CHECK(outcome.status == Fixture::Status::success);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(outcome.point[2]).epsilon(1e-13));
    CHECK(outcome.point[1] == doctest::Approx(outcome.point[3]).epsilon(1e-13));
    CHECK(outcome.point[0] < outcome.point[1]);
}

template <typename NT>
void call_test_order_polytope_ghmc_simultaneous_shared_failure() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels{{0, 1}};
    // x1(t)=2*x0(t), so lower wall x0=0 and relation x0=x1 are
    // reached together.  Their supports overlap; no reflection ordering is
    // selected here.  The leg rolls back before reporting a fatal failure.
    auto const outcome = Fixture::run(
        2, rels, {NT(0.25), NT(0.5)},
        {NT(0), NT(0), NT(-1), NT(-2)}, {NT(0), NT(0.75)});

    CHECK(outcome.failure.find("shared-coordinate") != std::string::npos);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(NT(0.25)));
    CHECK(outcome.point[1] == doctest::Approx(NT(0.5)));
}

template <typename NT>
void call_test_order_polytope_ghmc_near_simultaneous_disjoint() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels;
    auto const outcome = Fixture::run(
        2, rels, {NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(2), NT(1.999999999)}, {NT(0), NT(0.75)});

    CHECK(outcome.failure.empty());
    CHECK(outcome.status == Fixture::Status::success);
    CHECK(outcome.feasible);
    CHECK(std::abs(outcome.point[0] - outcome.point[1]) > NT(1e-12));
    CHECK(outcome.point[0] ==
          doctest::Approx(NT(0.08838611995487355)).epsilon(1e-12));
}

template <typename NT>
void call_test_order_polytope_ghmc_reflection_limit_rollback() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels;
    auto const outcome = Fixture::run(
        2, rels, {NT(0.5), NT(0.5)},
        {NT(0), NT(0), NT(2), NT(2)}, {NT(0), NT(0.75)}, 1);

    CHECK(outcome.failure.empty());
    CHECK(outcome.status == Fixture::Status::reflection_limit);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(NT(0.5)));
    CHECK(outcome.point[1] == doctest::Approx(NT(0.5)));
}

template <typename NT>
void call_test_order_polytope_ghmc_constructor_failure_reported() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels{{0, 1}};
    auto const outcome = Fixture::run(
        2, rels, {NT(0.25), NT(0.5)},
        {NT(-1), NT(-2)}, {NT(0.75)}, 100, 0);

    CHECK(outcome.failure.find("shared-coordinate") != std::string::npos);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(NT(0.25)));
    CHECK(outcome.point[1] == doctest::Approx(NT(0.5)));
}

template <typename NT>
void call_test_order_polytope_ghmc_nonfinite_failure_reported() {
    typedef ScriptedOrderHmcFixture<NT> Fixture;
    typedef typename Fixture::RV RV;
    RV rels;
    NT const nan = std::numeric_limits<NT>::quiet_NaN();
    // The constructor consumes the finite zero-velocity leg.  The first
    // apply leg receives a non-finite velocity and must reject before a
    // sample is produced; no random values exist for the requested second leg.
    auto const outcome = Fixture::run(
        2, rels, {NT(0.25), NT(0.5)},
        {NT(0), NT(0), nan, NT(1)}, {NT(0), NT(0.5)}, 100, 2);

    CHECK(outcome.failure.find("numerical event failure") != std::string::npos);
    CHECK(outcome.feasible);
    CHECK(outcome.point[0] == doctest::Approx(NT(0.25)));
    CHECK(outcome.point[1] == doctest::Approx(NT(0.5)));
}

TEST_CASE("order_polytope_ghmc_simultaneous_disjoint_batch") {
    call_test_order_polytope_ghmc_simultaneous_disjoint_batch<double>();
}

TEST_CASE("order_polytope_ghmc_single_event_equivalence") {
    call_test_order_polytope_ghmc_single_event_equivalence<double>();
}

TEST_CASE("order_polytope_ghmc_simultaneous_disjoint_relations") {
    call_test_order_polytope_ghmc_simultaneous_disjoint_relations<double>();
}

TEST_CASE("order_polytope_ghmc_simultaneous_shared_failure") {
    call_test_order_polytope_ghmc_simultaneous_shared_failure<double>();
}

TEST_CASE("order_polytope_ghmc_near_simultaneous_disjoint") {
    call_test_order_polytope_ghmc_near_simultaneous_disjoint<double>();
}

TEST_CASE("order_polytope_ghmc_reflection_limit_rollback") {
    call_test_order_polytope_ghmc_reflection_limit_rollback<double>();
}

TEST_CASE("order_polytope_ghmc_constructor_failure_reported") {
    call_test_order_polytope_ghmc_constructor_failure_reported<double>();
}

TEST_CASE("order_polytope_ghmc_nonfinite_failure_reported") {
    call_test_order_polytope_ghmc_nonfinite_failure_reported<double>();
}

TEST_CASE("coord_intersect_normalized_shifted") {
    call_test_coord_intersect_normalized_shifted<double>();
}


template <typename NT>
void call_test_event_queue_facets() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Point;
    typedef OrderPolytope<Point> OP_t;
    typedef typename Poset::RV RV;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 13> RNGType;
    typedef OrderPolytopeGaussianHamiltonianMonteCarloExactWalk WalkPolicy;
    typedef typename WalkPolicy::template Walk<OP_t, RNGType> Walk;

    // Chain: only 2 true facets per vertex (lower for source, upper for sink,
    // cover relation for each edge in the Hasse diagram).
    {
        RV rels{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        Poset poset(5, rels);
        OP_t OP(poset);

        // Facet count: lb(0) + ub(4) + 4 covers = 6
        CHECK(OP.num_true_facets() == 6);
    }

    // Antichain: every element gets both lower and upper bound facets.
    {
        RV rels;
        Poset poset(4, rels);
        OP_t OP(poset);

        CHECK(OP.num_true_facets() == 8);
    }

    // Diamond: 0->1, 0->2, 1->3, 2->3. lb(0), ub(3), 4 covers = 6
    {
        RV rels{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        Poset poset(4, rels);
        OP_t OP(poset);

        CHECK(OP.num_true_facets() == 6);
    }
}


TEST_CASE("event_queue_facets") {
    call_test_event_queue_facets<double>();
}
