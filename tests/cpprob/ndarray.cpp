#include "gtest/gtest.h"

#include <vector>

#include "cpprob/ndarray.hpp"

using cpprob::NDArray;


TEST(ndarray, trivial_constructor)
{
    NDArray<double> test_double{};
    EXPECT_EQ(test_double.values(), std::vector<double>{});
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>{});

    NDArray<int> arr_int{};
    EXPECT_EQ(arr_int.values(), std::vector<int>{});
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>{});
}

TEST(ndarray, constructor_from_object_same_type)
{
    NDArray<double> test_double{2.0};
    EXPECT_EQ(test_double.values(), std::vector<double>{2.0});
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>{1});

    NDArray<int> arr_int{3};
    EXPECT_EQ(arr_int.values(), std::vector<int>{3});
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>{1});
}

TEST(ndarray, constructor_from_object_compatible_type)
{
    NDArray<double> test_double{2};
    EXPECT_EQ(test_double.values(), std::vector<double>{2.0});
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>{1});

    NDArray<double> arr_int{3.0f};
    EXPECT_EQ(arr_int.values(), std::vector<double>{3});
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>{1});
}

TEST(ndarray, constructor_from_vector_same_type)
{
    std::vector<double> vec_double{2, 3, 8, -2};
    NDArray<double> test_double{vec_double.begin(), vec_double.end()};
    EXPECT_EQ(test_double.values(), vec_double);
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>{vec_double.size()});

    std::vector<int> vec_int{2, 3, 8, -2, 0};
    NDArray<int> arr_int{vec_int.begin(), vec_int.end()};
    EXPECT_EQ(arr_int.values(), vec_int);
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>{vec_int.size()});
}

TEST(ndarray, constructor_from_vector_compatible_type)
{
    std::vector<int> vec_int{2, 3, 8, -2};
    std::vector<double> vec_int_double(vec_int.begin(), vec_int.end());
    NDArray<double> test_double_int{vec_int.begin(), vec_int.end()};
    EXPECT_EQ(test_double_int.values(), vec_int_double);
    EXPECT_EQ(test_double_int.shape(), std::vector<std::size_t>{vec_int.size()});

    std::vector<float> vec_float{2, 3, 8, -2, 0};
    std::vector<double> vec_float_double(vec_float.begin(), vec_float.end());
    NDArray<double> test_double_float{vec_float.begin(), vec_float.end()};
    EXPECT_EQ(test_double_float.values(), vec_float_double);
    EXPECT_EQ(test_double_float.shape(), std::vector<std::size_t>{vec_float.size()});
}

TEST(ndarray, constructor_from_matrix_same_type)
{
    std::vector<std::vector<double>> mat_double{{2, 3, 8, -2}, {5, 1, 3, 6}};
    NDArray<double> test_double{mat_double.begin(), mat_double.end()};
    EXPECT_EQ(test_double.values(), std::vector<double>({2, 3, 8, -2,
                                                         5, 1, 3,  6}));
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>({2, 4}));

    std::vector<std::vector<int>> mat_int{{2, 3, 8, -2, 0},
                                          {1, 2, 5,  1, 2}};
    NDArray<int> arr_int{mat_int.begin(), mat_int.end()};
    EXPECT_EQ(arr_int.values(), std::vector<int>({2, 3, 8, -2, 0,
                                                 1, 2, 5, 1,  2}));
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>({2, 5}));
}

TEST(ndarray, constructor_from_matrix_compatible_type)
{
    std::vector<std::vector<int>> mat_double{{2, 3, 8, -2},
                                             {5, 1, 3, 6}};
    NDArray<double> test_double{mat_double.begin(), mat_double.end()};
    EXPECT_EQ(test_double.values(), std::vector<double>({2, 3, 8, -2,
                                                         5, 1, 3,  6}));
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>({2, 4}));

    std::vector<std::vector<float>> mat_float{{2.0f, 3.0f, 8.0f, -2.0f, 0.0f},
                                              {1.0f, 2.0f, 5.0f, 1.0f,  2.0f}};
    NDArray<double> arr_int{mat_float.begin(), mat_float.end()};
    EXPECT_EQ(arr_int.values(), std::vector<double>({2, 3, 8, -2, 0,
                                                    1, 2, 5, 1,  2}));
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>({2, 5}));
}

TEST(ndarray, constructor_from_matrix_same_type_padding)
{
    std::vector<std::vector<double>> mat_double{{2, 3, 8, -2}, {5, 1}};
    NDArray<double> test_double{mat_double.begin(), mat_double.end()};
    EXPECT_EQ(test_double.values(), std::vector<double>({2, 3, 8, -2,
                                                         5, 1, 3,  6}));
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>({2, 4}));

    std::vector<std::vector<int>> mat_int{{2, 3, 8},
                                          {1, 2, 5, 1,  2}};
    NDArray<int> arr_int{mat_int.begin(), mat_int.end()};
    EXPECT_EQ(arr_int.values(), std::vector<int>({2, 3, 8, 0, 0,
                                                  1, 2, 5, 1, 2}));
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>({2, 5}));
}

TEST(ndarray, constructor_from_matrix_compatible_type_padding)
{
    std::vector<std::vector<int>> mat_int{{2, 3},
                                          {5, 1, 3, 6}};
    NDArray<double> test_double{mat_int.begin(), mat_int.end()};
    EXPECT_EQ(test_double.values(), std::vector<double>({2, 3, 0, 0,
                                                         5, 1, 3, 6}));
    EXPECT_EQ(test_double.shape(), std::vector<std::size_t>({2, 4}));

    std::vector<std::vector<float>> mat_float{{2, 3, 8, -2, 0},
                                              {2, 3, 8},
                                              {1, 2, 5, 1,  2},
                                              {1, 2}};
    NDArray<double> arr_int{mat_float.begin(), mat_float.end()};
    EXPECT_EQ(arr_int.values(), std::vector<double>({2, 3, 8, -2, 0,
                                                     2, 3, 8,  0, 0,
                                                     1, 2, 5,  1, 2,
                                                     1, 2, 0,  0, 0}));
    EXPECT_EQ(arr_int.shape(), std::vector<std::size_t>({4, 5}));
}


TEST(ndarray, constructor_from_four_tensor)
{
    std::vector<std::vector<std::vector<int>>>
            first_three_tensor{
                         {{2, 3},
                          {5, 1, 3, 6}},
                         {{1},
                          {2,3,6},
                          {1,7}}
                        };
    std::vector<std::vector<std::vector<int>>>
            second_three_tensor{
                    {{2, 3},
                     {5, 1, 3, 6}},
                    {{1},
                     {2,3,6},
                     {1,7}},
                    {{1},
                     {2,3,6},
                     {2,3,6},
                     {1,7}}
                    };
    std::vector<std::vector<std::vector<std::vector<int>>>> four_tensor{first_three_tensor, second_three_tensor};
    NDArray<double> test_tensor{four_tensor.begin(), four_tensor.end()};
    std::vector<double> solution{2,3,0,0,5,1,3,6,0,0,0,0,0,0,0,0,
                                 1,0,0,0,2,3,6,0,1,7,0,0,0,0,0,0,
                                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                 2,3,0,0,5,1,3,6,0,0,0,0,0,0,0,0,
                                 1,0,0,0,2,3,6,0,1,7,0,0,0,0,0,0,
                                 1,0,0,0,2,3,6,0,2,3,6,0,1,7,0,0};
    EXPECT_EQ(test_tensor.values(), solution);
    EXPECT_EQ(test_tensor.shape(), std::vector<std::size_t>({2, 3, 4, 4}));
}
