/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd.h>
#include <af/nn.h>
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

using af::allTrue;
using af::array;
using af::autograd::Variable;
using af::constant;
using af::dim4;
using af::exp;
using af::randu;
using af::range;
using af::sigmoid;
using af::sum;
using af::tanh;
using af::tile;

TEST(Autograd, Multiply)
{
    auto x = Variable(randu(5), true);
    auto y = x * x;
    auto dy = Variable(constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diff = dx.array() - 2 * x.array();
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
}

TEST(Autograd, MultiplyAdd)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = dx.array() - 2 * x.array() - y.array();
    auto diffy = dy.array() - 2 * y.array() - x.array();

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, noCalcGrad)
{
    auto x = Variable(randu(5), false);
    auto y = Variable(randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();

    auto diffy = (dy.array() - 2 * y.array() - x.array());
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
    try {
        auto dx = x.grad();
    } catch(af::exception &ex) {
        std::cout << ex.what() << std::endl;
        return;
    }
    printf("%s:%d No Gradient check Failed\n");
}

TEST(Autograd, MultiplySub)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5), true);
    auto z = x * x - x * y;
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (2 * x.array() - y.array()));
    auto diffy = (dy.array() - (-x.array()));

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, DivideAdd)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5), true);
    auto z = x + x / y + y;
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (1.0 + 1.0 / y.array()));
    auto diffy = (dy.array() - (1.0 - x.array() / (y.array() * y.array())));

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, MultiplyAddScalar)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5), true);
    auto z = 2 * x + x * y + y;
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (2.0 + y.array()));
    auto diffy = (dy.array() - (1.0 + x.array()));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Exp)
{
    auto x = Variable(randu(5), true);
    auto y = exp(x);
    auto dy = Variable(constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (exp(x.array())));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
}

TEST(Autograd, Sigmoid)
{
    auto x  = Variable(randu(5), true);
    auto y  = sigmoid(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    array diffy = (dx.array() - (y.array() * (1 - y.array())));
    array diffx = (dx.array() - (sigmoid(x.array()) * (1 - sigmoid(x.array()))));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Softmax)
{
    auto x = Variable(randu(dim4(5, 2)), true);
    auto y  = softmax(x);

    auto dy = Variable(constant(1.0, 5, 2), false);
    y.backward(dy);

    auto dx = x.grad();

    EXPECT_TRUE(abs((x.dims()[1] * 1.0) - sum<double>(y.array())) < 1E-5);
    EXPECT_TRUE(abs(sum<double>(dx.array())) < 1E-5);
}

TEST(Autograd, assign)
{
    auto x   = Variable(range(5) + 0.5, true);
    auto idx = Variable(range(2) + 1, false);

    auto y = assign(x, idx, Variable(constant(-2.0, idx.dims()), false));
    auto z = sum(2*y, {0});
    z.backward();

    auto expected_grad = constant(2, x.dims());
    expected_grad(idx.array()) = 0;

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
}

TEST(Autograd, lookup)
{
    auto x   = Variable(randu(5), true);
    auto idx = Variable(range(2) + 1, false);

    auto y = lookup(x, idx);
    auto z = sum(2*y, {0});
    z.backward();

    auto expected_grad = constant(0, x.dims());
    expected_grad(idx.array()) = 2;

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
}

TEST(Autograd, Tanh)
{
    auto x = Variable(randu(5), true);
    auto y = tanh(x);
    auto dy = Variable(constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (1 - y.array() * y.array()));
    auto diffy = (dx.array() - (1 + tanh(x.array())) * (1 - tanh(x.array())));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Tile)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5, 2), true);
    auto z = y * tileAs(x, y);
    auto dz = Variable(constant(1.0, 5, 2), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - tile(x.array(), 1, 2));
    auto diffy = (dx.array() - sum(y.array(), 1));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Sum)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5, 2), true);
    auto z = x * sumAs(y, x);
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - tile(x.array(), 1, 2));
    auto diffy = (dx.array() - sum(y.array(), 1));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Mean)
{
    auto x = Variable(randu(5), true);
    auto y = Variable(randu(5, 3, 2), true);
    auto z = x * mean(y, {1,2});
    auto dz = Variable(constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - 6 * tile(x.array(), 1, 3, 2));
    auto diffy = (dx.array() - mean(mean(y.array(), 1), 2));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Conv2)
{
    auto x = Variable(randu(10, 10), true);
    auto f = Variable(randu(3, 3), true);
    auto z = convolve2(x, f, 1, 1, 1, 1, 1, 1);
    auto dz = Variable(constant(1.0, z.dims()), false);
    z.backward(dz);
}

TEST(Autograd, MaxPool2)
{
    float h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     3, 1, 1, 3, 1, 1, 3,
                     1, 1, 1, 1, 1, 1, 1,
                     1, 4, 1, 4, 1, 4, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     5, 1, 5, 1, 6, 1, 2 };
    array input(7, 7, h_in);
    auto x = Variable(input, true);
    auto z = maxpool2(x, 3, 3, 3, 3, 1, 1);

    float h_out[] = { 2, 2, 2,
                      4, 4, 4,
                      5, 6, 2 };

    array output(3, 3, h_out);
    auto diff = (z.array() - output);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));

    auto dz = Variable(range(z.dims()) + 1 + range(z.dims(), 1), false);
    z.backward(dz);

    float h_pool_grad[] = { 0, 1, 0, 2, 0, 3, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 2, 0, 3, 0, 4, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            3, 0, 0, 0, 4, 0, 5 };

    array pool_grad(7, 7, h_pool_grad);

    auto gdiff = (x.grad().array() - pool_grad);
    EXPECT_TRUE(allTrue<bool>(abs(gdiff) < 1E-5));
}

TEST(Autograd, MaxPool2batch)
{
    float h_in[] = { 1, 2, 1, 2, 1, 2, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     3, 1, 1, 3, 1, 1, 3,
                     1, 1, 1, 1, 1, 1, 1,
                     1, 4, 1, 4, 1, 4, 1,
                     1, 1, 1, 1, 1, 1, 1,
                     5, 1, 5, 1, 6, 1, 2 };
    array input(7, 7, h_in);
    input = tile(input, dim4(1, 1, 2, 2));
    auto x = Variable(input, true);
    auto z = maxpool2(x, 3, 3, 3, 3, 1, 1);

    float h_out[] = { 2, 2, 2,
                      4, 4, 4,
                      5, 6, 2 };

    array output(3, 3, h_out);
    output = tile(output, dim4(1, 1, 2, 2));
    auto diff = (z.array() - output);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));

    auto dz = Variable(range(z.dims()) + 1 + range(z.dims(), 1), false);
    z.backward(dz);

    float h_pool_grad[] = { 0, 1, 0, 2, 0, 3, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            0, 2, 0, 3, 0, 4, 0,
                            0, 0, 0, 0, 0, 0, 0,
                            3, 0, 0, 0, 4, 0, 5 };

    array pool_grad(7, 7, h_pool_grad);
    pool_grad = tile(pool_grad, dim4(1, 1, 2, 2));

    auto gdiff = (x.grad().array() - pool_grad);
    EXPECT_TRUE(allTrue<bool>(abs(gdiff) < 1E-5));
}
