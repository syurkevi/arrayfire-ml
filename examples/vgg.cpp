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
#include <iostream>

using std::cout;
using std::endl;
using namespace af;
using namespace af::nn;
using namespace af::autograd;

nn::Sequential vgg19_factory() {
    nn::Sequential net;

    int wx = 3, wy = 3;
    int sx = 1, sy = 1;
    int px = 1, py = 1;
    int dx = 1, dy = 1;
    int n_in = 3;
    int n_out = 64;
    bool with_bias = true;

    char filename[] = "vgg19_af.weights";
    Variable Convolve2W0 = Variable(readArray(filename, 0u), true);
    Variable Convolve2b0 = Variable(readArray(filename, 1u), true);
    net.add(nn::Convolve2(Convolve2W0, Convolve2b0, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    n_in = 64;
    Variable Convolve2W1 = Variable(readArray(filename, 2u), true);
    Variable Convolve2b1 = Variable(readArray(filename, 3u), true);
    net.add(nn::Convolve2(Convolve2W1, Convolve2b1, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    int pool_wx = 2;
    int pool_wy = 2;
    int pool_sx = 2;
    int pool_sy = 2;
    int pool_px = 0;
    int pool_py = 0;
    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    n_in = 64;
    n_out = 128;
    Variable Convolve2W2 = Variable(readArray(filename, 4u), true);
    Variable Convolve2b2 = Variable(readArray(filename, 5u), true);
    net.add(nn::Convolve2(Convolve2W2, Convolve2b2, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    n_in = 128;
    Variable Convolve2W3 = Variable(readArray(filename, 6u), true);
    Variable Convolve2b3 = Variable(readArray(filename, 7u), true);
    net.add(nn::Convolve2(Convolve2W3, Convolve2b3, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    n_in = 128;
    n_out = 256;
    Variable Convolve2W4 = Variable(readArray(filename, 8u), true);
    Variable Convolve2b4 = Variable(readArray(filename, 9u), true);
    net.add(nn::Convolve2(Convolve2W4, Convolve2b4, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    n_in = 256;
    Variable Convolve2W5 = Variable(readArray(filename, 10u), true);
    Variable Convolve2b5 = Variable(readArray(filename, 11u), true);
    net.add(nn::Convolve2(Convolve2W5, Convolve2b5, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W6 = Variable(readArray(filename, 12u), true);
    Variable Convolve2b6 = Variable(readArray(filename, 13u), true);
    net.add(nn::Convolve2(Convolve2W6, Convolve2b6, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W7 = Variable(readArray(filename, 14u), true);
    Variable Convolve2b7 = Variable(readArray(filename, 15u), true);
    net.add(nn::Convolve2(Convolve2W7, Convolve2b7, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    n_in = 256;
    n_out = 512;
    Variable Convolve2W8 = Variable(readArray(filename, 16u), true);
    Variable Convolve2b8 = Variable(readArray(filename, 17u), true);
    net.add(nn::Convolve2(Convolve2W8, Convolve2b8, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    n_in = 512;
    Variable Convolve2W9 = Variable(readArray(filename, 18u), true);
    Variable Convolve2b9 = Variable(readArray(filename, 19u), true);
    net.add(nn::Convolve2(Convolve2W9, Convolve2b9, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W10 = Variable(readArray(filename, 20u), true);
    Variable Convolve2b10 = Variable(readArray(filename, 21u), true);
    net.add(nn::Convolve2(Convolve2W10, Convolve2b10, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W11 = Variable(readArray(filename, 22u), true);
    Variable Convolve2b11 = Variable(readArray(filename, 23u), true);
    net.add(nn::Convolve2(Convolve2W11, Convolve2b11, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    Variable Convolve2W12 = Variable(readArray(filename, 24u), true);
    Variable Convolve2b12 = Variable(readArray(filename, 25u), true);
    net.add(nn::Convolve2(Convolve2W12, Convolve2b12, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W13 = Variable(readArray(filename, 26u), true);
    Variable Convolve2b13 = Variable(readArray(filename, 27u), true);
    net.add(nn::Convolve2(Convolve2W13, Convolve2b13, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W14 = Variable(readArray(filename, 28u), true);
    Variable Convolve2b14 = Variable(readArray(filename, 29u), true);
    net.add(nn::Convolve2(Convolve2W14, Convolve2b14, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    Variable Convolve2W15 = Variable(readArray(filename, 30u), true);
    Variable Convolve2b15 = Variable(readArray(filename, 31u), true);
    net.add(nn::Convolve2(Convolve2W15, Convolve2b15, sx, sy, px, py, dx, dy));
    net.add(nn::ReLU());

    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    net.add(nn::Flatten());

    Variable LinearW0 = Variable(readArray(filename, 32u), true);
    Variable Linearb0 = Variable(readArray(filename, 33u), true);
    net.add(nn::Linear(LinearW0, Linearb0));
    net.add(nn::ReLU());
    net.add(nn::Dropout(0.5));

    Variable LinearW1 = Variable(readArray(filename, 34u), true);
    Variable Linearb1 = Variable(readArray(filename, 35u), true);
    net.add(nn::Linear(LinearW1, Linearb1));
    net.add(nn::ReLU());
    net.add(nn::Dropout(0.5));

    Variable LinearW2 = Variable(readArray(filename, 36u), true);
    Variable Linearb2 = Variable(readArray(filename, 37u), true);
    net.add(nn::Linear(LinearW2, Linearb2));

    net.add(nn::Softmax());

    return net;
}

int main()
{
    af::info();
    const double lr = 0.1;
    const int numSamples = 1;
    auto in = af::randu(224, 224, 3, numSamples);
    auto out = af::randu(1000, numSamples);

    nn::Sequential vgg19 = vgg19_factory();

    Variable result;
    timer::start();
    const int iters = 10;
    for(int i=0; i<iters; ++i)
        result = vgg19.forward(nn::input(in));
    printf("elapsed seconds: %g\n", timer::stop() / (float)iters);
    cout << result.array().dims() << endl;

    return 0;
}
