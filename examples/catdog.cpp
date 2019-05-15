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
#include <af/optim.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

#include <string>
#include <memory>
#include <iostream>

using std::cout;
using std::endl;
using std::string;

using namespace af;
using namespace af::nn;
using namespace af::autograd;

namespace fs = boost::filesystem;

const static int img_sz = 64;

nn::Sequential catdog_network() {
    nn::Sequential net;

    /*
    int wx = 3, wy = 3;
    int sx = 1, sy = 1;
    int px = 1, py = 1;
    int dx = 1, dy = 1;
    int n_in = 1;
    int n_out = 32;
    bool with_bias = true;

    net.add(nn::Convolve2(wx, wy, sx, sy, px, py, dx, dy, n_in, n_out));
    net.add(nn::ReLU());

    int pool_wx = 2;
    int pool_wy = 2;
    int pool_sx = 2;
    int pool_sy = 2;
    int pool_px = 0;
    int pool_py = 0;
    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    n_in  = 32;
    n_out = 32;

    net.add(nn::Convolve2(wx, wy, sx, sy, px, py, dx, dy, n_in, n_out));
    net.add(nn::ReLU());
    net.add(nn::Pool2(pool_wx, pool_wy, pool_sx, pool_sy, pool_px, pool_py));

    net.add(nn::Flatten());

    */
    //net.add(nn::Linear(4096, 8192));
    int flat_size = 8192; //TODO: programattic?
    int fc_size = 128; //TODO: programattic?
    //net.add(nn::Linear(flat_size, fc_size, true, 0));
    //net.add(nn::ReLU());

    int output_size = 1;
    //net.add(nn::Linear(fc_size, output_size, true, 0));
    net.add(nn::Linear(4096, output_size));
    net.add(nn::Sigmoid());
    return net;
}

int dir_file_count(const fs::path path) {
    if(!fs::exists(path)) {
        cout << "\nNot found: " << path.string() << endl;
    }

    int file_count = 0;

    if (fs::is_directory(path)) {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(path);
            dir_itr != end_iter;
            ++dir_itr) {

            try {
                if (fs::is_regular_file(dir_itr->status())) {
                    file_count++;
                }
            } catch ( const std::exception & ex ) {
                std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
            }
        }
    }
    return file_count;
}

int load_dir_imgs_resize(const fs::path path, af::array &imgs, const int n, const unsigned offset=0) {
    const int w = imgs.dims(0);//is dims(0) height or width in forge?
    const int h = imgs.dims(1);
    const int c = imgs.dims(2);

    int file_count = 0;
    if (fs::is_directory(path)) {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(path);
            dir_itr != end_iter;
            ++dir_itr) {

            try {
                if (fs::is_regular_file(dir_itr->status())) {
                    array img = loadImage((dir_itr->path().string()).c_str(), false);
                    if(img.dims(0) < img.dims(1)) {
                        float ratio = img.dims(1) / (float)img.dims(0);
                        img = resize(img, w, w * ratio, AF_INTERP_BILINEAR);
                        img = img(span, seq(0, h-1), span, span);
                    } else {
                        float ratio = img.dims(0) / (float)img.dims(1);
                        img = resize(img, h * ratio, h, AF_INTERP_BILINEAR);
                        img = img(seq(0, h-1), span, span, span);
                    }
                    imgs(span, span, span, offset + file_count++) = img;
                    if(file_count > (n-1))
                        break;
                }
            } catch ( const std::exception & ex ) {
                std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
            }
        }
    }

    return file_count;
}


void load_catdog_dataset(array &data_train, array &labels_train,
                         array &data_valid, array &labels_valid,
                         const string base_path) {

    string train_path = base_path + "/train/";
    string valid_path = base_path + "/valid/";

    unsigned long file_count = 0;
    unsigned long dir_count = 0;
    unsigned long other_count = 0;
    unsigned long err_count = 0;

    ///
    /// Load training data from training directory
    fs::path cat_path = fs::system_complete(fs::path((train_path + "/cats").c_str()));
    int n_cat_imgs = dir_file_count(cat_path);
    n_cat_imgs = 100;
    if(n_cat_imgs > 2000) n_cat_imgs = 2000;

    fs::path dog_path = fs::system_complete(fs::path((train_path + "/dogs").c_str()));
    int n_dog_imgs = dir_file_count(dog_path);
    n_dog_imgs = 100;
    if(n_dog_imgs > 2000) n_dog_imgs = 2000;

    array timgs(img_sz, img_sz, 1, n_cat_imgs + n_dog_imgs);
    array tlabels(n_cat_imgs + n_dog_imgs);

    printf("loading %d cat images from %s\n", n_cat_imgs, train_path.c_str());
    load_dir_imgs_resize(cat_path, timgs, n_cat_imgs);
    tlabels(seq(0, n_cat_imgs-1)) = 0;

    printf("loading %d dog images from %s\n", n_dog_imgs, train_path.c_str());
    load_dir_imgs_resize(dog_path, timgs, n_dog_imgs, n_cat_imgs);
    tlabels(seq(n_cat_imgs, n_cat_imgs + n_dog_imgs-1)) = 1;

    data_train = timgs / 255.f;
    cout << data_train.dims() << endl;
    data_train = moddims(data_train, 4096, 200);
    labels_train = moddims(tlabels, 1, tlabels.dims(0));
    //labels_valid = tlabels;


    ///
    /// Load validation data from validation directory
    cat_path = fs::system_complete(fs::path((valid_path + "/cats").c_str()));
    n_cat_imgs = dir_file_count(cat_path);
    if(n_cat_imgs > 100) n_cat_imgs = 100;

    dog_path = fs::system_complete(fs::path((valid_path + "/dogs").c_str()));
    n_dog_imgs = dir_file_count(dog_path);
    if(n_dog_imgs > 100) n_dog_imgs = 100;

    array vimgs(img_sz, img_sz, 1, n_cat_imgs + n_dog_imgs);
    array vlabels(n_cat_imgs + n_dog_imgs);

    printf("loading %d cat images from %s\n", n_cat_imgs, valid_path.c_str());
    load_dir_imgs_resize(cat_path, vimgs, n_cat_imgs);
    vlabels(seq(0, n_cat_imgs-1)) = 0;

    printf("loading %d dog images from %s\n", n_dog_imgs, valid_path.c_str());
    load_dir_imgs_resize(dog_path, vimgs, n_dog_imgs, n_cat_imgs);
    vlabels(seq(n_cat_imgs, n_cat_imgs + n_dog_imgs-1)) = 1;

    data_valid = vimgs / 255.f;
    data_valid = moddims(data_valid, 4096, 200);
    labels_valid = moddims(vlabels, 1, vlabels.dims(0));
    //labels_valid = vlabels;
}

int main()
{
    af::info();

    array data_train, labels_train;
    array data_valid, labels_valid;

    load_catdog_dataset(data_train, labels_train, data_valid, labels_valid, "./dogscats");

    cout << data_train.dims() << endl;
    cout << labels_train.dims() << endl;
    cout << data_valid.dims() << endl;
    cout << labels_valid.dims() << endl;

    //nn::Sequential net = catdog_network();

    nn::Sequential net;

    net.add(nn::Linear(4096, 100));
    net.add(nn::Sigmoid());
    net.add(nn::Linear(100, 1));
    net.add(nn::Sigmoid());

    auto loss = nn::MeanSquaredError();

    //const double lr = 0.01;
    const double lr = 0.01;
    std::unique_ptr<optim::Optimizer> optim;
    optim = std::unique_ptr<optim::Optimizer>(new optim::AdamOptimizer(net.parameters(), lr));

    //TODO: random batches
    //int numSamples = 1;
    //int numSamples = data_train.dims(3);
    int numSamples = data_train.dims(1);

    Variable result, l;
    for (int i = 0; i < 1000; i++) {
        //for (int j = 0; j < numSamples/100; j++) {
        for (int j = 0; j < numSamples; j++) {

            net.train();
            optim->zeroGrad();

            //af::array in_batch = data_train(span, span, span, af::seq(j*100, j*100+99));
            //af::array out_batch = labels_train(af::seq(j*100, j*100+9));

            //af::array in_batch = data_train(span, span, span, af::seq(j*10, j*10+9));
            //af::array out_batch = labels_train(af::seq(j*10, j*10+9));

            af::array in_batch = data_train(span, j);
            af::array out_batch = labels_train(af::span, j);

            // Forward propagation
            result = net(nn::input(in_batch));
            // af_print(result.array());

            // Calculate loss
            l = loss(result, nn::noGrad(out_batch));

            // Backward propagation
            l.backward();

            // Update parameters
            optim->update();

            //af_print(l.array());
        }

        if ((i + 1) % 10 == 0) {
            net.eval();

            // Forward propagation
            result = net(nn::input(data_train));
            Variable test_loss = loss(result, nn::noGrad(labels_train));

            // Calculate loss
            // TODO: Use loss function
            printf("Average Error at iteration(%d) : %lf\n", i + 1, test_loss.array().scalar<float>());
            printf("\n\n");
        }
    }

    return 0;
}
