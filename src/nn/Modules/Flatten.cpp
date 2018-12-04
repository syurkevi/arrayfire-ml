/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/autograd/Functions.hpp>

#include <af/nn/Init.hpp>
#include <af/nn/Modules/Flatten.hpp>

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Flatten::Flatten(bool batched_input) :
            batched_input_(batched_input)
        {
        }

        Variable Flatten::forward(const Variable &input)
        {
            if(batched_input_) {
                af::dim4 odims = input.dims();
                return moddims(input, af::dim4(odims[0] * odims[1] * odims[2], odims[3]));
            }
            return flat(input);
        }
    }
}
