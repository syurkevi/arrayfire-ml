/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Flatten : public Module
        {
        private:
            bool batched_input_;
        public:
            Flatten(bool batched_input = true);
            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
