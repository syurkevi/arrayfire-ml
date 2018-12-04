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
        class Pool2 : public Module
        {
        private:
            int m_wx;
            int m_wy;
            int m_sx;
            int m_sy;
            int m_px;
            int m_py;

        public:
            Pool2(int wx, int wy, int sx = 1, int sy = 1, int px = 0, int py = 0);
            autograd::Variable forward(const autograd::Variable &input);

        };
    }
}
