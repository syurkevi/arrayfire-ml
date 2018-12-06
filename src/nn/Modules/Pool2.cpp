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
#include <af/nn/Modules/Pool2.hpp>

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Pool2::Pool2(int wx, int wy, int sx, int sy, int px, int py) :
            m_wx(wx),
            m_wy(wy),
            m_sx(sx),
            m_sy(sy),
            m_px(px),
            m_py(py)
        {}

        Variable Pool2::forward(const Variable &input)
        {
            auto res = maxpool2(input, m_wx, m_wy, m_sx, m_sy, m_px, m_py);
            return res;
        }
    }
}
