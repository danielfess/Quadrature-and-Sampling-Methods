"""
Copyright (c) 2013-2014 Heiko Strathmann, Dino Sejdinovic
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
"""
from kerpy.GaussianKernel import GaussianKernel
import numpy as np
from kerpy.MaternKernel import MaternKernel

class UnitTests():
    @staticmethod
    def UnitTestBagKernel(which_bag_kernel):
            num_bagsX = 20
            num_bagsY = 30
            shift = 2.0
            dim = 3
            bagsize = 50
            qvar = 0.6
            baglistx = list()
            baglisty = list()
            for _ in range(num_bagsX):
                muX = np.sqrt(qvar) * np.random.randn(1, dim)
                baglistx.append(muX + np.sqrt(1 - qvar) * np.random.randn(bagsize, dim))
            for _ in range(num_bagsY):
                muY = np.sqrt(qvar) * np.random.randn(1, dim)
                muY[:, 0] = muY[:, 0] + shift
                baglisty.append(muY + np.sqrt(1 - qvar) * np.random.randn(bagsize, dim))
            data_kernel = MaternKernel(1.0)
            bag_kernel = which_bag_kernel(data_kernel)
            bag_kernel.show_kernel_matrix(baglistx + baglisty)
            bagmmd = bag_kernel.estimateMMD(baglistx, baglisty)
            print 'successfully computed mmd on bags: ', bagmmd
            print 'unit test ran for ', bag_kernel.__str__()
