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
