"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2015 Dino Sejdinovic
"""

from kerpy.BagKernel import BagKernel
import numpy as np
from tools.GenericTests import GenericTests
from kerpy.GaussianKernel import GaussianKernel
from abc import abstractmethod

class LinearBagKernel(BagKernel):
    def __init__(self,data_kernel):
        BagKernel.__init__(self,data_kernel)
        
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "" + BagKernel.__str__(self)
        s += "]"
        return s
    
    @abstractmethod
    def compute_BagKernel_value(self,bag1,bag2):
        innerK=self.data_kernel.kernel(bag1,bag2)
        return np.mean(innerK[:])
    
    
if __name__ == '__main__':
    from tools.UnitTests import UnitTests
    UnitTests.UnitTestBagKernel(LinearBagKernel)