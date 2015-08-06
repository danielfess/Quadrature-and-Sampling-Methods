"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2015 Dino Sejdinovic
"""

from kerpy.BagKernel import BagKernel
from abc import abstractmethod
from numpy import exp

class GaussianBagKernel(BagKernel):
    def __init__(self,data_kernel,sigma=0.1):
        BagKernel.__init__(self,data_kernel)
        self.width=sigma
        
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "width="+ str(self.width)
        s += ", " + BagKernel.__str__(self)
        s += "]"
        return s
    
    @abstractmethod
    def compute_BagKernel_value(self,bag1,bag2):
        return exp(-0.5 * self.data_kernel.estimateMMD(bag1,bag2) / self.width ** 2)
    
    
if __name__ == '__main__':
    from tools.UnitTests import UnitTests
    UnitTests.UnitTestBagKernel(GaussianBagKernel)