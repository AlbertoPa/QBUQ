#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: moments                                                  C
#  Purpose: Estimate moments and low order statistics, including mean,   C
#           variance, skewness, and kurtosis.                            C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from numpy import sqrt
from scipy import zeros

def calc_low_order(qi, Ncell, Nsample, maxMomentOrder, weights):
    """ Return moments up to order of maxMomentOrder, mean,variance, skewness,
    and kurtosis.
    """
    Nmoments = maxMomentOrder+1
    moments = zeros((Ncell, Nmoments))
    for i in range(Ncell):
        for j in range(Nmoments):
            for k in range(Nsample):
                moments[i, j] += weights[k]*qi[i, k]**j
    for j in range(Nmoments-1, -1, -1):
        moments[:, j] /= moments[:, 0]
    mu = moments[:, 1]/moments[:, 0]
    sigSquare = moments[:, 2]/moments[:, 0]-mu**2
    sig = sqrt(sigSquare)
    gamma1 = (moments[:, 3]/moments[:, 0] \
              -3*mu*moments[:, 2]/moments[:, 0]+2*mu**3)/sig**3
    gamma2 = (moments[:, 4]/moments[:, 0] \
              -4*mu*moments[:, 3]/moments[:, 0] \
              +6*mu**2*moments[:, 2]/moments[:, 0]-3*mu**4)/sigSquare**2
    return moments, mu, sigSquare, gamma1, gamma2
