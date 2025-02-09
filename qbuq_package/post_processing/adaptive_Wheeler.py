#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: adaptive_Wheeler                                         C
#  Purpose: Generate weights and nodes using adaptive Wheeler algorithm. C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#  Literature/Document Reference:                                        C
#  1. Yuan, C., Fox, R.O., 2011. Journal of Computational Physics 230,   C
#  	  8216-8246.                                                         C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from scipy import sqrt, zeros
from scipy.linalg import eig

def Wheeler_moments_adaptive(mom, n, rmin, eabs):
    """ Return weights,  nodes, and number of nodes using adaptive Wheeler
    algorithm.
    """
    cutoff = 0
    werror = 0 # Return 1 if error occurs.
    # Check if moments are unrealizable.
    if mom[0] <= 0:
        print("Moments are not realizable, moment[0] <= 0. Program exits.")
        werror = 1
        exit()
    if n == 1 or mom[0] < rmin[0]:
        w = mom[0]
        x = mom[1]/mom[0]
        nout = 1
        return w, x, nout, werror
    # Set modified moments equal to input moments.
    nu = mom
    # Construct recurrence matrix
    ind = n
    a = zeros(ind)
    b = zeros(ind)
    sig = zeros((2*ind+1, 2*ind+1))
    for i in range(1, 2*ind+1):
        sig[1, i ]= nu[i-1]
    a[0] = nu[1]/nu[0]
    b[0] = 0
    for k in range(2, ind+1):
        for l in range(k, 2*ind-k+2):
            sig[k, l] = sig[k-1, l+1]-a[k-2]*sig[k-1, l]-b[k-2]*sig[k-2, l]
        a[k-1] = sig[k, k+1]/sig[k, k]-sig[k-1, k]/sig[k-1, k-1]
        b[k-1] = sig[k, k]/sig[k-1, k-1]
    # Find maximum n using diagonal element of sig
    for k in range(ind,1,-1):
        if sig[k,k] <= cutoff:
            n = k-1
            if n == 1:
                w = mom[0]
                x = mom[1]/mom[0]
                nout = 1
                return w, x, nout, werror
    # Use maximum n to re-calculate recurrence matrix
    a = zeros(n)
    b = zeros(n)
    w = zeros(n)
    x = zeros(n)
    sig = zeros((2*n+1, 2*n+1))
    for i in range(1, 2*n+1):
        sig[1, i ]= nu[i-1]
    a[0] = nu[1]/nu[0]
    b[0] = 0
    for k in range(2, n+1):
        for l in range(k, 2*n-k+2):
            sig[k, l] = sig[k-1, l+1]-a[k-2]*sig[k-1, l]-b[k-2]*sig[k-2, l]
        a[k-1] = sig[k, k+1]/sig[k, k]-sig[k-1, k]/sig[k-1, k-1]
        b[k-1] = sig[k, k]/sig[k-1, k-1]
    # Check if moments are not realizable (should never happen)
    if b.min() < 0:
        print("Moments in Wheeler_moments are not realizable! Program exits.")
        werror = 1
        exit()
    # Setup Jacobi matrix for n-point quadrature, adapt n using rmin and eabs
    for n1 in range(n,0, -1 ):
        if n1 == 1:
            w = mom[0]
            x = mom[1]/mom[0]
            nout = 1
            return w, x, nout, werror
        z = zeros((n1, n1))
        for i in range(n1-1):
            z[i, i] = a[i]
            z[i, i+1] = sqrt(b[i+1])
            z[i+1, i] = z[i, i+1]
        z[n1-1, n1-1] = a[n1-1]
        # Compute weights and abscissas
        eigenvalues, eigenvectors = eig(z)
        idx = eigenvalues.argsort()
        x = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        w = mom[0]*eigenvectors[0, :]**2
        dab = zeros(n1)
        mab = zeros(n1)
        for i in range(n1-1, 0, -1):
            dab[i] = min(abs(x[i]-x[0:i]))
            mab[i] = max(abs(x[i]-x[0:i]))
        mindab = min(dab[1:n1])
        maxmab = max(mab[1:n1])
        if n1 == 2:
            maxmab = 1
        # Adaptive conditions. When both satisfied, return the results.
        if min(w)/max(w) > rmin[n1-1] and mindab/maxmab > eabs:
            nout = n1
            return w, x, nout, werror
