#! /usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: pre_algorithms                                           C
#  Purpose: Algorithms to generate samples for uncertain parameter(s).   C
#                                                                        C
#  Author: Xiaofei Hu                      Date:                         C
#  Reviewer:                                                             C
#                                                                        C
#  Literature/Document Reference:                                        C
#  Gautschi, W., 2004. Oxford University Press, New York.                C
#  Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 1992. C
#  Cambridge University Press, Cambridge.                                C
#  Yuan, C., Fox, R.O., 2011. Journal of Computational Physics 230,      C
#  8216-8246.                                                            C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from scipy import column_stack, diag, dot, insert, newaxis, pi, spacing, sqrt
from scipy import savetxt, zeros
from scipy.linalg import eig, solve
from sympy import diff, exp as exp_sym, symbols
from sympy.matrices import Matrix, matrix2numpy, zeros as zeros_sym

#========================================================================#
# 1. Univariate problem                                                  #
#========================================================================#

#------------------------------------------------------------------------#
# 1.1 Gauss quadrature for known distributions.                          #
#------------------------------------------------------------------------#

def r_hermite(N):
    """" Return the first n recurrence coefficients for monic Hermite
    polynomials.
    """
    A = zeros(N)
    B = zeros(N)
    B[0] = sqrt(pi)
    for i in range(1, N):
        B[i] = i/2
    ab = column_stack((A[:, newaxis], B[:, newaxis]))
    return ab

def r_legendre(N):
    """" Return the first n recurrence coefficients for monic Legendre
    polynomials.
    """
    A = zeros(N)
    B = zeros(N)
    B[0] = 2
    for i in range(1, N):
        B[i] = 1/(4-i**(-2))
    ab = column_stack((A[:, newaxis], B[:, newaxis]))
    return ab

def gauss(N, ab):
    """" Return the gauss quadrature weights and nodes."""
    J = zeros((N, N))
    for i in range(0, N):
        J[i, i] = ab[i, 0]
    for i in range(1, N):
        J[i, i-1] = sqrt(ab[i, 1])
        J[i-1, i] = J[i, i-1]
    eigenvalues, eigenvectors = eig(J)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    weights = ab[0, 1]*eigenvectors[0, :]**2
    xw = column_stack((eigenvalues[:, newaxis], weights[:, newaxis]))
    return xw

#------------------------------------------------------------------------#
# 1.2 Adaptive Wheeler algorithm for unknown distributions.              #
#------------------------------------------------------------------------#

def Wheeler_moments_adaptive(mom, n, rmin, eabs):
    """ Return weights,  nodes, and number of nodes using adaptive Wheeler
    algorithm.
    """
    cutoff = 0
    werror = 0 # Return 1 if error occurs.
    # Check if moments are unrealizable.
    if mom[0] <= 0:
        print("Moments are NOT realizable, moment[0] <= 0.0. Program exits.")
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

#========================================================================#
# 2. Bivariate problem                                                   #
#                                                                        #
# Conditional quadrature method of moments is used.                      #
#========================================================================#

def gen_CQMOM_2D(mom, rmin1, rmin2, eabs, nodex, nodey):
    """ Return weights and nodes in each direction using 2D CQMOM.""" 
    if nodex == 0 or nodey == 0:
        print('Cannot use 0 nodes!')
        exit()
    n = zeros(nodex*nodey)
    u1 = zeros(nodex*nodey)
    u2 = zeros(nodex*nodey)
    small = 10*spacing(1)
    if mom[0, 0] <= 0:
        print("Moments are not realizable, moment[0] <= 0.0! Program exits.")
        exit()
    elif mom[0, 0] < rmin1[0]:
        n[0] = mom[0, 0]
        u1[0] = mom[1, 0]/mom[0, 0]
        u2[0] = mom[1, 1]/mom[0, 0]
        nout1 = 1
        nout2 = 1
        n = n[0]
        u1 = u1[0]
        u2 = u2[0]
        return n, u1, u2
    elif mom[0, 0] < small*100:
        nodex = min(2, nodex)
        nodey = min(2, nodey)
    # 1D quadrature in first direction
    m1 = mom[0:2*nodex, 0]
    w, x, nout1, werror = Wheeler_moments_adaptive(m1, nodex, rmin1, eabs)
    if werror > 0:
        print("1D quadrature failed on first step! Program exits.")
        exit()
    # Condition on second direction
    if nout1 == 1:
        mom_call = mom[0, 1:2*nodey]
        mc = mom_call.copy()
        mom_con = insert(mc, 0, 1)
        m2 = mom[0, 0]*mom_con
        node12 = 0
        w2, x2, nout2, werror = Wheeler_moments_adaptive(m2, nodey, rmin2, eabs)
        if werror > 0:
            print("1D quadrature failed on second step! Program exits.")
            exit()
        else:
            pass
        if nout2 == 1:
            n[0] = w*w2/mom[0, 0]
            u1[0] = x
            u2[0] = x2
            node12 = 1
        else:
            for j in range(nout2):
                n[node12+j] = w*w2[j]/mom[0, 0]
                u1[node12+j] = x
                u2[node12+j] = x2[j]
            node12 = node12+nout2
    else:
        # Construct the Vandermonde matrix
        A = zeros((nout1, nout1))
        for i in range(nout1):
            for j in range(nout1):
                A[i, j] = x[j]**i
        # Diagonal matrix of weights
        Nall = diag(w)
        diagN = Nall[0:nout1, 0:nout1]
        # Moments used to solve for conditional moments
        mom_call = zeros((nout1, 2*nodey-1))
        for i in range(2*nodey-1):
            mom_call[:, i] = mom[:nout1, i+1]
        mom_c = mom_call.copy()
        x1 = x[0:nout1]
        mom_c1 = mom_c.copy()
        # Solve for conditional moments
        for i in range(2*nodey-1):
            temp = mom_c[:, i]
            q = temp.copy()
            # Vandermonde matrix solver
            mom_c1[:, i] = vanderls(x1, q, nout1)
            # Use iterative method to reduce round-off errors
            err = dot(A, mom_c1[:, i])-q
            mom_c1[:, i] = mom_c1[:, i]-vanderls(x, err, nout1)
            err = dot(A, mom_c1[:, i])-q
            maxerror = max(abs(err))
            if maxerror > small:
                print(maxerror)
        mc = solve(diagN, mom_c1)
        mom_con = insert(mc, 0, 1, axis = 1)
        # Use conditional moments in Wheeler adaptive algorithm to obtain
        # weights and abscissas in the second direction.
        node12 = 0
        for i in range(nout1):
            m2 = mom[0, 0]*mom_con[i, :]
            w2, x2, nout2, werror \
            = Wheeler_moments_adaptive(m2, nodey, rmin2, eabs)
            if werror > 0:
                print("1D quadrature failed on second step! Program exits.")
                exit()
            else:
                pass
            if nout2 == 1:
                n[node12] = w[i]*w2/mom[0, 0]
                print(n)
                u1[node12] = x[i]
                print(u1)
                u2[node12] = x2
                print(u2)
                node12 = node12+nout2
                print(node12)
            else:
                for j in range(nout2):
                    n[node12+j] = w[i]*w2[j]/mom[0, 0]
                    u1[node12+j] = x[i]
                    u2[node12+j] = x2[j]
                node12 = node12+nout2
    n = n[:node12]
    u1 = u1[:node12]
    u2 = u2[:node12]
    return n, u1, u2

# Moment generation function of bivariate Gaussian distribution
def MGF_2D_Gaussian(N1, N2, mu1, mu2, sigma1, sigma2, rho):
    """ Return moments of bivariate Gaussian distribution using moment
    generation function.
    """
    t1, t2 = symbols('t1, t2', real=True)
    t1m, t2m = symbols('t1m, t2m', real=True)
    sig1, sig2, r = symbols('sig1, sig2, rho',  real=True) 
    sig12 = r*sig1*sig2
    t = Matrix([t1, t2])
    mu = Matrix([t1m, t2m])
    sig = Matrix(([sig1**2, sig12], [sig12,  sig2**2]))
    ep = t.T*mu+t.T*sig*t/2
    mgf = exp_sym(ep[0])
    mom = zeros_sym(2*N1, 2*N2)
    for i in range(N1):
        for j in range(1, 2*N2):
            m1 = diff(mgf, t1, i)
            m2 = diff(m1, t2, j)
            mom[i, j] = m2.subs([(t1, 0),  (t2, 0)])
    for i in range(2*N1):
        m1 = diff(mgf, t1, i)
        mom[i, 0] = m1.subs([(t1, 0),  (t2, 0)])
    mom1 = mom.subs([(t1m, mu1),  (t2m, mu2),  (sig1, sigma1), (sig2, sigma2),\
                       (r, rho)])
    # Transform the moments from matrix in sympy to array in numpy
    moments = matrix2numpy(mom1)
    savetxt('MGF_2D_Gaussian_moments.txt', moments)
    print("A text file containing moments of bivariate Gaussian distribution "\
          "has been generated successfully.")
    return moments

# Vandermonde matrix solver
def vanderls(x, q, n):
    """ Return solution of Vandermonde linear system x**(k-1)*w = q."""
    c = zeros(n)
    w = zeros(n)
    if n == 1:
        w[0] = q[0]
        print(w)
    else:
        c[n-1] = -x[0]
        for i in range(1, n):
            xx = -x[i]
            for j in range(n-1-i, n-1):
                c[j] = c[j]+xx*c[j+1]
            c[n-1] = c[n-1]+xx
        for i in range(n):
            xx = x[i]
            t = 1
            b = 1
            s = q[n-1]
            for k in range(n-1, 0, -1):
                b = c[k]+xx*b
                s = s+q[k-1]*b
                t = xx*t+b
            w[i] = s/t
    return w
