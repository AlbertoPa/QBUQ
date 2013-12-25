#! /usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: GaussianEQMOM                                            C
#  Purpose: Reconstruct the the probability distribution function using  C
#           2-node Gaussian EQMOM on (-inf, inf).                        C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#  Literature/Document Reference:                                        C
#  Chalons, C., Fox, R.O., Massot, M., 2010. Center for Turbulence       C
#  Research, Stanford University. Proceedings of the Summer Program      C 
#  2010, 347-358.                                                        C
#  Desjardins, O., Fox, R.O., Villedieu, P., 2008. Journal of            C
#  Computational Physics 227, 2514-2539.                                 C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from scipy import spacing, sqrt, zeros

def GaussianEQMOM2node(weights, v):
    """ Return weights, nodes, sigma, and number of nodes using 2-node Gaussian
    EQMOM for the whole real set.
    """
    # Five moments are needed for 2-node Gaussian EQMOM.
    nmom = 5
    m = zeros(nmom)
    for i in range(nmom):
        for j in range(len(v)):
            m[i] += weights[j]*v[j]**i
    m = m/m[0]
    macheps = spacing(1)
    w = zeros(2)
    x = zeros(2)
    # Check if moments are realizable.
    if m[0] <= 0.0:
        sig = 0
        nout = 0
        print("Moments are not realizable, m[0] <= 0.0")
        return w, x, sig, nout
    elif m[0] <= 1e-15:
        sig = 0
        nout = 0
        print("m[0] is too small. Return 0 solutions.")
        return w, x, sig, nout
    # Compute central moments e, q, and eta.
    e = (m[0]*m[2]-m[1]**2)/m[0]**2
    q = ((m[3]*m[0]**2-m[1]**3)-3*m[1]*(m[0]*m[2]-m[1]**2))/m[0]**3
    eta = (-3*m[1]**4+m[4]*m[0]**3-4*m[0]**2*m[1]*m[3]+6*m[0]*m[1]**2*m[2])\
    /m[0]**4
    # Check if system of moments is well-defined.
    if e <= 0.0:
        sig = 0
        nout = 0
        print("The system is not well-defined, e <= 0.0")
        return w, x, sig, nout
    elif abs(q) < macheps:
        if eta > 3*e**2:
            sig = 0
            nout = 0
            print("The system is not well-defined, eta > 3e^2 when q = 0.")
            return w, x, sig, nout
        elif abs(eta-3*e**2) <= macheps:
            x = m[1]/m[0]
            w = m[0]
            sig = sqrt(e)
            nout = 1
            return w, x, sig, nout
        elif eta >= e**2:
            x0 = ((3*e**2-eta)/2)**(0.25)
            x[0] = -x0+m[1]/m[0]
            x[1] = x0+m[1]/m[0]
            w[0] = 0.5*m[0]
            w[1] = w[0]
            sig = sqrt(e-x0**2)
            nout = 2
            return w, x, sig, nout
        else:
            sig = 0
            nout = 0
            print("The system is not well-defined, eta < e^2 when q = 0.")
            return w, x, sig, nout
    elif eta <= e**2+q**2/e:
        sig = 0
        nout = 0
        print("The system is not well-defined, eta <= e^2 + q^2/e.")
        return w, x, sig, nout
    # Return one node if e is small.
    if e <= 1e-8:
        w = m[0]
        x = m[1]/m[0]
        sig = 0
        nout = 1
        return w, x, sig, nout
    # Calculate sigma**2.
    c1 = (eta/e**2-3)/6
    c2 = q**2/(4*e**3)
    tmp = sqrt(c1**3+c2**2)
    c3 = (tmp.real+c2)**(1/3)
    temp2 = c3-c1/c3
    temp2 = max(temp2, 0)
    temp2 = min(temp2, 1)
    sig1 = e*(1-temp2)
    # Return one node if sigma**2 = e.
    if abs(sig1-e) < macheps:
        w = m[0]
        x = m[1]/m[0]
        sig = sqrt(e)
        nout = 1
        return w, x, sig, nout
    # Calculate weights and nodes.
    x0 = (q/2)/sqrt(q**2+4*e**3)
    w[0] = (0.5+x0)*m[0]
    w[1] = (0.5-x0)*m[0]
    x[0] = m[1]/m[0]-sqrt(w[1]/w[0]*e)
    x[1] = m[1]/m[0]+sqrt(w[0]/w[1]*e)
    sig = sqrt(sig1)
    nout = 2
    return w, x, sig, nout
