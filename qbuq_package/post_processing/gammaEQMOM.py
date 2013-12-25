#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: gammaEQMOM                                               C
#  Purpose: Reconstruct the the probability distribution function using  C
#           gamma EQMOM on interval [a, inf).                            C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#  Literature/Document Reference:                                        C
#  1. Yuan, C., Laurent, F., Fox, R.O., 2012. Journal of Aerosol Science C
#     51, 1-23.                                                          C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from scipy import array, spacing, sqrt, zeros
from scipy.linalg import det, norm
from qbuq_package.post_processing.adaptive_Wheeler import \
Wheeler_moments_adaptive

def gamma_EQMOM_general(weights, v, a, nnodes, rmin, eabs):
    """ Return weights, nodes, sigma, number of nodes, and iteration number
    using gamma EQMOM on general support [a, inf). Maximum number of nodes
    is 5.
    """
    # Check if number of nodes is suitable for gamma EQMOM.
    if nnodes > 5:
        print("Maximum number of nodes for gamma EQMOM exceeded. Program exits.")
        exit()
    elif nnodes <= 0:
        print("Non-positive number of nodes. Program exits.")
        exit()
    # Transform the interval to [0, inf)
    vn = v-a
    # Estimate moments using transformed data
    m = zeros(2*nnodes+1)
    for i in range(2*nnodes+1):
        for j in range(len(vn)):
            m[i] += weights[j]*vn[j]**i
    m = m/m[0]
    # Use gamma EQMOM on [0, inf) to determine nodes, weights, and sigma.
    w, x, sig, nout, flag = gamma_EQMOM(m, nnodes, rmin, eabs)
    # Transform nodes back to [a, inf)
    xout = x+a
    return w, x, xout, sig, nout, flag
    
def gamma_EQMOM(mom, nnodes, rmin, eabs):
    """ Return weights, nodes, sigma, number of nodes, and iteration number
    using gamma EQMOM for positive support [0,inf).
    """
    # Make sure number of nodes is no larger than 5.
    if nnodes > 5:
        print("Maximum number of nodes for gamma EQMOM exceeded. Program exits.")
        exit()
    elif nnodes <= 0:
        print("Non-positive number of nodes. Program exits.")
        exit()
    else:
        n = nnodes
    # 2n+1 moments are needed.
    nmom = 2*n+1
    if len(mom) < nmom:
        print("Not enough moments to generate "+str(n)+" nodes. "\
        "Program exits.")
        exit()
    else:
        mom1 = mom[:nmom]
    m = mom1.copy()
    w = zeros(n)
    x = zeros(n)
    macheps = spacing(1)
    itmax = 1000 # maximum number of iteration
    mtol = 1e-13 # tolerance of moments conservation
    ftol = 1e-14 # tolerance of target function
    stol = 1e-14 # tolerance of sigma
    ztol = 1e-14 # tolerance for function of zbrent method
    # flag for hankel detminant solution. 
    #1 means one sig from hankel matrix is computed.
    detflag = 0 
    flag = 0 # iteration number
    # check if moments are realizable
    if m[0] <= 0.0:
        sig = 0
        nout = 0
        print("Moments are not realizable, m[0] <= 0.")
        return w, x, sig, nout, flag
    # return 0 if the zero-th order moment is too small
    elif m[0] <= 1e-15:
        sig = 0
        nout = 0
        print("m[0] is too small. Return 0 solutions.")
        return w, x, sig, nout, flag
    # Find maximum value of sigma (total variance), used as upper bound
    sigmax = abs(m[2]/m[0]-m[1]/m[0])
    sigmaxx = sigmax
    if nmom > 3:
        sigmax = 0.75*m[1]/m[0]+0.25*sqrt((m[1]/m[0])**2+8*m[3]/m[1])
        sigmaxx = min(sigmax, sigmaxx)
    sigmax = sigmaxx
    # Initial guess of sigma: sig = 0, used as lower bound
    sold = 0
    sig = sold
    # Check if sig = 0 satisfies Hdet >= 0 or else use QMOM.
    for i in range(n, 0, -1):
        Hdet = gamma_targetHdet(m, sig, n)
        if Hdet < 0:
            n = n-1
            nmom = 2*n+1
            w = zeros(n)
            x = zeros(n)
        else:
            if n == 1:
                w = m[0]
                x = m[1]/m[0]
                nout = 1
                return w, x, sig, nout, flag
            break
    # sigposi is the sig value that makes Hdet >= 0, 
    # used as lower bound for Brent method.
    posinum = 0
    sigposi = []
    if Hdet >= -macheps and detflag == 0:
        sigposi.append(sig)
        posinum += 1
    else:
        # Use QMOM if cannot find n to satisfy Hdet >= 0
        w, x, nout, _ = \
        Wheeler_moments_adaptive(m[:2*nnodes], nnodes, rmin, eabs)
        sig = 0
        print("QMOM is used because of negative Hdet.")
        print("Hdet = "+str(Hdet))
        return w, x, sig, nout, flag
    # Compute f(sig) = fold first.
    ms = gamma_m2ms(m, sig, nmom-1)
    fold = gamma_targetfun(m, sig, n, nnodes, rmin, eabs)
    # Check if sig = 0 is the solution.
    if abs(fold) < ftol:
        w, x, nout, _ = Wheeler_moments_adaptive(m[:2*n], n, rmin, eabs)
        print("Sigma = 0 is the solution. QMOM is used.")
        return w, x, sig, nout, flag
    else:
        df = gamma_initial_df(ms, n)
        ds = -fold/df
        snew = sold+ds
        sig = min(snew, sigmax)
        sig = max(sig, 0)
        Hdet = gamma_targetHdet(m, sig, n)
        if  Hdet < -macheps:
            sig = gamma_zbrent(m, n, sigposi[0], sig, ztol)
        elif Hdet >= -macheps and detflag == 0:
            sigposi.append(sig)
            posinum += 1
        snew = sig;
        fnew = gamma_targetfun(m, snew, n, nnodes, rmin, eabs)
    # method = 0 for bounded secant method, and 1 for bisection.
    method = 0
    lam = 0.1
    dsmax = sigmaxx
    # Check if current sig is the solution, if not start iteration.
    if abs(fnew) < ftol:
        ms = gamma_m2ms(m, sig, nmom-1)
        w, x, nout, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
        msnew = gamma_calc_ms(w, x, 2*nnodes+1)
        mnew = gamma_ms2m(msnew, sig, 2*nnodes+1)
        # Check if moments are conserved.
        mdiff = (mnew-m)/m
        normmdiff = norm(mdiff)
        if normmdiff > mtol:
            print("m = "+str(m))
            print("mnew = "+str(mnew))
            print("mdiff = (mnew - m)/m = "+str(mdiff))
            print("norm(mdiff) = "+str(normmdiff))
            print("Moments are not conserved.")
        return w, x, sig, nout, flag
    else:
        df = (fnew-fold)/(snew-sold)
        if df >= 0:
            df = -1e-6
        ds = -fnew/df
        sig = snew+ds
        if fnew < 0:
            sigmax = snew
            sig = snew/2
            snew = 0
        sig = min(sig, sigmax)
        sig = max(sig, 0)
        Hdet = gamma_targetHdet(m, sig, n)
        if Hdet >= -macheps and detflag == 0:
            sigposi.append(sig)
            posinum += 1
        ds = sig-snew
    # Start iteration
    for j in range(1, itmax+1):
        ms = gamma_m2ms(m, sig, nmom-1)
        w, x, noutT, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
        Hdet = gamma_targetHdet(m, sig, noutT)
        if Hdet < -macheps and detflag == 0:
            # Use the largest sigposi as lower bound of Brent's method.
            sig_low = max(sigposi)
            sig_low = min(sig_low, snew)
            sig_det = gamma_zbrent(m, n, sig_low, sig, ztol)
            ms = gamma_m2ms(m, sig_det, nmom-1)
            w, x, _, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
            msnew = gamma_calc_ms(w, x, 2*nnodes+1)
            mnew = gamma_ms2m(msnew, sig_det, 2*nnodes+1)
            normmdiff_det = norm((mnew-m)/m)
            detflag = 1
        #Compute new value of f
        f = gamma_targetfun(m, sig, noutT, nnodes, rmin, eabs)
        df = f-fold
        #Check for convergence
        if abs(f) < ftol or abs(ds) < stol:
            if detflag == 1:
                sigtemp = sig
                ms = gamma_m2ms(m, sigtemp, nmom-1)
                w, x, _, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
                msnew = gamma_calc_ms(w, x, 2*nnodes+1)
                mnew = gamma_ms2m(msnew, sigtemp, 2*nnodes+1)
                mdiff = (mnew-m)/m
                normmdiff = norm(mdiff)
                Hdet = gamma_targetHdet(m, sigtemp, n)
                # Use better sigma compared with sigma calculated by zbrent.
                if Hdet >= 0:
                    if normmdiff_det < normmdiff:
                        sig = sig_det
                    else:
                        sig = sigtemp
                else:
                    sig = sig_det
            # Compute w and x using the final sig
            ms = gamma_m2ms(m, sig, nmom-1)
            w, x, nout, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
            msnew = gamma_calc_ms(w, x, 2*nnodes+1)
            mnew = gamma_ms2m(msnew, sig, 2*nnodes+1)
            # Check if moments are conserved.
            mdiff = (mnew-m)/m
            normmdiff = norm(mdiff)
            if normmdiff > mtol:
                print("m = "+str(m))
                print("mnew = "+str(mnew))
                print("mdiff = (mnew - m)/m = "+str(mdiff))
                print("norm(mdiff) = "+str(normmdiff))
                print("Moments are not conserved.")
            return w, x, sig, nout, flag
        if dsmax == 0 and f > 0:
            print("Target function has no zero!")
            msnew = gamma_calc_ms(w, x, 2*nnodes+1)
            mnew = gamma_ms2m(msnew, sig, 2*nnodes+1)
            mdiff = (mnew-m)/m
            normmdiff = norm(mdiff)
            if normmdiff > mtol:
                print("norm(mdiff) is larger than tolerance.")
                print(normmdiff)
            return w, x, sig, nout, flag
        # Compute new guess. If method = 0 and f > 0, use bounded secant method
        # Otherwise, use bisection method.
        if f > 0 and method == 0: # bounded secant method
            fnew = f
            snew = sig
            dfds = df/(snew-sold)
            ds = -fnew/dfds
            sig1 = snew+lam*ds
            sig1 = min(sig1, sigmax)
            sig1 = max(sig1, 0)
            dsmax = sigmaxx-sig1
            kk = 0
            while dsmax == 0 and kk <= itmax:
                kk += 1
                ds = lam*min(ds, sigmaxx-snew)/2
                sig1 = snew+ds
                sig1 = min(sig1, sigmax)
                sig1 = max(sig1, 0)
                dsmax = sigmaxx - sig1
            sig = sig1
            fold = fnew
            sold = snew
            if j > 10:
                lam = 1
        else: # bisection method
            if f > 0:
                fold = f
                sold = sig
                ds = (snew-sold)/2
            else:
                fnew = f
                snew = sig
                ds = (snew-sold)/2
                sigmax = sig
            sig = sold+ds
            sig = min(sig, sigmax)
            sig = max(sig, 0)
            Hdet = gamma_targetHdet(m, sig, n)
            if Hdet >= -macheps and detflag == 0:
                sigposi.append(sig)
                posinum += 1
            dsmax = sigmaxx-sig
            method = 1
        flag = j
    # Compute results if number of iteration exceed itmax.
    if detflag == 1:
        sigtemp = sig
        ms = gamma_m2ms(m, sigtemp, nmom-1)
        w, x, _, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
        msnew = gamma_calc_ms(w, x, 2*nnodes+1)
        mnew = gamma_ms2m(msnew, sigtemp, 2*nnodes+1)
        mdiff = (mnew-m)/m
        normmdiff = norm(mdiff)
        Hdet = gamma_targetHdet(m, sigtemp, n)
        if Hdet >= 0:
            if normmdiff_det < normmdiff:
                sig = sig_det
            else:
                sig = sigtemp
        else:
            sig = sig_det
    # Compute w and x using the final sig if itmax is reached.
    ms = gamma_m2ms(m, sig, nmom-1)
    w, x, nout, _ = Wheeler_moments_adaptive(ms[:2*n], n, rmin, eabs)
    msnew = gamma_calc_ms(w, x, 2*nnodes+1)
    mnew = gamma_ms2m(msnew, sig, 2*nnodes+1)
    mdiff = (mnew-m)/m
    normmdiff = norm(mdiff)
    print("Exceed maximum number of iterations! Increase itmax.")
    print("fnew = "+str(fnew))
    print("fold = "+str(fold))
    print("method = "+str(method)+". 0 is bound secant method. " \
    "1 is bisection method.")
    if normmdiff > mtol:
        print("norm(mdiff) is larger than tolerance.")
        print(normmdiff)
    return w, x, sig, nout, flag    

def gamma_targetHdet(m, sig, n):
    """ Return the minimum Hankel determinats for gamma EQMOM."""
    ms = gamma_m2ms(m, sig, 2*n)
    H00 = ms[0]
    H10 = ms[1]
    minHdet = min(H00, H10)
    if n >= 2:
        H01 = array(([ms[0], ms[1]], [ms[1], ms[2]]))
        H11 = array(([ms[1], ms[2]], [ms[2], ms[3]]))
        minHdet = min(minHdet, det(H01))
        minHdet = min(minHdet, det(H11))
    if n >= 3:
        H02 = array(([ms[0], ms[1], ms[2]], [ms[1], ms[2], ms[3]], \
        [ms[2], ms[3], ms[4]]))
        H12 = array(([ms[1], ms[2], ms[3]], [ms[2], ms[3], ms[4]], \
        [ms[3], ms[4], ms[5]]))
        minHdet = min(minHdet, det(H02))
        minHdet = min(minHdet, det(H12))
    if n >= 4:
        H03 = array(([ms[0], ms[1], ms[2], ms[3]], \
        [ms[1], ms[2], ms[3], ms[4]], [ms[2], ms[3], ms[4], ms[5]], \
        [ms[3], ms[4], ms[5], ms[6]]))
        H13 = array(([ms[1], ms[2], ms[3], ms[4]], \
        [ms[2], ms[3], ms[4], ms[5]], [ms[3], ms[4], ms[5], ms[6]], \
        [ms[4], ms[5], ms[6], ms[7]]))
        minHdet = min(minHdet, det(H03))
        minHdet = min(minHdet, det(H13))
    if n >= 5:
        H04 = array(([ms[0], ms[1], ms[2], ms[3], ms[4]], \
        [ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]]))
        H14 = array(([ms[1], ms[2], ms[3], ms[4], ms[5]], \
        [ms[2], ms[3], ms[4], ms[5], ms[6]], \
        [ms[3], ms[4], ms[5], ms[6], ms[7]], \
        [ms[4], ms[5], ms[6], ms[7], ms[8]], \
        [ms[5], ms[6], ms[7], ms[8], ms[9]]))
        minHdet = min(minHdet, det(H04))
        minHdet = min(minHdet, det(H14))
    if n >= 6:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
    return minHdet

def gamma_m2ms(m, sig, nmom):
    """ Return mom_star calculated from mom."""
    ms = zeros(nmom)
    ms[0] = m[0]
    ms[1] = m[1]
    if nmom >= 3:
        ms[2] = m[2]-sig*ms[1]
    if nmom >= 4:
        ms[3] = m[3]-3*sig*ms[2]-2*sig**2*ms[1]
    if nmom >= 5:
        ms[4] = m[4]-6*sig*ms[3]-11*sig**2*ms[2]-6*sig**3*ms[1]
    if nmom >= 6:
        ms[5] = m[5]-10*sig*ms[4]-35*sig**2*ms[3]-50*sig**3*ms[2]\
        -24*sig**4*ms[1]
    if nmom >= 7:
        ms[6] = m[6]-15*sig*ms[5]-85*sig**2*ms[4]-225*sig**3*ms[3]\
        -274*sig**4*ms[2]-120*sig**5*ms[1]
    if nmom >= 8:
        ms[7] = m[7]-21*sig*ms[6]-175*sig**2*ms[5]-735*sig**3*ms[4]\
        -1624*sig**4*ms[3]-1764*sig**5*ms[2]-720*sig**6*ms[1]
    if nmom >= 9:
        ms[8] = m[8]-28*sig*ms[7]-322*sig**2*ms[6]-1960*sig**3*ms[5]\
        -6769*sig**4*ms[4]-13132*sig**5*ms[3]-13068*sig**6*ms[2]\
        -5040*sig**7*ms[1]
    if nmom >= 10:
        ms[9] = m[9]-36*sig*ms[8]-546*sig**2*ms[7]-4536*sig**3*ms[6]\
        -22449*sig**4*ms[5]-67284*sig**5*ms[4]-118124*sig**6*ms[3]\
        -109584*sig**7*ms[2]-40320*sig**8*ms[1]
    if nmom >= 11:
        ms[10] = m[10]-45*sig*ms[9]-870*sig**2*ms[8]-9450*sig**3*ms[7]\
        -63273*sig**4*ms[6]-269325*sig**5*ms[5]-723680*sig**6*ms[4]\
        -1172700*sig**7*ms[3]-1026576*sig**8*ms[2]-362880*sig**9*ms[1]
    if nmom >= 12:
        print("Trying to use too many EQMOM nodes. Program exits")
        exit()
    return ms
    
def gamma_ms2m(ms, sig, nmom):
    """ Return mom calculated from mom_star."""
    m = zeros(nmom)
    m[0] = ms[0]
    m[1] = ms[1]
    m[2] = ms[2]+sig*ms[1]
    if nmom >= 4:
        m[3] = ms[3]+3*sig*ms[2]+2*sig**2*ms[1]
    if nmom >= 5:
        m[4] = ms[4]+6*sig*ms[3]+11*sig**2*ms[2]+6*sig**3*ms[1]
    if nmom >= 6:
        m[5] = ms[5]+10*sig*ms[4]+35*sig**2*ms[3]+50*sig**3*ms[2]\
        +24*sig**4*ms[1]
    if nmom >= 7:
        m[6] = ms[6]+15*sig*ms[5]+85*sig**2*ms[4]+225*sig**3*ms[3]\
        +274*sig**4*ms[2]+120*sig**5*ms[1]
    if nmom >= 8:
        m[7] = ms[7]+21*sig*ms[6]+175*sig**2*ms[5]+735*sig**3*ms[4]\
        +1624*sig**4*ms[3]+1764*sig**5*ms[2]+720*sig**6*ms[1]
    if nmom >= 9:
        m[8] = ms[8]+28*sig*ms[7]+322*sig**2*ms[6]+1960*sig**3*ms[5]\
        +6769*sig**4*ms[4]+13132*sig**5*ms[3]+13068*sig**6*ms[2]\
        +5040*sig**7*ms[1]
    if nmom >= 10:
        m[9] = ms[9]+36*sig*ms[8]+546*sig**2*ms[7]+4536*sig**3*ms[6]\
        +22449*sig**4*ms[5]+67284*sig**5*ms[4]+118124*sig**6*ms[3]\
        +109584*sig**7*ms[2]+40320*sig**8*ms[1]
    if nmom >= 11:
        m[10] = ms[10]+45*sig*ms[9]+870*sig**2*ms[8]+9450*sig**3*ms[7]\
        +63273*sig**4*ms[6]+269325*sig**5*ms[5]+723680*sig**6*ms[4]\
        +1172700*sig**7*ms[3]+1026576*sig**8*ms[2]+362880*sig**9*ms[1]
    if nmom >= 12:
        print("Too many EQMOM nodes try to be used. Program exits.")
        exit()
    return m

def gamma_calc_ms(w, x, nmom):
    """ Return mom_star calculated with weights and abscissas."""
    ms = zeros(nmom)
    try:
        n = len(w)
    except TypeError:
        n = 1
    if n == 1:
        for i in range(nmom):
            ms[i] = w*x**i
    else:
        for i in range(nmom):
            for j in range(n):
                ms[i] += w[j]*x[j]**i
    return ms
    
def gamma_targetfun(m, sig, n, nnodes, rmin, eabs):
    """ Return target function calculated from given moments and moments
    computed from guess of sig.
    """
    nmom = 2*n+1
    ms = gamma_m2ms(m, sig, nmom-1)
    w, x, _, _ = Wheeler_moments_adaptive(ms, n, rmin, eabs)
    ms_temp = gamma_calc_ms(w, x, nmom)
    m_temp = gamma_ms2m(ms_temp, sig, nmom)
    f = (m[nmom-1]-m_temp[nmom-1])/m[nmom-1] #Normalized to avoid small m[nom]
    if n < nnodes:
        nn = 2*nnodes+1
        ms = gamma_m2ms(m, sig, 2*nnodes)
        w, x, _, _ = Wheeler_moments_adaptive(ms, nnodes, rmin, eabs)
        ms_temp = gamma_calc_ms(w, x, nn)
        m_temp = gamma_ms2m(ms_temp, sig, nn)
        fold = 0
        for i in range(nn):
            f = (m[i]-m_temp[i])/m[i] #Normalized to avoid small m[i]
            f = max(f, fold)
            fold = f
    return f

def gamma_initial_df(ms, n):
    """ Return the initial approximate slope df to start iteration."""
    if n == 1:
        df = -ms[1]
    elif n == 2:
        df = -6*ms[3]
    elif n == 3:
        df = -15*ms[5]
    elif n == 4:
        df = -28*ms[7]
    elif n == 5:
        df = -45*ms[9]
    else:
        print("Trying to use too many EQMOM nodes!")
    return df

def gamma_zbrent(m, n, x1, x2, tol):
    """ Return the root b found by Brent's method lying between x1 and x2. 
    Decrease b until gamma_targetHdet > 0.
    """
    ITMAX = 100
    iimax = 100
    EPS = 1.0e-14
    a = x1
    b = x2
    c = x2
    fa = gamma_targetHdet(m, a, n)
    fb = gamma_targetHdet(m, b, n)
    if fa*fb > 0:
        print("Root must be bracketed in zbrent! Program exits.")
        exit()
    fc = fb
    for iter in range(1, ITMAX+1):
        if fb*fc > 0:
            c = a
            fc = fa
            d = b-a
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol1 = 2.0*EPS*abs(b)+0.5*tol
        xm = 0.5*(c-b)
        if abs(xm) <= tol1 or fb == 0.0:
            fb = gamma_targetHdet(m, b, n)
            ii = 0
            while fb < 0:
                b = b-abs(tol1)
                fb = gamma_targetHdet(m, b, n)
                ii += 1
                if ii > iimax:
                    print("Cannot find a sigma for which Hdet >= 0. "\
                    "Program exits.")
                    exit()
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb/fa
            if a == c:
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0*xm*q-abs(tol1*q)
            min2 = abs(e*q)
            if 2.0*p < min(min1, min2):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            if xm > 0:
                b += abs(tol1)
            else:
                b -= abs(tol1)
        fb = gamma_targetHdet(m, b, n)
    print("Maximum number of iterations exceeded in zbrent. Program exits.")
    exit()
