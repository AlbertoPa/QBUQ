#! /usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: post_functions                                           C
#  Purpose: Interactive functions for post_qbuq.py                       C
#                                                                        C
#  Author: Xiaofei Hu                      Date:                         C
#  Reviewer:                                                             C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from os.path import isfile
from os import getcwd
from scipy import loadtxt, savetxt, zeros
from qbuq_package.post_processing.moments import calc_low_order
from qbuq_package.post_processing.betaEQMOM import beta_EQMOM_general
from qbuq_package.post_processing.gammaEQMOM import gamma_EQMOM_general
from qbuq_package.post_processing.GaussianEQMOM import GaussianEQMOM2node

# Ask users to select a job.
def ask_post_job():
    """ Read users' job selection."""
    print("Please select your job or press q to quit.")
    print("1 -- Calculate low order statistics.")
    print("2 -- Reconstruct the probability distribution function.")
    job = input(">>>")
    if job == '1':
        print("Low order statistics will be calculated.")
        # Get current working directory
        cf = getcwd()
        # Read data extracted by post_mfix
        qi, Ncell, Nsample = read_low_order_sample_data(cf)
        # Load quadrature weights for each sample
        wx = loadtxt(cf+'/quadrature_weights_nodes.txt')
        weights = wx[0, :]
        # Check if data for all samples are loaded.
        if len(weights) != Nsample:
            print("Number of samples does NOT match number of weights. " \
            "Program exits.")
            exit()
        # Read maximum order of moments
        maxMomentOrder = ask_maxMomentOrder()
        # Estimation of moments and low order statistics
        moments, mean, variance, skewness, kurtosis \
        = calc_low_order(qi, Ncell, Nsample, maxMomentOrder, weights)
        # Save moments and low order statistics as fext files
        savetxt('moments.txt', moments)
        savetxt('mean.txt', mean)
        savetxt('variance.txt', variance)
        savetxt('skewness.txt', skewness)
        savetxt('kurtosis.txt', kurtosis)
        print("Moments and low order statistics are estimated successfully.")
    elif job == '2':
        print("PDF will be reconstructed.")
        # Method can be chosen from beta-EQMOM, gamma-EQMOM, and 2-node
        # Gaussian EQMOM
        ask_recon_method()
    elif str.lower(job) == 'q':
        exit()
    else:
        print("Please select again.")
        job = ask_post_job()
    return job
    
#========================================================================#
# 1. Calculate low order statistics.                                     #
#========================================================================#

def read_low_order_sample_data(cf):
    """ Read data of quantity of interest of each sample as an array for 
    calculation of low order statistics.  Return the array, number of cells, 
    and number of samples.
    """
    # Check if list_of_cases.txt exists.
    check_file("list_of_cases.txt")
    # Start reading data case by case
    with open(cf+'/list_of_cases.txt', 'r') as lc:
        print("Please enter the file name of the quantity of interest or "\
        "press q to quit.")
        fname = input(">>>")
        if str.lower(fname) == 'q':
            exit()
        # Use the first case to determine Ncell
        case0 = lc.readline().strip()
        filename = cf+'/'+case0+'/'+fname
        check_file(filename)
        # Count number of header lines in data file extracted by post_mfix
        # so that these lines can be skipped when data is loaded here.
        with open(filename, 'r') as f:
            h = count_header(f)
        Nsample = ask_sample_number()
        # In data extracted by post_mfix, the last column is the value of
        # system response.
        temp = loadtxt(filename, skiprows=h, usecols=(-1, ))
        # Use length of the array 'temp' as Ncell. In case of just one cell,
        # temp is a number instead of an array, len(temp) will give a TypeError.
        # Use try statement to avoid this error.
        try:
            Ncell = len(temp)
        except TypeError:
            Ncell = 1
        qi = zeros((Ncell, Nsample))
        qi[:, 0] = temp
        i = 0
        # Start reading data for other cases
        for line in lc:
            i += 1 # use i as a marker of case number
            if i >= Nsample:
                print("Case number in \'list_of_cases.txt\' does NOT match " \
                "number of samples. Program exits.")
                exit()
            cases = line.strip()
            filename = cf+'/'+cases+'/'+fname
            check_file(filename)
            with open(filename, 'r') as f:
                h = count_header(f)
            qi[:, i] = loadtxt(filename, skiprows=h, usecols=(-1, ))
        # Check if case number in 'list_of_cases.txt' matches Nsample.
        if i != Nsample-1:
            print("Case number in \'list_of_cases.txt\' does NOT match " \
            "number of samples. Program exits.")
            exit()
    return qi, Ncell, Nsample

def ask_maxMomentOrder():
    """ Return user defined maximum order of moments."""
    print("What is the maximum order of moments? At least 4th order moment "\
    "is needed to calculate low order statistics. Press q to quit.")
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif str.isdigit(n) is False:
        print("Please enter an positive integer. Press q to quit.")
        n = ask_maxMomentOrder()
    num = int(n)
    if num < 4:
        print("At least 4th order moment is needed to calculate low order "\
        "statistics.Please enter again or press q to quit.")
        n = ask_maxMomentOrder()
        num = int(n)
    return num

def check_file(filename):
    """ Check if a file exists. Return if true, else program exits."""
    if isfile(filename) != True:
        print(filename+ " does NOT exist. Program exits.")
        exit()
    else:
        return

# Count number of header lines in data file extracted by post_mfix so that
# these lines can be skipped when data file is loaded. 
def count_header(f):
    """ Return an integer represents number of header lines."""
    n = 0
    for line in f:
        data_line = line.strip()
        if data_line == '':
            n += 1
        elif data_line[0] == '-':
            break
        elif data_line[0]<'0' or data_line[0] > '9':
            n += 1
        else:
            break
    return n

def ask_sample_number():
    """ Return number of samples."""
    print('How many samples do you have? Press q to quit.')
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif str.isdigit(n) is False:
        print("Please enter an positive integer. Press q to quit.")
        n = ask_sample_number()
    elif n == '0':
        print("Number of samples cannot be 0. Please enter an positive "\
              "integer. Press q to quit.")
        n = ask_sample_number()
    num = int(n)
    return num

#========================================================================#
# 2. Reconstruct the probability distribution function.                  #
#========================================================================#

def ask_recon_method():
    """ Read users' selection of reconstruction method, including beta EQMOM,
    gamma EQMOM, and 2-node Gaussian EQMOM.
    """
    print("Please select your method or press q to quit.")
    print("1 -- Beta EQMOM")
    print("2 -- Gamma EQMOM")
    print("3 -- 2 node Gaussian EQMOM")
    cf = getcwd()
    # Load input quadrature weights
    wx = loadtxt(cf+'/quadrature_weights_nodes.txt')
    input_weights = wx[0, :]
    method = input(">>>")
    if method == '1':
        print("Beta EQMOM will be used.")
        check_file("list_of_cases.txt")
        # Ask number of nodes to reconstruct the PDF
        nnodes = ask_node_number()
        # Ask parameters of adaptive Wheeler method
        rmin, eabs = ask_para_Wheeler(nnodes)
        Nsample = ask_sample_number()
        # Check if Nsample matches number of samples generated by pre_qbuq.py.
        if len(input_weights) != Nsample:
            print("Number of samples does NOT match number of weights. " \
            "Program exits.")
            exit()
        # Read data at a specific location for reconstruction of the PDF
        qi = read_recon_sample_data(cf, Nsample)
        # Ask the bounded value for beta EQMOM
        a, b = ask_beta_support(qi)
        # Use beta EQMOM to reconstruct the PDF
        w, x, xout, sig, nout, flag = \
        beta_EQMOM_general(input_weights, qi, a, b, nnodes, rmin, eabs)
        print(str(nout)+" beta EQMOM node(s) generated. Files containing "\
        "weights, nodes, sigma, and the data set used for beta EQMOM are "\
        "generated successfully.")
        print("sigma = "+str(sig))
        print("weights = \n"+', '.join(map(str, w)))
        print("nodes on [0, 1] = \n"+', '.join(map(str,x)))
        print("nodes on [a, b] = \n"+', '.join(map(str,xout)))
        savetxt('betaEQMOM_weights_nodes.txt', (w, x, xout))
        savetxt('betaEQMOM_sigma.txt', (sig, ))
        savetxt('data_set_for_betaEQMOM.txt', (qi, ))
    elif method == '2':
        print("Gamma EQMOM will be used.")
        check_file("list_of_cases.txt")
        nnodes = ask_node_number()
        rmin, eabs = ask_para_Wheeler(nnodes)
        Nsample = ask_sample_number()
        if len(input_weights) != Nsample:
            print("Number of samples does NOT match number of weights. " \
            "Program exits.")
            exit()
        qi = read_recon_sample_data(cf, Nsample)
        # Ask the lower bound of gamma EQMOM.
        a = ask_gamma_support(qi)
        # Use gamma EQMOM to reconstruct the PDF
        w, x, xout, sig, nout, flag = \
        gamma_EQMOM_general(input_weights, qi, a, nnodes, rmin, eabs)
        print(str(nout)+" gamma EQMOM node(s) generated. Files containing "\
        "weights, nodes, sigma, and the data set used for gamma EQMOM are "\
        "generated successfully.")
        print("sigma = "+str(sig))
        print("weights = \n"+', '.join(map(str, w)))
        print("nodes on [0, inf) = \n"+', '.join(map(str,x)))
        print("nodes on [a, inf) = \n"+', '.join(map(str,xout)))
        savetxt('gammaEQMOM_weights_nodes.txt', (w, x, xout))
        savetxt('gammaEQMOM_sigma.txt', (sig, ))
        savetxt('data_set_for_gammaEQMOM.txt', (qi, ))
    elif method == '3':
        print("2 node Gaussian EQMOM will be used.")
        check_file("list_of_cases.txt")
        Nsample = ask_sample_number()
        if len(input_weights) != Nsample:
            print("Number of samples does NOT match number of weights. " \
            "Program exits.")
            exit()
        qi = read_recon_sample_data(cf, Nsample)
        # Use 2-node Gaussian EQMOM to reconstruct the PDF
        w, xout, sig, nout = \
        GaussianEQMOM2node(input_weights, qi)
        print(str(nout)+" Gaussian EQMOM node(s) generated. Files containing "\
        "weights, nodes, sigma, and the data set used for Gaussian EQMOM are "\
        "generated successfully.")
        print("sigma = "+str(sig))
        print("weights = \n"+', '.join(map(str, w)))
        print("nodes = \n"+', '.join(map(str, xout)))
        savetxt('GaussianEQMOM_weights_nodes.txt', (w, xout))
        savetxt('GaussianEQMOM_sigma.txt', (sig, ))
        savetxt('data_set_for_GaussianEQMOM.txt', (qi, ))
    elif str.lower(method) == 'q':
        exit()
    else:
        print("Please select again.")
        method = ask_recon_method()
    return method

def ask_node_number():
    """ Return number of nodes needed to reconstruct PDF."""
    print("How many nodes do you need to reconstruct the PDF? Maximum number "\
    "of nodes is 5. Press q to quit.")
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif str.isdigit(n) is False:
        print("Please enter an positive integer. Press q to quit.")
        n = ask_node_number()
    elif n == '0':
        print("Number of nodes cannot be 0. Please enter an positive "\
              "integer. Press q to quit.")
        n = ask_node_number()
    num = int(n)
    if num > 5:
        print("Maximum number of nodes cannot exceed 5. Please enter again or "\
        "press q to quit.")
        n = ask_node_number()
        num = int(n)
    return num

def ask_para_Wheeler(N):
    """ Return rmin and eabs for adaptive Wheeler algorithm."""
    print("Please enter "+str(N)+" ratios of minimum weights to maximum "\
      "weights (rmin). See documents for details about these ratios.")
    rmin = zeros(N)
    for i in range(N):
        print("Please enter rmin["+str(i)+"].")
        rmin[i] = ask_num()
    print("Please enter the minimum distance between distinct nodes (eabs). " \
          "See documents for details about this parameter.")
    eabs = ask_num()
    return rmin, eabs

def read_recon_sample_data(cf, Nsample):
    """ Read data of quantity of interest of each sample as an array for 
    PDF reconstruction.  Return the array.
    """
    qi = zeros(Nsample)
    with open(cf+'/list_of_cases.txt', 'r') as lc:
        print("Please enter the file name of the quantity of interest or "\
        "press q to quit.")
        fname = input(">>>")
        if fname == 'q':
            exit()
        # Use data of the first case to check if data of only one specific
        # location is contained in the file.
        case0 = lc.readline().strip()
        filename = cf+'/'+case0+'/'+fname
        check_file(filename)
        # Count number of header lines in data file extracted by post_mfix
        # so that these lines can be skipped when data is loaded here.
        with open(filename, 'r') as f:
            h = count_header(f)
        temp = loadtxt(filename, skiprows=h)
        if temp.shape != ():
            print("Reconstruction can be done at only one specific "\
            "location each time. Please check your data. Program exits.")
            exit()
        qi[0] = temp
        # Start loading data of other cases
        i = 0
        for line in lc:
            i += 1 # use i as a marker of case number
            if i >= Nsample:
                print("Case number in \'list_of_cases.txt\' does NOT match " \
                "number of samples. Program exits.")
                exit()
            cases = line.strip()
            filename = cf+'/'+cases+'/'+fname
            with open(filename, 'r') as f:
                h = count_header(f)
            qi[i] = loadtxt(filename, skiprows=h)
        # Check if case number in 'list_of_cases.txt' matches Nsample.
        if i != Nsample-1:
            print("Case number in \'list_of_cases.txt\' does NOT match " \
            "number of samples. Program exits.")
            exit()
    return qi

def ask_num():
    """ Return a number entered by users."""
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif isnumber(n) is False:
        print("Please enter a number or press q to quit.")
        n = ask_num()
    return float(n)

def isnumber(s):
    """ Return True if s is a number, false if not."""
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def ask_beta_support(qi):
    """ Return the bounded interval for beta EQMOM. """
    print("Please choose your method to set the bounded interval for beta "\
    "EQMOM. Press q to quit.")
    print("1 -- Use the minimum and maximum value of the data as the interval.")
    print("2 -- Set the bounded interval manually.")
    m = input(">>>")
    if m == '1':
        a = min(qi)
        b = max(qi)
    elif m == '2':
        print("Please enter the lower bound of the interval.")
        a = ask_num()
        print("Please enter the upper bound of the interval.")
        b = ask_num()
        # Make sure a is smaller than b
        if a >= b:
            print("Lower bound \'a\' must be SMALLER than upper bound \'b\'. "\
            "Please choose your method and enter the interval again.")
            a, b = ask_beta_support(qi)
    elif str.lower(m) == 'q':
        exit()
    else:
        print("Please choose your method or press q to quit.")
        a, b = ask_beta_support(qi)
    return a, b
    
def ask_gamma_support(qi):
    """ Return the bounded interval for beta EQMOM. """
    print("Please choose your method to set the interval for gamma EQMOM. "\
    "Press q to quit.")
    print("1 -- Use the minimum value of the data as the lower bound.")
    print("2 -- Set the lower bound manually.")
    m = input(">>>")
    if m == '1':
        a = min(qi)
    elif m == '2':
        print("Please enter the lower bound of the interval.")
        a = ask_num()
    elif str.lower(m) == 'q':
        exit()
    else:
        print("Please choose your method or press q to quit.")
        a = ask_gamma_support(qi)
    return a
