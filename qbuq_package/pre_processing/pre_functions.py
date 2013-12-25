#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: pre_functions                                            C
#  Purpose: Interactive functions for pre_qbuq.py                        C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from os import close as os_close, getcwd, makedirs
from os.path import isfile
from scipy import loadtxt, savetxt, sum, zeros
from shutil import move
from tempfile import mkstemp
from qbuq_package.pre_processing import pre_algorithms

#========================================================================#
# 1. Generate samples for the uncertain parameters.                      #
#========================================================================#

def gen_samples():
    """ Returns weights and nodes of the uncertain parameters. """
    d = ask_dimension()
    print("A text file of weights and nodes of samples has been generated "\
          "successfully.")
    return d

def ask_dimension():
    """ Reads the dimension of user's problem."""
    print("Please select the dimension of your problem or press q to exit.")
    print("1 -- Univariate problem")
    print("2 -- Bivariate problem")
    d = input(">>>")
    if d == '1':
        # If univariate problem is chosen, ask if the distribution is known.
        ask_known_1D_distribution()
    elif d == '2':
        # If the problem is bivariate, CQMOM is used.
        weights, node1, node2 = CQMOM_2D()
        savetxt('quadrature_weights_nodes.txt', (weights, node1, node2))
    elif str.lower(d) == 'q':
        exit()
    else:
        # If none of "1", "2" or "q" is selected, ask users to select again.
        print("The script can now be used only for univariate or bivariate " \
        "problems. Please select again.")
        d = ask_dimension()
    return d

def ask_known_1D_distribution():
    """ Asks if the distribution of the variable is known."""
    print("Is the distribution of the variable known (y/n)?")
    ans = input(">>>")
    if str.lower(ans) == 'y':
        # Gaussian quadrature formulae is used for known distribution.
        # Ask what kind distribution it is.
        ask_1D_distribution()
    elif str.lower(ans) == 'n':
        # Adaptive Wheeler algorithm for unknown distribution.
        print("Adaptive Wheeler algorithm is used for unknown distribution.")
        w, x, nout, werror = adaptive_Wheeler()
        # Save quadrature weights and nodes to a text file.
        savetxt('quadrature_weights_nodes.txt', (w, x))
    elif str.lower(ans) == 'q':
        exit()
    else:
        # If none of "y", "n", or "q" is chosen, ask users to select again.
        print("Please press y for known distribution or n for unknown " \
        "distribution. Press q to quit.")
        ask_known_1D_distribution()

def ask_1D_distribution():
    """ Read the distribution selected by users."""
    print("Please select the distribution of the uncertain parameter or " \
    "press q to quit.")
    print("1 -- Uniform distribution")
    print("2 -- Gaussian distribution")
    d = input(">>>")
    if d == '1':
        w, x = uniform_sample()
        # Save quadrature weights and nodes to a text file.
        savetxt('quadrature_weights_nodes.txt', (w, x))
    elif d == '2':
        w, x = gauss_sample()
        # Save quadrature weights and nodes to a text file.
        savetxt('quadrature_weights_nodes.txt', (w, x))
    elif str.lower(d) == 'q':
        exit()
    else:
        print("The script can now be used only for uniform or Gaussian " \
        "distribution. Please select again.")
        ask_1D_distribution()

def uniform_sample():
    """ Returns weights and nodes of uniform distribution on interval [a,b]."""
    N = ask_sample_number()
    # Calculate recurrence coefficients for monic Legendre polynomials.
    ab = pre_algorithms.r_legendre(N)
    # Calculate quadrature weights and nodes for bounded interval [0,1].
    xw = pre_algorithms.gauss(N, ab)
    # Ask the boundary values for bounded interval [a,b].
    a, b = ask_uniform_interval()
    # Calculate nodes for uniform distribution on bounded interval [a.b].
    x = (b-a)*xw[:, 0]/2+(b+a)/2
    # Normalize w so that sum(w) = 1.
    w = xw[:, 1]/sum(xw[:, 1])
    if min(x) < 0:
        print("Negative nodes are generated. If unphysical, please adjust "\
              "your distribution.")
    return w, x

def gauss_sample():
    """ Returns weights and nodes of Gaussian distribution with mean mu and
    sigma sig.
    """
    N = ask_sample_number()
    # Calculate recurrence coefficients for monic Hermite polynomials.
    ab = pre_algorithms.r_hermite(N)
    # Calculate quadrature weights and nodes for standard normal distribution.
    xw = pre_algorithms.gauss(N, ab)
    # Ask mean and standard deviation.
    print("Please enter the mean value of the uncertain parameter.")
    mu = ask_num()
    print("Please enter the standard deviation of the uncertain parameter.")
    sig = ask_num()
    # Calculate quadrature nodes for arbitary Guassian distribution with mean mu
    # and standard deviation sig.
    x = sig*xw[:, 0]+mu
    # Normalize w so that sum(w) = 1.
    w = xw[:, 1]/sum(xw[:, 1])
    if min(x) < 0:
        print("Negative nodes are generated. If unphysical, please adjust "\
              "your distribution.")
    return w, x

def adaptive_Wheeler():
    """ Returns weights and nodes using adaptive Wheeler algorithm with users'
    input conditions.
    """
    N = ask_sample_number()
    print("Moments from 0th to "+str(2*N-1)+"th order are needed.")
    mom = ask_moments()
    # Check if moments are enough to generate N nodes.
    if len(mom)<2*N:
        print("Moments to "+str(len(mom))+"th order are not enough to "\
        "generate "+str(N)+" samples.")
        print("Please decrease number of samples or use another moment file.")
        w, x, nout, werror = adaptive_Wheeler()
    else:
        # Ask for rmin (ratios of minimum weights to maximum weights) and
        # eabs (minimum distance between distinct nodes).
        rmin, eabs = ask_para_Wheeler(N)
        w, x, nout, werror = \
        pre_algorithms.Wheeler_moments_adaptive(mom, N, rmin, eabs)
    return w, x, nout, werror

def CQMOM_2D():
    """ Returns weights and nodes using 2D CQMOM algorithm with users' input
    conditions.
    """
    print("For uncertain parameter 1")
    nodex = ask_sample_number()
    print("For uncertain parameter 2")
    nodey = ask_sample_number()
    # Ask if bivariate distribution is known
    mom = ask_known_2D_distribution(nodex, nodey)
    # Check if moments are enough to generate required number of nodes.
    if mom.shape[0]<2*nodex:
        print("The moments to "+str(mom.shape[0]-1)+"th order are not enough "\
        "to generate "+str(nodex)+" samples in the first direction. Please "\
        "decrease the number of samples, use another moment file, or press "\
        "q to quit.")
        weights, node1, node2 = CQMOM_2D()
    elif mom.shape[1]<2*nodey:
        print("The moments to "+str(mom.shape[1]-1)+"th order are not enough "\
        "to generate "+str(nodey)+" samples in the second direction. Please "\
        "decrease the number of samples, use another moment file, or press "\
        "q to quit.")
        weights, node1, node2 = CQMOM_2D()
    else:
        rmin1, rmin2, eabs = ask_para_CQMOM_2D(nodex, nodey)
        weights, node1, node2 \
        = pre_algorithms.gen_CQMOM_2D(mom, rmin1, rmin2, eabs, nodex, nodey)
    if min(node1) < 0:
        print("Negative nodes in the first direction are generated. "\
        "If unphysical, please adjust your distribution.")
    elif min(node2)<0:
        print("Negative nodes in the second direction are generated. "\
        "If unphysical, please adjust your distribution.")
    return weights, node1, node2

def ask_known_2D_distribution(nodex, nodey):
    """ Returns moments needed in CQMOM for bivariate distribution."""
    print("Is the bivariate distribution known (y/n)?")
    ans = input(">>>")
    if str.lower(ans) == 'y':
        print("Only bivariate Gaussian distribution is implemented for now. "\
              "Press any key to continue or press q to quit.")
        d = input(">>>")
        if str.lower(d) == 'q':
            exit()
        else:
            pass
        mu1, mu2, sigma1, sigma2, rho = ask_coeff_2D_Gaussian()
        moments = pre_algorithms.MGF_2D_Gaussian(nodex, nodey, \
                                                 mu1, mu2, sigma1, sigma2, rho)
    elif str.lower(ans) == 'n':
        moments = ask_moments()
    elif str.lower(ans) == 'q':
        exit()
    else:
        print("Please press y for known distribution or n for unknown " \
        "distribution. Press q to quit.")
        moments = ask_known_2D_distribution()
    return moments

def ask_sample_number():
    """ Returns number of samples need to generate."""
    print('How many samples do you need to generate? Press q to quit.')
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

def ask_moments():
    """ Returns moments read from local text file."""
    print("Is the moments text file in the current work directory (y/n)?")
    m = input(">>>")
    if str.lower(m) == 'y':
        print("Please enter the name of the moments text file. "\
        "(Eg. moments.txt)")
        name = input(">>>")
        if str.lower(name) == 'q':
            exit()
        else:
            # Check if the file exists
            check_file(name)
            moments = loadtxt(name)
    elif str.lower(m) == 'n':
        print("Please enter the path of the moments text file. "\
              "(Eg. /home/usr/run/moments.txt)")
        name = input(">>>")
        if str.lower(name) == 'q':
            exit()
        else:
            # Check if the file exists
            check_file(name)
            moments = loadtxt(name)
    elif str.lower(m) == 'q':
        exit()
    else:
        print("Please press y for yes or n for no. Press q to quit.")
        moments = ask_moments()
    return moments

def ask_para_Wheeler(N):
    """ Returns rmin and eabs for adaptive Wheeler algorithm."""
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

def ask_coeff_2D_Gaussian():
    """ Returns the coefficients bivariate Gaussian distribution."""
    print("Please enter the mean value of uncertain parameter 1.")
    mu1 = ask_num()
    print("Please enter the standard deviation of uncertain parameter 1.")
    sigma1 = ask_num()
    print("Please enter the mean value of uncertain parameter 2.")
    mu2 = ask_num()
    print("Please enter the standard deviation of uncertain parameter 2.")
    sigma2 = ask_num()
    print("Please enter the correlation coefficient rho.")
    rho = ask_num()
    return mu1, mu2, sigma1, sigma2, rho

def ask_para_CQMOM_2D(nodex, nodey):
    """ Returns rmin1, rmin2, and eabs for 2D CQMOM."""
    print("Please enter "+str(nodex)+" ratios of minimum weights to "\
              "maximum weights (rmin1) for parameter 1.")
    rmin1 = zeros(nodex)
    for i in range(nodex):
        print("Please enter rmin1["+str(i)+"].")
        rmin1[i] = ask_num()
    print("Please enter "+str(nodey)+" ratios of minimum weights to "\
          "maximum weights (rmin2) for parameter 2.")
    rmin2 = zeros(nodey)
    for i in range(nodey):
        print("Please enter rmin2["+str(i)+"].")
        rmin2[i] = ask_num()
    print("Please enter the minimum distance between distinct nodes (eabs).")
    eabs = ask_num()
    return rmin1, rmin2, eabs

def ask_uniform_interval():
    """ Returns the bounded interval for arbitary uniform distribution."""
    print("Please enter the minimum value of the uncertain parameter.")
    a = ask_num()
    print("Please enter the maximum value of the uncertain paramter.")
    b = ask_num()
    if a >= b:
        print("Lower bound \'a\' must be SMALLER than upper bound \'b\'. "\
        "Please enter the interval again.")
        a, b = ask_uniform_interval()
    return a, b

def ask_num():
    """ Returns a number entered by users."""
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif isnumber(n) is False:
        print("Please enter a number or press q to quit.")
        n = ask_num()
    return float(n)

def isnumber(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def ask_int():
    """ Returns an integer entered by users."""
    n = input(">>>")
    if str.lower(n) == 'q':
        exit()
    elif str.isdigit(n) is False:
        print("Please enter an integer. Press q to quit.")
        n = ask_int()
    num = int(n)
    return num

def check_file(filename):
    """ Checks if a file exists. Return if true, else program exits."""
    if isfile(filename) != True:
        print(filename+ " does NOT exist. Program exits.")
        exit()
    else:
        return

#========================================================================#
# 2. Generate MFIX input files.                                          #
#========================================================================#

def gen_input(d):
    """ Generates MFIX input files for each sample."""
    print("Start generating MFIX input files.")
    # Ask for the head of the run name.
    name_head = ask_name_head()
    # Ask for the keywords needed to be changed in mfix.dat
    if d == '1':
        obj = ask_keyword_1D()
        print(obj)
    elif d == '2':
        obj = ask_keyword_2D()
        print(obj)
    # Load quadrature weights and nodes
    check_file('quadrature_weights_nodes.txt')
    xw = loadtxt('quadrature_weights_nodes.txt')
    # Generate mfix.dat for each sample, stored in separate folders.
    gen_mfix(name_head, obj, xw)
    print('MFIX input files are generated successfully.')

def ask_name_head():
    """ Returns the head of the run name (case sensitive). This name + the sample 
    number is the name of the run directory and the real run name for each
    sample.
    """
    print("Please enter the head of the run name (case sensitive).")
    name_head = input(">>>")
    return name_head
    
def ask_keyword_1D():
    """ Returns a tuple containing keywords using the value of the uncertain
    parameter.
    """
    # Create a blank list to store the keywords using the value of the parameter.
    obj = []
    print("How many keywords use the value of the uncertain parameter?")
    n = ask_int()
    if n == 0:
        print("No MFIX input file is generated because no keyword uses " \
              "the value of the uncertain parameter. Program exits.")
        exit()
    else:
        for i in range(n):
            print("Please enter keyword No. "+str(i+1)+".")
            keyword = input(">>>")
            obj.append(keyword)
    obj_key = tuple(obj), 
    return obj_key

def ask_keyword_2D():
    """ Returns a tuple containing keywords using values of uncertain parameters.
    """
    # Use lists in a list to store keywords for each parameter.
    obj= [[], []]
    print("How many keywords use the value of uncertain parameter 1?")
    n1 = ask_int()
    if n1 == 0:
        print("No keyword uses the value of uncertain parameter 1.")
    else:
        for i  in range(n1):
            print("Please enter keyword No. "+str(i+1)+" of uncertain " \
                  "parameter 1.")
            keyword = input(">>>")
            obj[0].append(keyword)
    print("How many keywords use the value of uncertain parameter 2?")
    n2 = ask_int()
    if n2 == 0:
        print("No keyword uses the value of uncertain parameter 2.")
        if len(obj[0]) == 0:
            print("No MFIX input file is generated because no keyword uses "\
              "values of the two uncertain parameters. Program exits.")
            exit()
    else:
        for i  in range(n2):
            print("Please enter keyword No. "+str(i+1)+" of uncertain " \
                  "parameter 2.")
            keyword = input(">>>")
            obj[1].append(keyword)
    obj_key = tuple(tuple(x) for x in obj)
    return obj_key
    
def gen_mfix(name_head, obj, xw):
    """ Replaces the value of the keywords in the basic mfix.dat file with the
    sample values, and stored the new mfix.dat files in separated folders.
    """
    cf = getcwd()
    # First keyword in mfix.dat that needs to be replaced is the run_name.
    name = 'run_name'
    check_file(cf+'/mfix.dat')
    with open(cf+'/mfix.dat', 'r') as old_file, \
    open('list_of_cases.txt', 'w') as case_list:
        # Check how many samples are generated.
        if len(xw.shape) == 1:
            iter = 1
        else:
            iter = xw.shape[1]
        # Start iterations to replace values of keywords in mfix.dat
        for i in range(iter):
            # Write case run_name in order in list_of_cases.txt
            case_list.write(name_head+str(i)+'\n')
            # Python dictionary is used to store values for keywords
            value_dict = {}
            for j in range(len(obj)):
                if iter == 1:
                    value_dict[obj[j]] = xw[j+1]
                else:
                    value_dict[obj[j]] = xw[j+1, i]
            # Make new folders for each sample
            new = cf+'/'+name_head+str(i)
            makedirs(new)
            fh, abs_path = mkstemp()
            with open(abs_path, 'w') as new_file:
                # Start checking each line of basic mfix.dat to replace values
                # if matched.
                for line in old_file:
                    data_line = line.strip()
                    # Skip blank and comment lines
                    if data_line == '':
                        new_file.write(data_line+'\n')
                    elif data_line[0] == '#' or data_line[0] == '!':
                        new_file.write(data_line+'\n')
                    # Replace run_name for each sample
                    elif str.lower(data_line)[0:len(name)] == str.lower(name):
                        data_line = data_line[0:len(name)]+' = \''+name_head+\
                        str(i)+'\''
                        new_file.write(data_line+'\n')
                    # Search and replace values for selected keywords
                    else:
                        flag = 0
                        for obj1 in obj:
                            for obj2 in obj1:
                                if str.lower(data_line)[0:len(obj2)] == \
                                str.lower(obj2):
                                    # Use flag to mark if values are replaced
                                    flag = 1
                                    data_line = data_line[0:len(obj2)]+' = '+\
                                    str(value_dict[obj1])
                                    print(data_line)
                                    new_file.write(data_line+'\n')
                        # Copy the line if no keywords matched
                        if flag != 1:
                            new_file.write(data_line+'\n')
            os_close(fh)
            # Move the finished mfix.dat to the sample folder
            move(abs_path, new+'/mfix.dat')
            old_file.seek(0)


