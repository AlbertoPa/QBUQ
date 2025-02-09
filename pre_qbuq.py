#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: pre_qbuq                                                 C
#  Purpose: Main script to generate samples for uncertain parameters     C
#           and corresponding MFIX input file mfix.dat for each sample.  C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from qbuq_package.pre_processing.pre_functions import gen_samples, gen_input

print("This script generates samples of the uncertain parameters and MFIX " \
      "input file mfix.dat for each sample.")
# Generate quadrature nodes and weights for the uncertain parameter(s).
d = gen_samples()
# Use generated nodes to generate corresponding MFIX input files.
gen_input(d)
