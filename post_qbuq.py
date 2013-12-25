#!/usr/bin python3

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvC
#                                                                        C
#  Module name: post_qbuq                                                C
#  Purpose: Main script to post-process the UQ data, including           C
#           estimation of moments and reconstruction of the probability  C
#           distribution function of system response.                    C
#                                                                        C
#  Author: Xiaofei Hu <xhu@iastate.edu>                                  C
#  Reviewer: Alberto Passalacqua <albertop@iastate.edu>                  C
#                                                                        C
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C

from qbuq_package.post_processing.post_functions import ask_post_job

print("This script post-processes the UQ data.")

# Ask users to choose whether to calculate low order statistics (estimate of 
# moments) or to reconstruct PDF of system response.
job = ask_post_job()
