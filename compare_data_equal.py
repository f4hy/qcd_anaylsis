#!/usr/bin/env python2
import logging
import argparse
import os
import numpy as np
import pandas as pd
import math
import re


def compare_data_equal(options):
    """ Compare two data files to find the differences """
    logging.debug("Called with {}".format(options))
    f1 = open(options.files[0])
    f2 = open(options.files[1])

    abs_errors = []
    rel_errors = []

    while True:
        l1 = f1.readline()
        l2 = f2.readline()
        if l1 == "":
            print l1
            print f1.readline()
            break

        if l1 != l2:
            nums1 = [float(i) for i in l1.split()[1:]]
            nums2 = [float(i) for i in l2.split()[1:]]
            for i,j in zip(nums1,nums2):
                if abs_errors and abs(i-j) > max(abs_errors):
                    print l1, l2
                    print i, j, i-j
                abs_errors.append(abs(i-j))
                rel_errors.append(abs(i-j)/abs(i))

    print "largest differences"
    print max(abs_errors)
    print max(rel_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two data files to find the differences")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='input files')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    compare_data_equal(args)
