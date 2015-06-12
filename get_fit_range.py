#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import os
import numpy as np
import pandas as pd
import math
import re


ranges={}
ranges[(64, "PP", "ud-ud", 0, 0)]=(15,27)
ranges[(64, "PP", "ud-ud", 0, 1)]=(10,25)
ranges[(64, "PP", "ud-ud", 0, 2)]=(8,20)
ranges[(64, "PP", "ud-ud", 1, 1)]=(8,25)
ranges[(64, "PP", "ud-ud", 2, 2)]=(6,25)

ranges[(64, "PP", "ud-s", 0, 0)]=(15,27)
ranges[(64, "PP", "ud-s", 0, 1)]=(10,25)
ranges[(64, "PP", "ud-s", 0, 2)]=(8,20)
ranges[(64, "PP", "ud-s", 1, 1)]=(8,25)
ranges[(64, "PP", "ud-s", 2, 2)]=(6,25)


def get_fit_range(options):
    """ get fit range """
    logging.debug("Called with {}".format(options))

    try:
        output = ranges[(options.period, options.operator, options.flavor, options.smearing1, options.smearing2)]
        print "{} {}".format(*output)
    except KeyError:
        raise SystemExit("no fit range defined")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="~/develop/anaysis_iroiro/")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('period', type=int, help='PERIOD')
    parser.add_argument('operator', type=str, help='OPERATOR')
    parser.add_argument('flavor', type=str, help='FLAVOR')
    parser.add_argument('smearing1', type=int, help='smearing1')
    parser.add_argument('smearing2', type=int, help='smearing2')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    get_fit_range(args)
