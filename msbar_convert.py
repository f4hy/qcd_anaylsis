#!/usr/bin/env python2
import logging
import argparse
import numpy as np

from alpha_s import get_alpha

pi = np.pi

def get_matm(m0, g):

    m2gv = 2.0
    a2gv = get_alpha(m2gv)
    As2gv = a2gv / pi

    c0 = As2gv**(4.0/9.0)*(1.0+0.895062*As2gv+
                              1.37143*As2gv**2.0+1.95168*As2gv**3.0)

    m1 = g
    for i in range(200):
        As = get_alpha(m1) / pi
        c1 = As**(4.0/9.0)*(1.0+0.895062*As+ 1.37143*As**2.0+1.95168*As**3.0)
        m1_out = (c1 / c0) * m0
        logging.debug("{} {} {} {}".format(i, m0, m1, m1_out))
        if np.isclose(m1,m1_out):
            return m1_out
        m1 = m1_out
    logging.error("did not converge")
    return np.nan


def main(options):

    result = get_matm(options.mass, options.guess)
    logging.info("found {} - > {}".format(options.mass, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert from msbar_2gev to m")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('mass', type=float,
                        help='list of scales to compute at')
    parser.add_argument('-g', "--guess", type=float,
                        help="Guess to converge to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.guess is None:
        logging.info("No guess given assuming same as input mass")
        args.guess = args.mass

    main(args)
