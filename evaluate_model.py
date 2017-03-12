#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import os
import numpy as np
import pandas as pd
import math
import re
from global_model2_0.global_fit_model2_0 import Model

from global_model2_0.fdssqrtms_models import * # noqa
from global_model2_0.single_heavy_fd_models import * # noqa
from global_model2_0.fd_models import * # noqa
from global_model2_0.pion_chiral_model import * # noqa
from global_model2_0.kaon_chiral_model import * # noqa


def eval_model(options):
    """ DESCRIPTION """
    logging.debug("Called with {}".format(options))

    header = options.boot_fit_file.readline().split(",")
    name = header[0].strip("# ")
    columns = [s.strip("\n ,") for s in header[1:]]

    df = pd.read_csv(options.boot_fit_file, sep=",", delimiter=",", names=columns)

    valid_models = {m.__name__: m for m in Model.__subclasses__()}
    logging.debug("valid models available {}".format(valid_models.keys()))
    model = valid_models[name]

    m = model([], options)

    try:
        evalmodes = m.evalmodes
    except AttributeError:
        evalmodes = [""]

    for mode in evalmodes:
        m.evalmode = mode
        means = df.mean()
        params = [means[i] for i in m.contlim_args]
        y = m.eval_fit(options.xpoint, *params)
        logging.info("{}{} on the mean {}".format(options.prefix, mode, y))
        values = []
        for i, row in df.iterrows():
            p = [row[n] for n in m.contlim_args]
            values.append(m.eval_fit(options.xpoint, *p))

        ponesigma = np.percentile(values, 84.1)
        monesigma = np.percentile(values, 15.9)
        ostring = "{}{}bootstraped mean: {}, std{}, +sigma{}, -sigma{}".format(options.prefix, mode, np.mean(values), np.std(values),
                                                                               ponesigma, monesigma)
        logging.info(ostring)
        if options.output_stub:
            filename = options.output_stub + "_" + mode + ".txt"
            with open(filename, 'w') as ofile:
                ofile.write(ostring + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a fit model")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-x", "--xpoint", type=float, required=True,
                        help="xvalue to evaluate the model at")
    parser.add_argument("--prefix", type=str, default="",
                        help="use this prefix when outputting")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('--err', nargs='?', type=argparse.FileType('w'),
                        default=None)
    parser.add_argument('boot_fit_file', metavar='f', type=argparse.FileType('r'),
                        help='input file')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.err is not None:
        root = logging.getLogger()
        ch = logging.StreamHandler(args.err)
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        root.addHandler(ch)

    eval_model(args)
