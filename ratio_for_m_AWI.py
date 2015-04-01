#!/usr/bin/env python2
import argparse
import logging
import numpy as np
import pandas as pd
import re
import readinput
import os
import sys
import os
sys.path.append( "/home/bfahy/develop/effectivemass")
import build_corr
import correlator
import configtimeobj


def lines_without_comments(filename, comment="#"):
    from cStringIO import StringIO
    s = StringIO()
    with open(filename) as f:
        for line in f:
            if not line.startswith(comment):
                s.write(line)
    s.seek(0)
    return s


def removecomma(s):
    return int(s.strip(','))


def myconverter(s):
    try:
        return np.float(s.strip(','))
    except:
        return np.nan


def determine_type(txt):
    firstline = txt.readline()
    txt.seek(0)
    if "(" in firstline and ")" in firstline:
        logging.debug("paren_complex file type detected")
        return "paren_complex"
    if "," in firstline:
        logging.debug("comma file type detected")
        return "comma"
    logging.debug("space sperated file assumed")
    return "space_seperated"


def read_file(filename):
    with open(filename, 'r') as f:
        first_line = f.readline()
    names = [s.strip(" #") for s in first_line.split(",")[0:-2]]
    txt = lines_without_comments(filename)
    filetype = determine_type(txt)
    if filetype == "paren_complex":
        df = pd.read_csv(txt, delimiter=' ', names=names,
                         converters={1: parse_pair, 2: parse_pair})
    if filetype == "comma":
        df = pd.read_csv(txt, sep=",", delimiter=",", names=names, skipinitialspace=True,
                         delim_whitespace=True, converters={0: removecomma, 1: myconverter, 2: myconverter})
    if filetype == "space_seperated":
        df = pd.read_csv(txt, delimiter=' ', names=names)
    return df

def read_full_correlator(filename, emass=False, eamp=False):
    try:
        cor = build_corr.corr_and_vev_from_files_pandas(filename, None, None)
    except AttributeError:
        logging.info("Failed to read with pandas, reading normal")
        cor = build_corr.corr_and_vev_from_files(filename, None, None)

    # times = cor.times
    # data = [cor.average_sub_vev()[t] for t in times]
    # errors = [cor.jackknifed_errors()[t]  for t in times]

    # d = {"time": times, "correlator": data, "error": errors, "quality": [float('NaN') for t in times]}
    # df = pd.DataFrame(d)
    return cor



def compute_m_AWI(axialfname, psuedofname, options):

    A = read_full_correlator(axialfname)
    P = read_full_correlator(psuedofname)
    print A
    print P
    Aasv = A.average_sub_vev()
    Pasv = P.average_sub_vev()

    m = options.mass
    T = options.period

    m_AWI = {}
    vev = {}
    for cfg in A.configs[0:10]:
        print cfg
        vev[cfg] = 0.0
        ratio = {}
        for t in A.times:
            # numerator = np.sqrt(A.get(config=cfg, time=t) / (np.exp(-1.0*m * t) - np.exp(-1.0*m *(T-t))))
            # denominator =  np.sqrt(P.get(config=cfg, time=t) / (np.exp(-1.0*m * t) + np.exp(-1.0*m *(T-t))))
            if options.funct == "div":
                numerator = (A.get(config=cfg, time=t) / (np.exp(-1.0*m * t) - np.exp(-1.0*m *(T-t))))
                denominator =  (P.get(config=cfg, time=t) / (np.exp(-1.0*m * t) + np.exp(-1.0*m *(T-t))))
            elif options.funct == "raw":
                numerator = A.get(config=cfg, time=t)
                denominator =  P.get(config=cfg, time=t)
            if np.isinf(numerator / denominator):
                ratio[t] = float("NaN")
            else:
                ratio[t] = float(numerator / denominator)
            # print numerator, denominator, ratio
            # exit()
            # if not np.isinf(ratio):
            #     m_AWI[t] = numerator / denominator
        m_AWI[cfg] = ratio

    print m_AWI
    print vev

    m_AWI_cto = configtimeobj.Cfgtimeobj.fromDataDict(m_AWI)
    print m_AWI_cto.times
    print m_AWI_cto.configs
    print m_AWI_cto.average_over_configs()
    print m_AWI_cto.jackknifed_errors()
    values = m_AWI_cto.average_over_configs()
    errors = m_AWI_cto.jackknifed_errors()

    if options.out_stub:
        outfilename = options.out_stub + "_{}".format(options.funct) + ".out"
        logging.info("Writing m_AWI to {}".format(outfilename))
        with open(outfilename,"w") as outfile:
            for t in A.times:
                outfile.write("{}, {}, {}\n".format(t,values[t], errors[t]))
    else:
        print m_AWI

    print "done"

if __name__ == "__main__":
    functs = ["raw", "div"]

    parser = argparse.ArgumentParser(description="average data files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--out_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-A", "--axial", type=str, required=False,
                        help="axial correlator")
    parser.add_argument("-P", "--psuedo", type=str, required=False,
                        help="psuedo scalar correlator")
    parser.add_argument("-m", "--mass", type=float, required=True,
                        help="mass")
    parser.add_argument("-f", "--funct", type=str, default="raw", required=False, choices=functs,
                        help="function to use")
    parser.add_argument("-T", "--period", type=int, required=True,
                        help="period")
    args = parser.parse_args()


    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Computing M_AQI for: Axial{}    P{}".format(args.axial, args.psuedo))

    if args.out_stub:
        outdir = os.path.dirname(args.out_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist, atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)


    compute_m_AWI(args.axial, args.psuedo, args)
