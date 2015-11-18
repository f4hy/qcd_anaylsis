#!/usr/bin/env python2
import argparse
import logging
import numpy as np
import pandas as pd
import re
import readinput
import os

from residualmasses import residual_mass
from ensamble_info import  data_params

from ensamble_info import Zs, Zv


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


def decay_constant(filename, options):
    logging.info("Reading bootstraped fit values from {}".format(filename))

    df = read_file(filename)

    dp = data_params(filename)

    beta = re.search("_b(4\.[0-9]*)_", filename).group(1)

    masses = {}
    masses["ud"] = float(re.search("mud([0-9]\.[0-9]*)_", filename).group(1))


    strange_mass = re.search("ms([a-z0-9.]+)", filename).group(1)
    try:
        masses["s"] = float(strange_mass)
    except ValueError:
        logging.warning("Found strange mass to be {}".format(strange_mass))
        if strange_mass == "shifted":
            with open("ms/{}.smass".format(beta)) as smassfile:
                masses["s"] = float(smassfile.readline())
    flavor = re.search("_([a-z]*-[a-z]*)_", f).group(1)
    smearing = re.search("([0-2]_[0-2])", filename).group(1)

    size = re.search("_([0-9]*x[0-9]*x[0-9]*)_", filename).group(1)


    masses["heavy"] = options.heavyquarkmass


    heavyness = re.search("_([a-z][a-z0-9])_", filename).group(1)

    if heavyness != "ll" and options.heavyquarkmass is None:
        raise RuntimeError("heavy mass must be specified for heavy quarks")



    quarktype1, quarktype2 = re.search("_([a-z][a-z]*-[a-z][a-z]*)_", filename).group(1).split("-")
    logging.debug("quarktypes".format(quarktype1, quarktype2))

    if heavyness == "ll":
        quarkmass1 = masses[quarktype1]+residual_mass(dp)
        quarkmass2 = masses[quarktype2]+residual_mass(dp)
    else:
        quarkmass1 = masses[quarktype1]
        quarkmass2 = masses[quarktype2]

    if options.volume:
        volumefactor = dp.volume
    else:
        volumefactor = 1
    logging.info("dividing amplitudes by volume {}".format(volumefactor))

    renorm_2qm = (quarkmass1 + quarkmass2) / Zs[dp.beta]
    renorm_2qm = (quarkmass1 + quarkmass2)

    if options.function == "axial":
        decay_constant = np.sqrt( renorm_2qm*2*(df.amp/volumefactor) / (df.mass**2))
    if options.function == "axialsimul01-11":
        decay_constant = np.sqrt( 2*((df.amp1/volumefactor)**2 / (df.amp2/volumefactor)) /(df.mass) )
    if options.function == "axialsimul00-11":
        decay_constant = np.sqrt( renorm_2qm*2*(df.amp1/volumefactor) / (df.mass**2))
    if options.function == "simul01-11":
        decay_constant = renorm_2qm * np.sqrt(2*(((df.amp1)**2 / (df.amp2))/volumefactor) / (df.mass**3))
    if options.function == "simul00-11" or options.function == "simul00-01":
        decay_constant = renorm_2qm * np.sqrt(2*(df.amp1/volumefactor) / (df.mass**3))
    if options.function == "standard":
        decay_constant = renorm_2qm * np.sqrt(2*(df.amp/volumefactor) / (df.mass**3))


    if options.out_stub:
        # if heavyness != "ll":
        #     heavyness = heavyness+"_heavy{}".format(options.heavyquarkmass)
        f1, f2 = flavor.split("-")
        logging.debug("flavors {},{}".format(f1, f2))
        x = masses[f1]+residual_mass(dp)
        header = "#{}, {}, {}\n".format(x, decay_constant.mean(), decay_constant.std())
        logging.info(header)
        outfilename = "{}_{}_{}_{}_b{}_mud{}_ms{}_decayconstant_{}-{}.boot".format(options.out_stub, size, heavyness, smearing, beta, masses["ud"], strange_mass,
                                                                                  quarktype1, quarktype2)
        logging.info("Writing bootstraped decay constants to {}".format(outfilename))
        csv_txt = decay_constant.to_csv(None, sep=" ", index=False, header=False)
        with open(outfilename,"w") as outfile:
            outfile.write(header)
            outfile.write(csv_txt)
    else:
        print(decay_constant)
        print("{}, {}, {}\n".format(masses["ud"]+residual_mass(dp), decay_constant.mean(), decay_constant.std()))

if __name__ == "__main__":
    functs = ["axial", "standard", "simul00-01", "simul00-11", "simul01-11", "axialsimul", "axialsimul01-11", "axialsimul00-11"]
    parser = argparse.ArgumentParser(description="average data files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--out_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-f", "--function", type=str, required=False, choices=functs,
                        help="function to use to compute the decay constant")
    parser.add_argument("-m", "--heavyquarkmass", type=float, required=False,
                        help="The heavyquarkmass to use")
    parser.add_argument("-b", "--bothquarks", action="store_true",
                        help="use both quark masses as the x value")
    parser.add_argument("-V", "--volume", action="store_true",
                        help="divide amplitudes by the volume")
    args = parser.parse_args()

    if args.function is None:
        args.function = readinput.selectchoices(functs)

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Computing decay constants for: {}".format("\n".join(args.files)))

    outdir = os.path.dirname(args.out_stub)
    if not os.path.exists(outdir):
        logging.info("directory for output {} does not exist, atempting to create".format(outdir))
        if outdir is not "":
            os.makedirs(outdir)


    for f in args.files:
        if "2k-pi" in f:
            continue
        decay_constant(f, args)
