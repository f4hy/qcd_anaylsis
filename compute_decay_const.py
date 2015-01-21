#!/usr/bin/env python2
import argparse
import logging
import numpy as np
import pandas as pd
import re
import readinput

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
    masses = {}
    masses["ud"] = float(re.search("mud([0-9]\.[0-9]*)_", filename).group(1))
    masses["s"] = float(re.search("ms([0-9]\.[0-9]*)", filename).group(1))
    print filename
    smearing = re.search("([0-2]_[0-2])", filename).group(1)
    print(masses)

    beta = re.search("_b(4\.[0-9]*)_", filename).group(1)


    if beta == "4.35":
        heavy_masses = {"m0": 0.12, "m1": 0.24, "m2": 0.36}
    if beta == "4.17":
        heavy_masses = {"m0": 0.2, "m1": 0.4, "m2": 0.6}
        #heavy_masses = {"m0": 0.35, "m1": 0.445, "m2": 0.52}

    heavyness = re.search("_([a-z][a-z0-9])_", filename).group(1)

    resisdual_masses = {(0.019, 0.030): 0.00047, (0.012, 0.030): 0.00040, (0.007, 0.030): 0.00042,
                        (0.019, 0.040): 0.00041, (0.012, 0.040): 0.00044, (0.007, 0.040): 0.00038,
                        (0.0035, 0.040): 0.00039,
                        (0.008, 0.018): 0.00000, (0.008, 0.025): 0.00000,
                        (0.012, 0.018): 0.000053, (0.012, 0.025): 0.000032,
                        (0.0042, 0.018): 0.00000, (0.0042, 0.025): 0.00000 }

    print masses["s"]
    print masses["ud"]
    print resisdual_masses[(masses["ud"], masses["s"])]

    if heavyness == "ll":
        quarktype1, quarktype2 = re.search("_([a-z][a-z]?-[a-z][a-z]??)_", filename).group(1).split("-")
        print quarktype1, quarktype2
        quarkmass1 = masses[quarktype1]+resisdual_masses[(masses["ud"], masses["s"])]
        quarkmass2 = masses[quarktype2]+resisdual_masses[(masses["ud"], masses["s"])]
    else:
        quarktype1, quarktype2 = re.search("_([a-z][a-z]*-[a-z][a-z]??)_", filename).group(1).split("-")
        print quarktype1, quarktype2
        quarkmass1 = heavy_masses[heavyness]
        quarkmass2 = masses[quarktype2]

    print quarkmass1, quarkmass2

    if options.function == "axial":
        decay_constant = np.sqrt(2*df.amp)
    if options.function == "simul":
        decay_constant = (quarkmass1 + quarkmass2) * np.sqrt(2*(df.amp1**2 / df.amp2) / (df.mass**3))
    else:
        decay_constant = (quarkmass1 + quarkmass2) * np.sqrt(2*df.amp / (df.mass**3))

    if options.out_stub:
        outfilename = "{}_mud{}_ms{}_decayconstant_{}-{}.boot".format(options.out_stub, masses["ud"], masses["s"],
                                                                      quarktype1, quarktype2)
        logging.info("Writing bootstraped decay constants to {}".format(outfilename))
        decay_constant.to_csv(outfilename, sep=" ", index=False, header=False)
        logging.info("Writing average decay constants to {}".format(outfilename))
        outfilename = "{}_{}_{}_b{}_mud{}_ms{}_decayconstant_{}-{}.out".format(options.out_stub, heavyness, smearing, beta, masses["ud"], masses["s"],
                                                                     quarktype1, quarktype2)
        with open(outfilename, 'w') as outfile:
            outfile.write("{}, {}, {}\n".format(masses["ud"]+resisdual_masses[(masses["ud"], masses["s"])], decay_constant.mean(), decay_constant.std()))
    else:
        print(decay_constant)
        print("{}, {}, {}\n".format(masses["ud"]+resisdual_masses[(masses["ud"], masses["s"])], decay_constant.mean(), decay_constant.std()))

if __name__ == "__main__":
    functs = ["axial", "standard", "simul"]
    parser = argparse.ArgumentParser(description="average data files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--out_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-f", "--function", type=str, required=False, choices=functs,
                        help="function to use to compute the decay constant")
    args = parser.parse_args()

    if args.function is None:
        args.function = readinput.selectchoices(functs)

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Computing decay constants for: {}".format("\n".join(args.files)))

    for f in args.files:
        decay_constant(f, args)
