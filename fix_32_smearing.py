#!/usr/bin/env python2
import logging
import argparse
import numpy as np
import re


def allEqual(lst):
     return not lst or lst.count(lst[0]) == len(lst)


def process_data(readfile, args):

    smearings = set()
    for line in readfile:
        if "sink" in line:
            for i in ["0","1","2"]:
                if i in line:
                    smearings.update(i)

    if len(smearings) < 3:
        print "does not have 3 smearings, not processing!"
        exit(-1)

    print "has 3 smearings, processing!"

    readfile.seek(0)
    lines = []
    read = True

    for line in readfile:
        if "sink" in line:
            if "1" in line:
                read = False
            else:
                read = True
            if "2" in line:
                line = line.replace("2", "1")
        if read:
            lines.append(line)
        else:
            logging.debug("ignoring line")

    if args.outfile:
        logging.info("writing to {}".format(args.outfile))
        with open(args.outfile, 'w') as ofile:
            ofile.write("".join(lines))
    else:
        logging.info("no output file")

    logging.info("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse iroiro correlator files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--outfile", type=str, default="out", required=False,
                        help="stub of name to write output to")
    parser.add_argument('--err', nargs='?', type=argparse.FileType('w'),
                        default=None)
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')

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


    for f in args.files:
        with open(f) as readfile:
            process_data(readfile, args)
