#!/usr/bin/env python2
import logging
import argparse
import numpy as np
import re

def check_done(lines):
    ends = [l == "" for l in lines]
    if any(ends):
        if all(ends):
            return True
        else:
            raise ValueError("Files are not the same length")
    return False

def allEqual(lst):
     return not lst or lst.count(lst[0]) == len(lst)



def check_control_lines(lines, data_info):
    infoline = [l.startswith((":","-","=")) for l in lines]
    if any(infoline):
        lines = [l.replace("heaby", "heavy") for l in lines]
        if not allEqual(lines):
            raise ValueError("Lines don't contain the same correlators")

        fline = lines[0]        # Only need one of the lines if they are all equal

        if fline.startswith(":"):
            description = fline.strip("\n :")
            data_info["flavor"] = description.split()[0]
            try:
                snk,src = description.split()[-1].strip("()").split(",")
                data_info["snk"] = snk
                data_info["src"] = src
            except ValueError:
                logging.info("heavy-heavy has only one sink,src")
                data_info["snk"] = 0
                data_info["src"] = 0
            logging.info("found {} {} {}".format(data_info["flavor"], data_info["snk"], data_info["src"]))
            return True
        elif fline.startswith("="):
            logging.info("line just says the hadron type, skipping")
            return True
        elif fline.startswith("-"):
            correlatortype = fline.strip("\n -")
            data_info["correlatortype"] = correlatortype
            logging.info("correlator type {}".format(correlatortype))
            return True
    return False

class filewriter:
    def __init__(self,data_info, shift):
        self.flavor = data_info["flavor"]
        self.snk = data_info["snk"]
        self.src = data_info["src"]
        self.correlatortype = data_info["correlatortype"]
        self.shift = shift
        self.data = {}
        self.shifted_data = {}
        self.averaged_data = {}

    def add_data(self,line,fromfile):

        if self.shift == "auto":
            shift = args.stride*int(re.search("src([0-9]+)", fromfile).group(1))
        else:
            shift = self.shift

        if args.offset:
            if args.offset_condition in fromfile:
                shift += args.offset

        if (fromfile,shift) not in self.data:
            self.data[(fromfile,shift)] = []

        time = int(line.strip().split()[0])
        data = ", ".join(line.strip().split()[1:])

        self.data[(fromfile,shift)].append((time, data) )

    def shift_data(self):
        logging.info("Shifting data")

        def index(i, period, shift):
            new = i+shift
            if new >= period:
                return new - period
            return new


        for key, data in self.data.iteritems():
            fromfile, shift = key
            ddata = dict(data)
            period = max(ddata.keys())+1
            self.shifted_data[fromfile] = [ (i,ddata[index(i, period, shift)]) for i in range(max(ddata.keys())+1) ]

    def average_sources(self):
        logging.info("Averaging sources")
        for t in range(20):
            sourceset = [i for i in self.shifted_data if i[-4:] == "src{}".format(t)]
            if len(sourceset) > 0:
                break
        config_groups = list(set(tuple(i for i in self.shifted_data if i[:-4] == z[:-4]) for z in sourceset))
        #assert allEqual([len(i) for i in config_groups])

        numberofsources = len(config_groups[0])
        logging.info("found {} configs with {} times each".format(len(config_groups), numberofsources))
        times = len(self.shifted_data[self.shifted_data.keys()[0]])
        for cg in config_groups:
            cgdata = {i: 0.0 for i in range(0,times)}
            for c in cg:
                for i in self.shifted_data[c]:
                    t, d = i
                    cgdata[t] += float(d)
            for k, v in cgdata.iteritems():
                cgdata[k] = v / len(cg)
            self.averaged_data[cg[0]] = cgdata



    def write(self):
        self.shift_data()
        self.average_sources()
        ofile_name = "{}_{}_{}-{}_{}".format(args.output_stub, self.flavor, self.snk, self.src, self.correlatortype)
        logging.info("opening file {}".format(ofile_name))
        ofile = open(ofile_name, "w")
        logging.info("writing data")

        for key,data in self.averaged_data.iteritems():
            for i in range(max(data)+1):
                ofile.write("{0}, {1:.16e}\n".format(i, data[i]))

def split_data(args):

    ofile = None

    linecounts = [sum(1 for line in open(fname)) for fname in args.files]

    target = max(linecounts)
    prunedfiles = [f for c,f in zip(linecounts, args.files) if c == target]

    for f in args.files:
        if f not in prunedfiles:
            logging.error("incomplete files {}".format(f))

    infiles = [open(fname) for fname in prunedfiles]

    data_info = {"flavor":None, "snk":None, "src":None, "correlatortype":None}
    fw = filewriter(data_info, args.shift)
    while True:
        lines = [f.readline() for f in infiles]
        logging.debug(lines[0])
        if check_done(lines):
            break
        if check_control_lines(lines, data_info):
            logging.info("control line found info updated")
            logging.debug("{}".format(repr(data_info)))
            if fw.data:
                fw.write()
            fw = filewriter(data_info, args.shift)
        else:
            # logging.debug("is a data line")
            for line,f in zip(lines,prunedfiles):
                fw.add_data(line,f)

    fw.write()
    logging.info("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse iroiro correlator files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output-stub", type=str, default="out", required=False,
                        help="stub of name to write output to")
    parser.add_argument("-s", "--shift", type=int, default=0, required=False,
                        help="shift the times on the data")
    parser.add_argument("--stride", type=int, required=False,
                        help="each source has an offset of STRIDE")
    parser.add_argument("--offset", type=int, required=False,
                        help="offset of the sources")
    parser.add_argument("--offset_condition", type=str, required=False,
                        help="condition to apply the offset")
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

    if args.stride:
        args.shift = "auto"

    split_data(args)
