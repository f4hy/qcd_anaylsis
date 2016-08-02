#!/usr/bin/env python2
import logging
import argparse
import numpy as np
from collections import defaultdict

from alpha_s import get_Cmu_mbar

from ensamble_info import phys_D, phys_Ds
from ensamble_info import phys_MB, phys_MBs
from ensamble_info import phys_mc_2gev, phys_mb_2gev

mass = {"D": phys_D, "B": phys_MB, "Ds": phys_Ds, "Bs": phys_MBs}
mq = {"D": phys_mc_2gev, "B": phys_mb_2gev, "Ds": phys_mc_2gev, "Bs": phys_mb_2gev}
Cmu_c = get_Cmu_mbar(phys_mc_2gev)
Cmu_b = get_Cmu_mbar(phys_mb_2gev)
Cmu = {"D": Cmu_c, "B": Cmu_b, "Ds": Cmu_c, "Bs": Cmu_b}

def main(options):
    logging.info("Extrapolating with {}".format(options.file))

    functionmap = {
        "fdsqrtm_chiral_dmss": extrpolate_fdsqrtm,
        "fdsqrtm_chiral_dmss_HQET": extrpolate_fdsqrtm,
        "fdsqrtm_HQET_matched_nom2": extrpolate_fdsqrtm_matched,
        "fdssqrtms_chiral_dmss_HQET": extrpolate_fdssqrtms,
        "fdsqrtm_HQET_matched": extrpolate_fdsqrtm_matched,
        "fdsqrtm_HQET_matched_alphas": extrpolate_fdsqrtm_matched,
        "fdAsqrtm_HQET_matched_alphas": extrpolate_fdsqrtm_matched,
        "fdssqrtms_chiral_dmss": extrpolate_fdssqrtms,
        "fdssqrtms_HQET_matched_nom2": extrpolate_fdssqrtms_matched,
        "fdssqrtms_HQET_matched_nom2_alphas": extrpolate_fdssqrtms_matched,
        "fdssqrtms_HQET_matched_alphas": extrpolate_fdssqrtms_matched,
        "fdsAsqrtms_HQET_matched_alphas": extrpolate_fdssqrtms_matched,
        "fdssqrtms_HQET_matched": extrpolate_fdssqrtms_matched
    }

    with open(options.file) as f:

        comment = f.readline().strip("#\n ")
        comments = comment.split(",")
        model, names = comments[0], comments[1:]
        results = defaultdict(list)
        for b, strap in enumerate(f):
            values = dict(zip(names, map(float, strap.split(","))))

            r = functionmap[model](values, options.extrapolation_point)
            logging.debug("bstrap{}:{}".format(b, r))
            for k, v in r.iteritems():
                results[k].append(v)


        if options.output_stub:
            outfilename = options.output_stub + ".out"
            ofile = open(outfilename, "w")
            ofile.write("#extrapolating {} to {}\n".format(model, options.extrapolation_point))
        for k, r in results.iteritems():
            logging.info(k)
            logging.info("average: {}, median {}, std {}".format(np.mean(r), np.median(r),
                                                                 np.std(r)))
            logging.info("answer {} +/- {}".format(np.mean(r), np.std(r)))
            if options.output_stub:
                ofile.write("{} average: {}, median {}, std {}\n".format(k, np.mean(r), np.median(r), np.std(r)))
                ofile.write("{}{}: {} +/- {}\n".format(k,options.extrapolation_point, np.mean(r), np.std(r)))


def extrpolate_fdsqrtm(values, point):
    logging.debug("Input M_B at {}".format(point))
    logging.debug("Extrapolting to {} using {}".format(point, values))
    masspoint = mass[point]
    C1 = values["C1"]
    C2 = values["C2"]
    Fsqrtm_inf = values["Fsqrtm_inf"]
    inv_md = 1.0 / mass[point]
    logging.debug("inv Md {}".format(inv_md))
    Fsqrtm = (Fsqrtm_inf * (1.0 + C1 * inv_md + C2 * (inv_md**2.0)))
    f = Fsqrtm / np.sqrt(mass[point])
    return {"Fsqrtm": Fsqrtm, "f": f}

def extrpolate_fdsqrtm_matched(values, point):
    logging.debug("Input M_B at {}".format(point))
    logging.debug("Extrapolting to {} using {}".format(point, values))

    C1 = values["C1"]
    C2 = values["C2"]
    Fsqrtm_inf = values["Fsqrtm_inf"]
    inv_md = 1.0 / mass[point]
    logging.debug("inv Md {}".format(inv_md))
    Fsqrtm = (Fsqrtm_inf * (1.0 + C1 * inv_md + C2 * (inv_md**2.0))) * Cmu[point]
    logging.info("C({})={}".format(mq[point], Cmu[point]))
    f = Fsqrtm / np.sqrt(mass[point])
    return {"Fsqrtm": Fsqrtm, "f": f}



def extrpolate_fdssqrtms(values, point):
    logging.debug("Input M_B as {}".format(point))
    logging.debug("Extrapolting to {} using {}".format(point, values))

    C1 = values["C1"]
    C2 = values["C2"]
    Fssqrtms_inf = values["Fssqrtms_inf"]
    inv_mds = 1.0 / mass[point]
    logging.debug("inv Mds {}".format(inv_mds))
    Fssqrtms = (Fssqrtms_inf * (1.0 + C1 * inv_mds + C2 * (inv_mds**2.0))) # / np.sqrt(point)
    f = Fssqrtms / np.sqrt(mass[point])
    return {"Fssqrtms": Fssqrtms, "f": f}

def extrpolate_fdssqrtms_matched(values, point):
    logging.debug("Input M_B as {}".format(point))
    logging.debug("Extrapolting to {} using {}".format(point, values))

    C1 = values["C1"]
    C2 = values["C2"]
    Fssqrtms_inf = values["Fssqrtms_inf"]
    inv_mds = 1.0 / mass[point]
    logging.debug("inv Mds {}".format(inv_mds))
    Fssqrtms = (Fssqrtms_inf * (1.0 + C1 * inv_mds + C2 * (inv_mds**2.0))) * Cmu[point]
    #logging.info("C({})={}".format(point, get_Cmu_mbar(point)))
    f = Fssqrtms / np.sqrt(mass[point])
    return {"Fssqrtms": Fssqrtms, "f": f}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute alpha_s at 1-4 loops")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('file', metavar='f', type=str,
                        help='fit file to use')
    parser.add_argument('-p', '--extrapolation_point', required=True, type=str,
                        help='point to extrapolate to')
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(args)
