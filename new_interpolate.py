#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
from iminuit import Minuit

from ensamble_info import data_params, read_fit_mass, scale, phys_pion, phys_kaon, phys_Fpi
from ensamble_info import Zs, Zv
from ensamble_info import phys_pionplus

from ensemble_data import ensemble_data, MissingData

import inspect
import collections

from global_fit_model import Model


def read_files(files, fitdata, cutoff=None, hqm_cutoff=None):
    data = collections.OrderedDict()

    for f in files:
        logging.info("reading file {}".format(f))
        if "32x64x12" in f and "0.0035" in f:
            logging.warn("skipping file {}".format(f))
            continue

        dp = data_params(f)

        ed = ensemble_data(dp)

        if cutoff:
            if np.mean(ed.pion_mass(scaled=True).mean()) > (cutoff):
                continue
        if hqm_cutoff:
            if dp.heavyq_mass > hqm_cutoff:
                logging.info("dp {} has hqm {} > {}".format(dp, dp.heavyq_mass, hqm_cutoff))
                continue

        data[dp] = ed

    return data


def interpolate(data, model_str, options):

    logging.info("Fitting data")

    params, model = Model(data, model_str, options).build_function()

    ARGS = inspect.getargspec(model).args[1:]
    logging.info("Params {}".format(params))
    fixed_parms = [p for p in params if "fix" in p and params[p]]
    Nfree_params = len(ARGS) - len(fixed_parms)
    if model_str.startswith("combined"):
        dof = float(len(data)*2-Nfree_params)
    else:
        dof = float(len(data)-Nfree_params)
    # if "all" in model_str:
    #     dof = dof + 4

    logging.info("DOF {}, data {}, free {}".format(dof, len(data), Nfree_params))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    m = Minuit(model, errordef=dof, print_level=0, pedantic=True, **params)
    m.set_strategy(2)
    results = m.migrad()

    logging.debug(results)

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(m.fval, dof, m.fval/dof))
    logging.info('covariance {}'.format(m.covariance))
    logging.info('fitted values {}'.format(m.values))
    logging.info('fitted errors {}'.format(m.errors))

    if not m.get_fmin().is_valid:
        logging.error("NOT VALID")
        exit(-1)

    return m


def write_data(fit_parameters, output_stub, suffix, model):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        chisqrbydof = fit_parameters.fval / fit_parameters.errordef
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fit_parameters.fval,
                                                                    fit_parameters.errordef,
                                                                    chisqrbydof))

        for name in fit_parameters.values.keys():
            ofile.write("{}, {} +/- {}\n".format(name, fit_parameters.values[name],
                                                 fit_parameters.errors[name]))


def interpolate_chiral_spacing(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    alldata = read_files(options.files, options.fitdata, cutoff=options.cutoff, hqm_cutoff=options.hqm_cutoff)

    fit_paramsters = interpolate(alldata, options.model, options)

    write_data(fit_paramsters, options.output_stub, ".fit", options.model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--pdf", action="store_true",
                        help="produce a pdf instead of a png")
    parser.add_argument("--seperate_strange", action="store_true",
                        help="fit different strange values seperately")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("--cutoff", required=False, type=float,
                        help="cutoff value")
    parser.add_argument("--hqm_cutoff", required=False, type=float,
                        help="cutoff value")
    parser.add_argument("-m", "--model", required=False, type=str, default="s_a_pi",
                        help="which model to use")
    parser.add_argument("-z", "--zero", required=False, type=str, nargs='+', default=[],
                        help="Zero a fit variable")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    interpolate_chiral_spacing(args)
