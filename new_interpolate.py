#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass, residual_mass_errors

from ensamble_info import data_params, read_fit_mass, scale, phys_pion, phys_kaon
from ensamble_info import Zs, Zv

from ensemble_data import ensemble_data

import inspect


def read_files(files, fitdata, cutoff=None):
    data = {}

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

        data[dp] = ed

    return data


class Model(object):

    def __init__(self, data, type_string):

        self.data = data
        self.type_string = type_string

        dps = self.data.keys()

        self.mpisqr = np.array([(data[dp].pion_mass(scaled=True).mean())**2 for dp in dps])
        self.mpisqr_std = np.array([(data[dp].pion_mass(scaled=True)**2).std() for dp in dps])
        self.mKsqrs = np.array([(data[dp].kaon_mass(scaled=True).mean())**2 for dp in dps])
        self.a = np.array([dp.latspacing for dp in dps])
        self.fpi = np.array([data[dp].fpi(scaled=True).mean() for dp in dps])
        self.fpi_var = np.array([data[dp].fpi(scaled=True).var() for dp in dps])
        self.xi = np.array([data[dp].xi(scaled=False).mean() for dp in dps])

        self.qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass) for dp in dps])
        self.renorm_qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass)/Zs[dp.beta] for
                                      dp in dps])
        self.res_err = np.array([data[dp].scale*residual_mass_errors(dp) for dp in dps])

    def build_function(self):

        LAMBDA4_GUESS = 1000.0
        LAMBDA3_GUESS = 600.0

        B_GUESS = 2661.69
        c3_GUESS = 4.0

        def paramdict(parameter, guess, err, limits=None, fix=False):

            paramdict = {parameter: guess}
            paramdict["error_"+parameter] = err
            paramdict["fix_"+parameter] = fix
            if limits:
                paramdict["limit_"+parameter] = limits
            return paramdict

        if self.type_string == "mpisqrbymq_const":
            params = paramdict("B", 2000.0, 100.0)
            fun = self.mpisqrbymq_const

        elif self.type_string == "mpisqrbymq_xi_NLO":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("c3", c3_GUESS, c3_GUESS/10))
            fun = self.mpisqrbymq_xi_NLO

        elif self.type_string == "mpisqrbymq_x_NLO":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("F_0", 118.038, 4.30, fix=True))
            fun = self.mpisqrbymq_x_NLO

        elif self.type_string == "FPI_x_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("B", 2826.1, 68.66, fix=True))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_x_NLO_only

        elif self.type_string == "FPI_XI_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("c4", LAMBDA4_GUESS, LAMBDA4_GUESS/10))
            fun = self.FPI_XI_NLO_only

        elif self.type_string == "FPI_XI_NLO_inverse_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_XI_NLO_inverse_only

        else:
            logging.error("Function not supported yet")
            raise RuntimeError("Function {} not supported yet".format(self.type_string))

        return params, fun

    def mpisqrbymq_const(self, B):

        mpierr = self.mpisqr_std
        data = self.mpisqr / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        M = 2*B
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NLO(self, B, c3):
        mpierr = self.mpisqr_std
        data = self.mpisqr / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        M = 2*B*(1.0+0.5*self.xi*np.log(self.xi) + 0.5*c3*self.xi)
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def mpisqrbymq_x_NLO(self, B, F_0, Lambda3):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))

        mpierr = self.mpisqr_std
        data = self.mpisqr / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        arg1 = (Lambda3**2)/Msqr
        M = 2*B*(1.0-0.5*x*np.log(arg1))
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def FPI_x_NLO_only(self, F_0, B, Lambda4):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)

    def FPI_x_NNLO_only(self, F_0, B, Lambda4, k_f, LambdaF):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(4*np.pi*F_0)**2
        arg1 = (Lambda4**2)/Msqr
        arg2 = (LambdaF**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1) - (5.0/4.0)*(x**2)*(np.log(arg2))**2 + k_f*x**2)
        sqr_diff = (self.fpi/np.sqrt(2) - M)**2
        return np.sum(sqr_diff/self.fpi_var)

    def FPI_XI_NLO_only(self, F_0, c4):
        M = F_0 * (1 - self.xi*np.log(self.xi) + c4*self.xi)
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)

    def FPI_XI_NLO_inverse_only(self, F_0, Lambda4):
        arg = self.mpisqr/(Lambda4**2)
        M = F_0 / (1 + self.xi*np.log(arg))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)

    def FPI_XI_NNLO_inverse_only(self, F_0, Lambda4, Omega_F, cF):
        arg1 = self.mpisqr/(Lambda4**2)
        arg2 = self.mpisqr/(Omega_F**2)
        XIs = self.xi
        M = F_0 / (1 + XIs*np.log(arg1) - (1.0/4.0)*(XIs*np.log(arg2))**2 - cF*(XIs**2))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)


def interpolate(data, model_str):

    logging.info("Fitting data")

    params, model = Model(data, model_str).build_function()

    ARGS = inspect.getargspec(model).args[1:]
    Nfree_params = len(ARGS)
    dof = float(len(data)-Nfree_params)

    logging.info("DOF {}".format(dof))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    m = Minuit(model, errordef=dof, print_level=0, pedantic=True, **params)

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

    alldata = read_files(options.files, options.fitdata, cutoff=options.cutoff)

    fit_paramsters = interpolate(alldata, options.model)

    write_data(fit_paramsters, options.output_stub, ".fit", options.model)


if __name__ == "__main__":

    models = ["chiral_NLO_only", "chiral_NNLO_only", "chiral_NLO_all", "chiral_NNLO_all", "s_a_pi",
              "MPI_XI_NLO_only", "FPI_x_NLO_only", "FPI_XI_NLO_only", "FPI_XI_NNLO_only",
              "FPI_XI_NLO_inverse_only", "FPI_XI_NNLO_inverse_only", "mpisqrbymq_const",
              "mpisqrbymq_xi_NLO", "mpisqrbymq_x_NLO"]

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
    parser.add_argument("-m", "--model", required=False, type=str, choices=models, default="s_a_pi",
                        help="which model to use")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    interpolate_chiral_spacing(args)
