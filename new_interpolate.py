#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass

from ensamble_info import data_params, read_fit_mass, scale, phys_pion, phys_kaon
from ensamble_info import Zs, Zv

from ensemble_data import ensemble_data

import inspect

colors = ['b', 'k', 'c', 'y', 'm', 'b']

# hbar*c / 1 fm  = 197.327 MeV
hbar_c = 197.327


def read_files(files, fitdata, cutoff=None):
    data = {}

    xtype = "mpisqr"

    for f in files:
        logging.info("reading file {}".format(f))
        if "32x64x12" in f and "0.0035" in f:
            logging.warn("skipping file {}".format(f))
            continue


        dp = data_params(f)

        ed = ensemble_data(dp)
        print ed.pion_mass().mean()
        print ed.kaon_mass().mean()
        print ed.xi()

        if cutoff:
            if np.mean(ed.pion_mass(scaled=True).mean()) > (cutoff):
                continue


        data[dp] = ed

    return data


class Model(object):

    def __init__(self, data, type_string):

        self.data = data
        self.type_string = type_string

        self.mpisqr = np.array([(d.pion_mass(scaled=True).mean())**2 for dp, d in self.data.iteritems()  ])
        self.mKsqrs = np.array([(d.kaon_mass(scaled=True).mean())**2 for dp, d in self.data.iteritems()  ])
        self.a = np.array([dp.latspacing for dp, d in self.data.iteritems()  ])
        self.fpi = np.array([d.fpi(scaled=True).mean() for dp, d in self.data.iteritems()  ])
        self.fpi_var = np.array([d.fpi(scaled=True).var() for dp, d in self.data.iteritems()  ])
        self.xi = np.array([d.xi(scaled=False).mean() for dp, d in self.data.iteritems()  ])

        self.qmass = np.array([d.scale*(residual_mass(dp)+dp.ud_mass) for dp, d in self.data.iteritems()  ])
        self.renorm_qmass = np.array([d.scale*(residual_mass(dp)+dp.ud_mass)/Zs[dp.beta] for dp, d in self.data.iteritems()  ])


        print self.xi
        print self.fpi


    def build_function(self):

        LAMBDA4_GUESS = 1000.0
        LAMBDA3_GUESS = 600.0

        def paramdict(parameter, guess, err, limits=None, fix=False):

            paramdict = {paremeter: guess}
            paramdict["error_"+parameter] = err
            paramdict["fix_"+parameter] = fix
            if limits:
                paramdict["limit_"+parameter] = limits


            return paramdict



        if self.type_string == "FPI_x_NLO_only":
            guess = [np.mean(self.fpi), 1950.0, LAMBDA4_GUESS]
            guess_errs = [np.mean(self.fpi_var)/10.0, 50.0, LAMBDA4_GUESS/100]
            fun = self.FPI_x_NLO_only

        elif self.type_string == "FPI_XI_NLO_only":
            guess = [np.mean(self.fpi)]
            guess_errs = [np.mean(self.fpi_var)/10.0]
            fun = self.FPI_XI_NLO_only

        elif self.type_string == "FPI_XI_NNLO_only":
            guess = [np.mean(self.fpi)]
            guess_errs = [np.mean(self.fpi_var)/10.0]
            fun = self.FPI_XI_NNLO_only


        elif self.type_string == "FPI_XI_NLO_inverse_only":
            guess = [np.mean(self.fpi), LAMBDA4_GUESS]
            guess_errs = [np.mean(self.fpi_var)/10.0, LAMBDA4_GUESS/100]
            fun = self.FPI_XI_NLO_inverse_only

        elif self.type_string == "FPI_XI_NNLO_inverse_only":
            guess = [np.mean(self.fpi), LAMBDA4_GUESS, 1000.0, 1.0]
            guess_errs = [np.mean(self.fpi_var)/10.0, LAMBDA4_GUESS/1000, 0.1, 0.1]
            fun = self.FPI_XI_NNLO_inverse_only

        elif self.type_string == "FPI_XI_NNLO_inverse_only":
            guess = [np.mean(self.fpi), LAMBDA4_GUESS, 1000.0, 1.0]
            guess_errs = [np.mean(self.fpi_var)/10.0, LAMBDA4_GUESS/1000, 0.1, 0.1]
            fun = self.FPI_XI_NNLO_inverse_only


        else:
            logging.error("Function not supported yet")
            raise RuntimeError("Function {} not supported yet".format(self.type_string))

        print self.type_string
        ARGS = inspect.getargspec(fun).args[1:]
        params = dict(zip(ARGS,guess))
        params.update(dict(zip(["error_"+a for a in ARGS],guess_errs)))

        # params["fix_B"] = True

        return params, fun


    def FPI_x_NLO_only(self, F_0, B, Lambda4):
        Msqr = B*(self.qmass+self.qmass)
        x = Msqr/(4*np.pi*F_0)**2
        arg1 = (Lambda4**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1))
        sqr_diff = (self.fpi/np.sqrt(2) - M)**2
        return np.sum(sqr_diff/self.fpi_var)


    def FPI_x_NNLO_only(self, F_0, B, Lambda4, k_f, LambdaF):
        Msqr = B*(self.qmass+self.qmass)
        x = Msqr/(4*np.pi*F_0)**2
        arg1 = (Lambda4**2)/Msqr
        arg2 = (LambdaF**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1)- (5.0/4.0)*(x**2)*(np.log(arg2))**2 + k_f*x**2  )
        sqr_diff = (self.fpi/np.sqrt(2) - M)**2
        return np.sum(sqr_diff/self.fpi_var)



    def FPI_XI_NLO_only(self, F_0):
        M = F_0 * (1 - self.xi*np.log(self.xi))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)

    def FPI_XI_NNLO_only(self, F_0):
        XIs = self.xi
        M = F_0 * (1 - XIs*np.log(XIs)+(5.0/4.0)*(XIs*np.log(XIs))**2)
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi_var)


    def FPI_XI_NLO_inverse_only(self, F_0, Lambda4):
        arg = self.mpisqr/(Lambda4**2)
        M = F_0 / (1 + self.xi*np.log(arg))
        sqr_diff = (self.fpi - M)**2
        print sqr_diff
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

    dps = []
    a = []
    mpisqrs = []
    mKsqrs = []
    observables = []
    variances = []
    XIs = []

    for dp, d in data.iteritems():
        dps.append(dp)
        a.append(dp.latspacing)
        mpisqrs.append((d.pion_mass(scaled=True).mean())**2)
        mKsqrs.append((d.kaon_mass(scaled=True).mean())**2)
        # a.append(d[0])
        # mpisqrs.append(np.mean(d[1]))
        #mKsqrs.append(np.mean(d[2]))
        #observables.append(np.mean(d[3]))
        # variances.append(np.var(d[3]))
        # XIs.append(np.mean(d[4]))

    print dps
    a = np.array(a)
    mpisqrs = np.array(mpisqrs)
    print mpisqrs

    # M = Model(data, model_str)
    # print M.mpisqr
    mKsqrs = np.array(mKsqrs)
    observables = np.array(observables)
    variances = np.array(variances)
    XIs = np.array(XIs)

    guess_F_0 = np.mean(observables)
    guess_phys_obs = np.mean(observables)
    guess_A = 0.0
    guess_M_pi = 1.0
    guess_chiral_log = 0.0
    guess_M_K = 1.0

    mqs = np.array([(scale[p.beta]*(p.ud_mass+residual_mass(p)) * 1.0/Zs[p.beta]) for p in dps])


    phys_mpisqr = (phys_pion)**2
    phys_mKsqr = (phys_kaon)**2


    params, model = Model(data, model_str).build_function()


    Nfree_params = len(params)/2.0
    dof = float(len(data)-Nfree_params)

    logging.info("DOF {}".format(dof))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    m = Minuit(model, errordef=dof, print_level=0, pedantic=True, **params)

    results = m.migrad()

    logging.debug(results)
    logging.debug("variances {}".format(variances))
    logging.debug("std {}".format(np.sqrt(variances)))

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(m.fval, dof, m.fval/dof))
    logging.info('covariance {}'.format(m.covariance))
    logging.info('fitted values {}'.format(m.values))
    logging.info('fitted errors {}'.format(m.errors))


    if not m.get_fmin().is_valid:
        print "NOT VALID"
        exit(-1)

    XI_phys = phys_mpisqr/(8*np.pi**2*m.values["F_0"]**2)


    return m


legend_handles = []


def plot_fitline(data, fit_params, ftype, phys_x, outstub):
    logging.info("ploting")

    c = colors.pop()
    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                        capthick=2, mew=3, aa=True, fmt='o')

    xvalues = []


    for dp, d in data.iteritems():
        xvalue, flow = d
        N = len(flow)
        y = np.mean(flow)
        err = np.sqrt((N-1)*(np.std(flow)**2))
        logging.info("{}, {}={}, err={}".format(dp, ftype, y, err))
        patch = plt.errorbar(xvalue, y, yerr=err, color=c, ecolor=c, mec=c, label="ms={}".format(dp.s_mass), **plotsettings)
        xvalues.append(xvalue)

    legend_handles.append(patch)

    xdata = np.arange(phys_x-0.01, max(xvalues)+0.005, 0.001)
    mydata = fit_params.values["phys_obs"]*(1+fit_params.values["C"]*xdata)

    t0_ch = fit_params.values["phys_obs"]*(1+fit_params.values["C"]*phys_x)

    plt.plot(xdata, mydata, color=c)

    m = fit_params
    t1 = (m.errors["A"]*(1+m.values["C"]*xdata))**2
    t2 = ((m.values["A"])*xdata*m.errors["C"])**2
    t3 = 2*xdata*mydata*m.covariance[("A", "C")]
    perry = t1+t2+t3

    plt.fill_between(xdata, mydata, mydata+np.sqrt(perry), facecolor=c, alpha=0.1, lw=0, zorder=-10)
    plt.fill_between(xdata, mydata, mydata-np.sqrt(perry), facecolor=c, alpha=0.1, lw=0, zorder=-10)

    t0_ch_variance = (m.errors["A"]*(1+m.values["C"]*phys_x))**2
    t0_ch_variance += ((m.values["A"])*phys_x*m.errors["C"])**2
    t0_ch_variance += 2*phys_x*t0_ch*m.covariance[("A", "C")]

    plt.errorbar(phys_x, t0_ch, yerr=np.sqrt(t0_ch_variance),
                 color="r", mec="r", **plotsettings)

    t0_ch_std = np.sqrt(t0_ch_variance)

    logging.info("Determined at physical point t_0 = {:.5f} +/- {:.5f}".format(t0_ch, t0_ch_std))

    return t0_ch, t0_ch_std


def finish_plot(beta, ftype, xlabel, legend_handles, outstub, pdf=False):
    fontsettings = dict(fontsize=20)

    plt.title(r'$\beta={}$    ${}$'.format(beta, ftype), **fontsettings)
    plt.ylabel("${} / a$".format(ftype), **fontsettings)
    plt.xlabel(xlabel, **fontsettings)

    plt.legend(handles=sorted(legend_handles), loc=0, **fontsettings)

    fileformat = ".pdf" if pdf else ".png"

    if outstub is not None:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        filename = outstub+fileformat
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_data(fit_parameters, output_stub, suffix, model):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fit_parameters.fval, fit_parameters.errordef, fit_parameters.fval/fit_parameters.errordef))

        for name in fit_parameters.values.keys():
            ofile.write("{}, {} +/- {}\n".format(name, fit_parameters.values[name], fit_parameters.errors[name]))


def interpolate_chiral_spacing(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    alldata = read_files(options.files, options.fitdata, cutoff=options.cutoff)

    fit_paramsters = interpolate(alldata, options.model)

    write_data(fit_paramsters, options.output_stub, ".fit", options.model)


if __name__ == "__main__":

    axis_choices = ["mud", "mud_s", "mpi", "tmpisqr", "t_2mksqr-mpisqr"]

    models = ["chiral_NLO_only", "chiral_NNLO_only", "chiral_NLO_all", "chiral_NNLO_all", "s_a_pi", "MPI_XI_NLO_only", "FPI_x_NLO_only", "FPI_XI_NLO_only", "FPI_XI_NNLO_only", "FPI_XI_NLO_inverse_only", "FPI_XI_NNLO_inverse_only", "mpisqrbymq"]

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
