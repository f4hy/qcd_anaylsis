#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass

from data_params import data_params, read_fit_mass, scale, phys_pion, phys_kaon
from data_params import Zs, Zv

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


        if "decay" in f:
            df = pd.read_csv(f, comment='#', names=["observable"])
        else:
            df = pd.read_csv(f, comment='#', names=["index", "observable", "", ""], index_col=0)

        mpisqr = xvalues(xtype, dp, fitdata)
        mKsqr = xvalues("mKsqr", dp, fitdata)
        XI = xvalues("xi", dp, './SymDW_sHtTanh_b2.0_smr3_*/xi/xi.out')

        if cutoff:
            if np.mean(mpisqr)*scale[dp.beta]**2 > (cutoff**2):
                continue

        logging.info("read {} giving mpi={}, <observable>={}".format(dp, np.mean(mpisqr), df.observable.mean()))
        #data[dp] = (scale[dp.beta], np.mean(xvalue), df.observable.values)
        hbar_c = 197.327

        if "ratio" in f:
            data[dp] = (hbar_c/scale[dp.beta], np.mean(mpisqr)*scale[dp.beta]**2, np.mean(mKsqr)*scale[dp.beta]**2, df.observable.values, XI)
        else:
            data[dp] = (hbar_c/scale[dp.beta], np.mean(mpisqr)*scale[dp.beta]**2, np.mean(mKsqr)*scale[dp.beta]**2, df.observable.values*scale[dp.beta], XI)

    return data


def xvalues(xaxis_type, data_properties, fitdata):
    #logging.info("using xaxis type {}".format(xaxis_type))

    if xaxis_type == "mud":
        residual = residual_mass(data_properties)
        return (t_0)*pd.Series((data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_mass(data_properties)
        return pd.Serites((data_properties.ud_mass + residual + data_properties.s_mass + residual))

    if xaxis_type == "mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return (pionmass)**2

    if xaxis_type == "mKsqr":
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return (kaonmass)**2


    if xaxis_type == "tmpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return ((t_0)*pionmass)**2

    if xaxis_type == "t_2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return (t_0)**2*(2.0*(kaonmass)**2 + (pionmass)**2)

    if xaxis_type == "xi":
        XI = read_fit_mass(data_properties, "xi", fitdata)
        return XI



def interpolate(data, phys_mpisqr, phys_mKsqr, model_str):

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
        a.append(d[0])
        mpisqrs.append(np.mean(d[1]))
        mKsqrs.append(np.mean(d[2]))
        observables.append(np.mean(d[3]))
        variances.append(np.var(d[3]))
        XIs.append(np.mean(d[4]))

    a = np.array(a)
    mpisqrs = np.array(mpisqrs)
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


    def s_a_pi(F_0,A,M_pi,M_K):
        M = (F_0
             - F_0 * mpisqrs  * np.log(mpisqrs/M_pi)
             + M_K * ((2*mKsqrs - mpisqrs) - (2*phys_mKsqr - phys_mpisqr))
             + A * (a**2) )
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    # rho_mass = 775.4
    # LAMBDA = 1000

    def chiral_LO_only(F_0, LAMBDA):
        XI = mpisqrs/(8*np.pi**2*F_0**2)
        M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2))
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)


    def chiral_LO_all(F_0, LAMBDA, A, M_K):
        XI = mpisqrs/(8*np.pi**2*F_0**2)
        M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2))
        M += A * (a**2)
        M += M_K * ((2*mKsqrs - mpisqrs) - (2*phys_mKsqr - phys_mpisqr))
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    def chiral_NLO_only(F_0, LAMBDA, OMEGA_F):
        XI = mpisqrs/(8*np.pi**2*F_0**2)
        # M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2))
        M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2) - 1.0/4.0 * XI**2*(np.log(mpisqrs/OMEGA_F**2))**2 )
        #M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)) - 1.0/4.0 * XI**2*(np.log(mpisqrs/OMEGA_F))**2 + c_F*XI**2)
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    def chiral_NLO_all(F_0, LAMBDA, OMEGA_F, A, M_K):
        XI = mpisqrs/(8*np.pi**2*F_0**2)
        M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2) - 1.0/4.0 * XI**2*(np.log(mpisqrs/OMEGA_F**2))**2 )
        M += A * (a**2)
        M += M_K * ((2*mKsqrs - mpisqrs) - (2*phys_mKsqr - phys_mpisqr))
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)


    def XI_LO_inverse_only(F_0, Lambda4):
        arg = mpisqrs/(Lambda4**2)
        M = F_0 / (1 + XIs*np.log(arg))
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    def XI_NLO_inverse_only(F_0, Lambda4, cF):
        arg = mpisqrs/(Lambda4**2)

        M = F_0 / (1 + XIs*np.log(arg) - (1.0/4.0)*(XIs*np.log(arg))**2 - cF*(XIs**2))
        sqr_diff = (observables - M)**2
        return np.sum(sqr_diff/variances)

    def XI_LO_only(F_0):
        M = F_0 * (1 - XIs*np.log(XIs))
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    def XI_NLO_only(F_0):
        M = F_0 * (1 - XIs*np.log(XIs)+(5.0/4.0)*(XIs*np.log(XIs))**2)
        sqr_diff = (observables - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)

    def mpisqrbymq_LO(F_0, c3):
        M = 2*F_0 * (1 + 0.5*XIs*np.log(XIs)+c3*XIs*(1.0-(9.0/2.0)*XIs*np.log(XIs)))
        sqr_diff = (observables**2/mqs - M)**2
        print sqr_diff
        return np.sum(sqr_diff/variances)


    # def model(F_0, LAMBDA, OMEGA_F):
    #     XI = mpisqrs/(8*np.pi**2*F_0**2)
    #     M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA)**2) - 1.0/4.0 * XI**2*(np.log(mpisqrs/OMEGA_F**2))**2 )
    #     #M = F_0 * (1 - XI*np.log(mpisqrs/(LAMBDA_4)) - 1.0/4.0 * XI**2*(np.log(mpisqrs/OMEGA_F))**2 + c_F*XI**2)
    #     sqr_diff = (observables - M)**2
    #     print sqr_diff
    #     return np.sum(sqr_diff/variances)

    models = {"chiral_LO_only": chiral_LO_only, "chiral_LO_all": chiral_LO_all, "s_a_pi": s_a_pi, "chiral_NLO_only": chiral_NLO_only, "chiral_NLO_all": chiral_NLO_all, "XI_LO_only": XI_LO_only, "XI_NLO_only": XI_NLO_only,
              "XI_LO_inverse_only": XI_LO_inverse_only, "XI_NLO_inverse_only": XI_NLO_inverse_only, "mpisqrbymq": mpisqrbymq_LO}
    model = models[model_str]

    ARGS = inspect.getargspec(model).args

    def weighted_sqr_diff(*args):
        sqr_diff = (observables - model(*args))**2
        return np.sum(sqr_diff/variances)

    # print a
    # exit()


    print dps
    print "obs", observables
    print a
    print mpisqrs
    print phys_mpisqr
    print mpisqrs-phys_mpisqr
    # exit(-1)

    dof = float(len(observables)-len(ARGS))

    logging.info("DOF {}".format(dof))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    #phys_obs_params = {"phys_obs":guess_phys_obs, "error_phys_obs":guess_phys_obs*0.01}
    F_0_params = {"F_0":guess_F_0, "error_F_0":guess_F_0*0.01}
    # A_params = {"A":guess_A, "error_A": 0.01}
    # M_pi_params = {"M_pi":guess_M_pi, "error_M_pi":min(guess_M_pi*0.01,0.01)}
    # M_K_params = {"M_K":guess_M_K, "error_M_K":min(guess_M_K*0.01,0.01)}
    # M_chiral_log = {"chiral_log":guess_chiral_log, "error_chiral_log": 0.01}

    # A_fixed = {"A":guess_A, "fix_A":True}
    # A_fixed = {"A": 0.0, "fix_A":True}

    #all_params= phys_obs_params.copy()
    all_params= F_0_params.copy()

    for arg in ARGS:
        all_params[arg] = 10.0
        e_arg = "error_"+ arg
        all_params[e_arg] = 0.1


    # all_params.update(A_params)
    # all_params.update(M_pi_params)
    # all_params.update(M_K_params)
    # all_params.update(M_chiral_log)
    # print all_params
    # exit(-1)


    m = Minuit(model, errordef=dof, print_level=0, pedantic=True, **all_params)

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
    # print "PHYSICAL", m.values["F_0"] * (1 - XI_phys*np.log(phys_mpisqr/(m.values["LAMBDA"])**2))


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

    phys_mpisqr = (phys_pion)**2
    phys_mKsqr = (phys_kaon)**2


    fit_paramsters = interpolate(alldata, phys_mpisqr, phys_mKsqr, options.model)

    write_data(fit_paramsters, options.output_stub, ".fit", options.model)

    exit()

    for data_group in groups:

        fit_params = interpolate(data_group)

        fitstring = ", ".join(["{}: {:.6f} +/- {:.6f}".format(k, v, fit_params.errors[k]) for
                               k, v in fit_params.values.iteritems()])
        logging.info("fit parameters found to be {}".format(fitstring))

        phys_t0 = 0.1465

        t0_ch, phys_err = plot_fitline(data_group, fit_params, ftype, phys_x,
                                        options.output_stub)

        ainv = hbar_c * t0_ch / phys_t0
        ainv_err = hbar_c * phys_err / phys_t0
        logging.info("ainv = {} +/- {}".format(ainv, ainv_err))

        write_data(t0_ch, ainv, ainv_err, options.output_stub, "_a_inv")

    xlabels = {"mud": r"$t_0^{1/2} m_{ud}$", "tmpisqr": r"$t_0 (m_{\pi})^2$",
               "t_2mksqr-mpisqr": r"$t_0 (2m_k^2+m_{\pi}^2)^2$"}
    finish_plot(beta, ftype, xlabels[options.xaxis], legend_handles, options.output_stub, options.pdf)


if __name__ == "__main__":

    axis_choices = ["mud", "mud_s", "mpi", "tmpisqr", "t_2mksqr-mpisqr"]

    models = ["chiral_LO_only", "chiral_NLO_only", "chiral_LO_all", "chiral_NLO_all", "s_a_pi", "XI_LO_only", "XI_NLO_only", "XI_LO_inverse_only", "XI_NLO_inverse_only", "mpisqrbymq"]

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
