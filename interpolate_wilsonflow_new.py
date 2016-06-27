#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass

from ensamble_info import data_params, read_fit_mass, phys_pion, phys_kaon

colors = ['b', 'k', 'c', 'y', 'm', 'b']

# hbar*c / 1 fm  = 197.327 MeV
hbar_c = 197.327


def read_files(files, xtype, fitdata):
    data = {}

    for f in files:
        logging.info("reading file {}".format(f))
        dp = data_params(f)
        df = pd.read_csv(f, comment='#', names=["flow"])

        mpi = xvalues("mpi", dp, fitdata)
        mk = xvalues("mk", dp, fitdata)

        data[dp] = ((np.mean(mpi), (np.mean(mk))), df.flow.values)
    return data


def xvalues(xaxis_type, data_properties, fitdata, t_0=None):
    logging.info("using xaxis type {}".format(xaxis_type))

    if xaxis_type == "mud":
        residual = residual_mass(data_properties)
        return (t_0)*pd.Series((data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_mass(data_properties)
        return pd.Serites((data_properties.ud_mass + residual + data_properties.s_mass + residual))


    if xaxis_type == "mpi":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return pionmass

    if xaxis_type == "mk":
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return kaonmass


    if xaxis_type == "tmpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return (t_0*pionmass)**2

    if xaxis_type == "t_2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return (t_0)**2*(2.0*(kaonmass)**2 + (pionmass)**2)


def interpolate(data, phys_mpi, phys_mk, phys_t0):

    logging.info("Fitting data")

    dps0 = []
    mpis0 = []
    mks0 = []
    flows0 = []
    variances0 = []

    dps1 = []
    mpis1 = []
    mks1 = []
    flows1 = []
    variances1 = []


    for dp, d in data[0].iteritems():
        dps0.append(dp)
        mpis0.append(d[0][0])
        mks0.append(d[0][1])
        flows0.append(np.mean(d[1]))
        n = float(len(d[1]))
        variances0.append(((n-1)*np.var(d[1])))

    for dp, d in data[1].iteritems():
        dps1.append(dp)
        mpis1.append(d[0][0])
        mks1.append(d[0][1])
        flows1.append(np.mean(d[1]))
        n = float(len(d[1]))
        variances1.append(((n-1)*np.var(d[1])))

    mpis0 = np.array(mpis0)
    mks0 = np.array(mks0)
    flows0 = np.array(flows0)
    variances0 = np.array(variances0)

    t0_0 = np.mean(flows0)


    mpis1 = np.array(mpis1)
    mks1 = np.array(mks1)
    flows1 = np.array(flows1)
    variances1 = np.array(variances1)
    t0_1 = np.mean(flows1)

    xmpi0 = (t0_0*mpis0)**2 - (phys_t0*phys_mpi)**2
    xmpi1 = (t0_1*mpis1)**2 - (phys_t0*phys_mpi)**2

    x2mkmpi0 = (t0_0**2)*(2*mks0**2-mpis0**2) - (phys_t0**2)*(2*phys_mk**2-phys_mpi**2)
    x2mkmpi1 = (t0_1**2)*(2*mks1**2-mpis1**2) - (phys_t0**2)*(2*phys_mk**2-phys_mpi**2)


    def weighted_sqr_diff(CP, CK, A, B):
        sqr_diff0 = (flows0 - A*(1+CP*(xmpi0)+CK*(x2mkmpi0)))**2
        sqr_diff1 = (flows1 - B*(1+CP*(xmpi1)+CK*(x2mkmpi1)))**2
        return np.sum(sqr_diff0/variances0) + np.sum(sqr_diff1/variances1)

    dof = float(len(flows0)+len(flows1))-4.0

    guess_A = np.mean(flows0)
    guess_B = np.mean(flows1)
    guess_slope1 = -0.3
    guess_slope2 = 0.0

    m = Minuit(weighted_sqr_diff,
               CP=guess_slope1, error_CP=guess_slope1*0.01,
               CK=guess_slope2, error_CK=guess_slope1*0.01,
               A=guess_A, error_A=guess_A*0.01,
               B=guess_B, error_B=guess_B*0.01,
               errordef=dof, print_level=0, pedantic=True)

    results = m.migrad()
    logging.debug(results)
    logging.debug("variances {} {}".format(variances0, variances1))
    logging.debug("stds {} {}".format(np.sqrt(variances0), np.sqrt(variances1)))

    logging.info("values: {}".format(m.values))
    logging.info("errors: {}".format(m.errors))
    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(m.fval, dof, m.fval/dof))
    logging.info('covariance {}'.format(m.covariance))

    if not m.get_fmin().is_valid:
        print "NOT VALID"
        exit(-1)

    return m


def plot_fitline(datagroups, fit_params, ftype, phys_all, outstub):
    logging.info("ploting")
    phys_mk, phys_mpi, phys_t0 = phys_all
    f, axes = plt.subplots(2, sharex=True)

    c = colors.pop()
    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                        capthick=2, mew=3, aa=True, fmt='o')

    xvalues = []

    t0_ch0 = fit_params.values["A"]
    t0_ch1 = fit_params.values["B"]

    m = fit_params

    t0_ch_std0 = m.errors["A"]

    logging.info("Determined at physical point t_0 = {:.5f} +/- {:.5f}".format(t0_ch0, t0_ch_std0))

    t0_ch_std1 = m.errors["B"]

    logging.info("Determined at physical point t_0 = {:.5f} +/- {:.5f}".format(t0_ch1, t0_ch_std1))

    return axes, (t0_ch0, t0_ch1), (t0_ch_std0, t0_ch_std1)


def finish_plot(axes, betas, ftype, xlabel, legend_handles, outstub):
    fontsettings = dict(fontsize=20)

    for i in range(len(betas)):
        axes[i].set_title(r'$\beta={}$    ${}$'.format(betas[i], ftype), **fontsettings)
        axes[i].set_ylabel("${} / a$".format(ftype), **fontsettings)
        axes[i].set_xlabel(xlabel, **fontsettings)

    plt.legend(handles=sorted(legend_handles), loc=0, **fontsettings)

    if outstub is not None:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        filename = outstub+".png"
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_data(betas, t0_chs, a_invs, a_inv_errs, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("{}, {}, {} +/- {}\n".format(betas[0], t0_chs[0], a_invs[0], a_inv_errs[0]))
        ofile.write("{}, {}, {} +/- {}\n".format(betas[1], t0_chs[1], a_invs[1], a_inv_errs[1]))


def interpolate_wilsonflow(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    try:
        beta = re.search("_b(4\.[0-9]*)_", options.files[0]).group(1)
    except:
        logging.error("failed to determine beta for {}".format(options.files[0]))
        exit(-1)

    if "t0" in options.files[0]:
        ftype = "t_0^{1/2}"
    elif "w0" in options.files[0]:
        ftype = "w_0"
    else:
        ftype = None

    alldata = read_files(options.files, options.xaxis, options.fitdata)

    # if options.xaxis == "mud":
    #     phys_x = 0
    # if options.xaxis == "tmpisqr":
    #     phys_x = ((0.1465/hbar_c)*phys_pion)**2
    # if options.xaxis == "t_2mksqr-mpisqr":
    #     phys_x = ((0.1465/hbar_c)**2)*(2*(phys_kaon**2) + phys_pion**2)

    phys_mk = phys_kaon
    phys_mpi = phys_pion
    phys_t0 = (0.1465/hbar_c)


    legend_handles = []

    betas = sorted(set(i.beta for i in alldata.keys()))

    groups = [{k: v for k, v in alldata.iteritems() if k.beta == b} for b in betas]



    fit_params = interpolate(groups, phys_mpi, phys_mk, phys_t0)

    fitstring = ", ".join(["{}: {:.6f} +/- {:.6f}".format(k, v, fit_params.errors[k]) for
                           k, v in fit_params.values.iteritems()])
    logging.info("fit parameters found to be {}".format(fitstring))

    phys_t0 = 0.1465

    axes, t0_chs, phys_errs = plot_fitline(groups, fit_params, ftype, (phys_mk, phys_mpi, phys_t0),
                                           options.output_stub)

    ainv0 = hbar_c * t0_chs[0] / phys_t0
    ainv_err0 = hbar_c * phys_errs[0] / phys_t0
    logging.info("ainv0 = {} +/- {}".format(ainv0, ainv_err0))

    ainv1 = hbar_c * t0_chs[1] / phys_t0
    ainv_err1 = hbar_c * phys_errs[1] / phys_t0
    logging.info("ainv1 = {} +/- {}".format(ainv1, ainv_err1))

    write_data(betas, t0_chs, (ainv0, ainv1), (ainv_err0, ainv_err1), options.output_stub, "_a_inv")


    xlabels = {"mud": r"$t_0^{1/2} m_{ud}$", "tmpisqr": r"$t_0 (m_{\pi})^2$",
               "t_2mksqr-mpisqr": r"$t_0 (2m_k^2+m_{\pi}^2)^2$"}
    finish_plot(axes, betas, ftype, xlabels[options.xaxis], legend_handles, options.output_stub)


if __name__ == "__main__":

    axis_choices = ["mud", "mud_s", "mpi", "tmpisqr", "t_2mksqr-mpisqr"]

    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--seperate_strange", action="store_true",
                        help="fit different strange values seperately")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("--spinaverage", action="store_true",
                        help="spinaverage vector with pseudoscalar")
    parser.add_argument("-p", "--physical", type=float,
                        help="add physical point")
    parser.add_argument("-m", "--mcc", type=float,
                        help="scale data using given heavy mass")
    parser.add_argument("-f", "--funct", type=str, default="line",
                        help="which function to fit to")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.xaxis == "tmpisqr":
        for f in args.files:
            if "t0" not in f:
                raise argparse.ArgumentTypeError("tmpisqr requires t0 data".format(f))

    interpolate_wilsonflow(args)
