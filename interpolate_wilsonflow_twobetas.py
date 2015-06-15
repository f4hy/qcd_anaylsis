#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass

from ensamble_info import data_params, read_fit_mass

colors = ['b', 'k', 'c', 'y', 'm', 'b']

# hbar*c / 1 fm  = 197.327 MeV
hbar_c = 197.327


def read_files(files, xtype, fitdata):
    data = {}

    for f in files:
        logging.info("reading file {}".format(f))
        dp = data_params(f)
        df = pd.read_csv(f, comment='#', names=["flow"])
        xvalue = xvalues(xtype, dp, fitdata, t_0=np.mean(df.flow.values))

        data[dp] = (np.mean(xvalue), df.flow.values)

    return data


def xvalues(xaxis_type, data_properties, fitdata, t_0=None):
    logging.info("using xaxis type {}".format(xaxis_type))

    if xaxis_type == "mud":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return (t_0)*pd.Series((data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return pd.Serites((data_properties.ud_mass + residual + data_properties.s_mass + residual))

    if xaxis_type == "tmpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return ((t_0)*pionmass)**2

    if xaxis_type == "t_2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return (t_0)**2*(2.0*(kaonmass)**2 + (pionmass)**2)


def interpolate(data, physical_x):

    logging.info("Fitting data")

    dps0 = []
    xvalues0 = []
    flows0 = []
    variances0 = []

    dps1 = []
    xvalues1 = []
    flows1 = []
    variances1 = []


    for dp, d in data[0].iteritems():
        dps0.append(dp)
        xvalues0.append(d[0])
        flows0.append(np.mean(d[1]))
        n = float(len(d[1]))
        variances0.append(((n-1)*np.var(d[1])))

    for dp, d in data[1].iteritems():
        dps1.append(dp)
        xvalues1.append(d[0])
        flows1.append(np.mean(d[1]))
        n = float(len(d[1]))
        variances1.append(((n-1)*np.var(d[1])))

    xvalues0 = np.array(xvalues0)
    flows0 = np.array(flows0)
    variances0 = np.array(variances0)

    xvalues1 = np.array(xvalues1)
    flows1 = np.array(flows1)
    variances1 = np.array(variances1)

    def weighted_sqr_diff(C, A, B):
        sqr_diff0 = (flows0 - A*(1+C*(xvalues0-physical_x)))**2
        sqr_diff1 = (flows1 - B*(1+C*(xvalues1-physical_x)))**2
        return np.sum(sqr_diff0/variances0) + np.sum(sqr_diff1/variances1)

    dof = float(len(flows0)+len(flows1))-3.0

    guess_A = np.mean(flows0)
    guess_B = np.mean(flows1)
    guess_slope = (min(flows1)-max(flows1)) / (max(physical_x-xvalues1) - min(xvalues1-physical_x))
    #guess_slope = -0.3

    logging.debug("Guessing {}".format((guess_slope, guess_A, guess_B)))
    logging.debug("fval at guess {}".format(weighted_sqr_diff(guess_slope, guess_A, guess_B)))


    m = Minuit(weighted_sqr_diff, C=guess_slope, error_C=guess_slope*0.01,
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

    #exit(-1)

    return m


def plot_fitline(datagroups, fit_params, ftype, phys_x, outstub):
    logging.info("ploting")

    f, axes = plt.subplots(2, sharex=True)

    c = colors.pop()
    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                        capthick=2, mew=3, aa=True, fmt='o')

    xvalues = []

    for i in range(len(datagroups)):
        data = datagroups[i]
        for dp, d in data.iteritems():
            xvalue, flow = d
            N = len(flow)
            y = np.mean(flow)
            err = np.sqrt((N-1)*(np.std(flow)**2))
            logging.info("{}, {}={}, err={}".format(dp, ftype, y, err))
            patch = axes[i].errorbar(xvalue, y, yerr=err, color=c, ecolor=c, mec=c, **plotsettings)

            xvalues.append(xvalue)


    xdata = np.arange(phys_x-0.01, max(xvalues)+0.005, 0.001)
    xdata_phys = xdata-phys_x

    mydata0 = fit_params.values["A"]*(1+fit_params.values["C"]*(xdata_phys))
    mydata1 = fit_params.values["B"]*(1+fit_params.values["C"]*(xdata_phys))

    # t0_ch0 = fit_params.values["A"]*(1+fit_params.values["C"]*phys_x)
    # t0_ch1 = fit_params.values["B"]*(1+fit_params.values["C"]*phys_x)
    # print t0_ch0
    # print t0_ch1

    t0_ch0 = fit_params.values["A"]
    t0_ch1 = fit_params.values["B"]

    axes[0].plot(xdata, mydata0, color=c)
    axes[1].plot(xdata, mydata1, color=c)


    m = fit_params
    t1 = (m.errors["A"]*(1+m.values["C"]*xdata_phys))**2
    t2 = ((m.values["A"])*xdata_phys*m.errors["C"])**2
    t3 = 2*xdata_phys*mydata0*m.covariance[("A", "C")]
    perry0 = t1+t2+t3

    t1 = (m.errors["B"]*(1+m.values["C"]*xdata_phys))**2
    t2 = ((m.values["B"])*xdata_phys*m.errors["C"])**2
    t3 = 2*xdata_phys*mydata1*m.covariance[("B", "C")]
    perry1 = t1+t2+t3



    axes[0].fill_between(xdata, mydata0, mydata0+np.sqrt(perry0), facecolor=c, alpha=0.1, lw=0, zorder=-10)
    axes[0].fill_between(xdata, mydata0, mydata0-np.sqrt(perry0), facecolor=c, alpha=0.1, lw=0, zorder=-10)
    axes[1].fill_between(xdata, mydata1, mydata1+np.sqrt(perry1), facecolor=c, alpha=0.1, lw=0, zorder=-10)
    axes[1].fill_between(xdata, mydata1, mydata1-np.sqrt(perry1), facecolor=c, alpha=0.1, lw=0, zorder=-10)



    t0_ch_std0 = m.errors["A"]
    axes[0].errorbar(phys_x, t0_ch0, yerr=t0_ch_std0,
                     color="r", mec="r", **plotsettings)


    logging.info("Determined at physical point t_0 = {:.5f} +/- {:.5f}".format(t0_ch0, t0_ch_std0))



    t0_ch_std1 = m.errors["B"]
    axes[1].errorbar(phys_x, t0_ch1, yerr=t0_ch_std1,
                     color="r", mec="r", **plotsettings)

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

    if options.xaxis == "mud":
        phys_x = 0
    if options.xaxis == "tmpisqr":
        phys_x = ((0.1465/hbar_c)*135.0)**2
    if options.xaxis == "t_2mksqr-mpisqr":
        phys_x = ((0.1465/hbar_c)**2)*(2*(495**2) + 138.0**2)

    legend_handles = []

    betas = sorted(set(i.beta for i in alldata.keys()))

    groups = [{k: v for k, v in alldata.iteritems() if k.beta == b} for b in betas]



    fit_params = interpolate(groups, phys_x)

    fitstring = ", ".join(["{}: {:.6f} +/- {:.6f}".format(k, v, fit_params.errors[k]) for
                           k, v in fit_params.values.iteritems()])
    logging.info("fit parameters found to be {}".format(fitstring))

    phys_t0 = 0.1465

    axes, t0_chs, phys_errs = plot_fitline(groups, fit_params, ftype, phys_x,
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
