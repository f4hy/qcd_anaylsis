import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import logging
import argparse
from matplotlib.widgets import CheckButtons
import os
import pandas as pd
import math

from cStringIO import StringIO
import numpy as np
import re

from residualmasses import residual_mass

from commonplotlib.plot_helpers import print_paren_error

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi
from data_params import Zs, Zv

from commonplotlib.auto_key import auto_key

from ratio_methods import ratio_chain

from alpha_s import get_alpha

plotsettings = dict(linestyle="none", ms=12, elinewidth=4,
                    capsize=8, capthick=2, mew=2, c="k")



meanplot = {}

#     addplot(plots, axe, fill, x=mDs_inv, y=y1, (label="fit $\\beta=4.17$",  ls="--", lw=2, color='b'))

def addplot(plots, axe, fill, save, x=None, y=None, params=None):

    if fill:
        axe.fill_between(x, meanplot[params["label"]], y, color=params["color"], alpha=0.01)
    else:
        if save:
            meanplot[params["label"]] = y
        plots.extend(axe.plot(x, y, **params))


def add_chiral_fit(axe, xran, chiral_fit_file=None, options=None):

    values = {}
    errors = {}

    for i in chiral_fit_file:
        if i.startswith("#"):
            fittype = i.split(" ")[0][1:]
            continue

        name, val, err = (j.strip() for j in i.replace("+/-", ",").split(","))
        values[name] = float(val)
        errors[name] = float(err)


    try:
        values[" M_\pi<"] = re.search("cut([0-9]+)", chiral_fit_file.name).group(1)
        errors[" M_\pi<"] = 0
    except:
        pass

    #ratio_chain(fittype, values)

    return choose_fit(axe, xran, fittype, values, errors, fill=False, save=True, options=options)

def add_boot_fit(axe, xran, boot_fit_file=None, options=None):

    # bootdata = pd.read_csv(boot_fit_file)
    # print bootdata
    # for i in bootdata:
    #     print i
    meanfilename = boot_fit_file.name.replace(".boot", ".fit")
    meanfile = open(meanfilename)
    meanfit = add_chiral_fit(axe, xran, chiral_fit_file=meanfile, options=options)

    comment = boot_fit_file.readline().strip("#\n ")
    comments = comment.split(",")
    model, names = comments[0], comments[1:]

    # for i in boot_fit_file:
    #     values = dict(zip(names,map(float,i.split(","))))
    #     errors = dict(zip(names,[0.0]*len(names)))

    #     x = choose_fit(axe, xran, model, values, errors, fill=True, save=False)
    return meanfit




def choose_fit(axe, xran, fittype, values, errors, fill=False, save=False, options=None):
    if fittype.startswith("combined"):
        if "fpi" in options.ydata:
            fittype = fittype.replace("combined", "FPI")
        else:
            fittype = fittype.replace("combined", "mpisqrbymq")
    if fittype.startswith("FPI_XI_inverse_NNLO"):
        return add_XI_inverse_NNLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_XI_NLO"):
        return add_XI_NLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_XI_NNLO"):
        return add_XI_NNLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_x_NLO_all"):
        return add_X_NLO_all_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_x_NLO"):
        return add_X_NLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_x_NNLO_fixa0"):
        return add_X_NNLO_all_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_x_NNLO_all"):
        return add_X_NNLO_fixa0_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FPI_x_NNLO"):
        return add_X_NNLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_const"):
        return add_mpisqrbymq_const_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_xi_NLO"):
        return add_mpisqrbymq_xi_NLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_XI_NNLO"):
        return add_mpisqrbymq_xi_NNLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_x_NLO_all"):
        return add_mpisqrbymq_x_NLO_all_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_x_NLO"):
        return add_mpisqrbymq_x_NLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_x_NNLO_fixa0"):
        return add_mpisqrbymq_x_NNLO_fixa0_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_x_NNLO_all"):
        return add_mpisqrbymq_x_NNLO_all_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mpisqrbymq_x_NNLO"):
        return add_mpisqrbymq_x_NNLO_fit(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fD_chiral"):
        return add_fD_chiral(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fDsbyfD_chiral"):
        return add_fDsbyfD_chiral(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("MD_linear"):
        return add_MD_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("MDs_linear"):
        return add_MDs_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FD_linear"):
        return add_FD_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FDA_linear"):
        return add_FD_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FDs_linear"):
        return add_FDs_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("FDsA_linear"):
        return add_FDs_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)


    if fittype.startswith("FDsbyFD_linear"):
        return add_FDsbyFD_linear_mpisqr(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("Mhs_minus_Mhh"):
        return add_Mhs_minus_Mhh(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("quad_Mhs_minus_Mhh"):
        return add_quad_Mhs_minus_Mhh(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdsqrtm_ratio"):
        return fdssqrtms_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("ms_mq_ma_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("m_mq_ma_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mD_corrected_pole_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mDs_corrected_pole_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mD_pole_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("mDs_pole_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)


    if fittype.startswith("fdsqrtmd_matched_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdsqrtm_mq_ma_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_matched_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_mq_ma_ratio"):
        return fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_mq_ratio"):
        return fdssqrtms_mq_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_ratio"):
        return fdssqrtms_ratio(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdsqrtm_HQET_matched_nom2"):
        return fdsqrtm_noscale(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdAsqrtm_HQET_matched_alphas"):
        return fdsqrtm_noscale_alphas(axe, xran, values, errors, fill=fill, save=save)


    if fittype.startswith("fdsqrtm_HQET_matched_alphas"):
        return fdsqrtm_noscale_alphas(axe, xran, values, errors, fill=fill, save=save)



    if fittype.startswith("fdsqrtm_HQET_matched"):
        return fdsqrtm_noscale(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdsqrtm"):
        return fdsqrtm(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdsAsqrtms_HQET_matched_alphas"):
        return fdssqrtms_noscale_alphas(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched_alphas"):
        return fdssqrtms_noscale_alphas(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched"):
        return fdssqrtms_noscale(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched_nom2_alphas"):
        return fdssqrtms_noscale_alphas(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched_nom2_pade2"):
        return fdssqrtms_noscale_pade2(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched_nom2_pade"):
        return fdssqrtms_noscale_pade(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms_HQET_matched_nom2"):
        return fdssqrtms_noscale(axe, xran, values, errors, fill=fill, save=save)

    if fittype.startswith("fdssqrtms"):
        return fdssqrtms(axe, xran, values, errors, fill=fill, save=save)

    logging.error("{} not supported".format(fittype))
    exit(-1)

    if "OMEGA_F" in values:
        return add_NNLO_chiral_fit(axe, xran, values, errors, fill=fill, save=save)
    else:
        return add_NLO_chiral_fit(axe, xran, values, errors, fill=fill, save=save)


def format_parameters(paramname):

    if paramname == "beta":
        return "\\beta"
    if paramname == "alpha":
        return "\\alpha"
    if paramname == "ellphys":
        return "\\ell^{phys}"
    if paramname == "c4":
        return "c_4"
    if paramname == "c3":
        return "c_3"
    if paramname == "Lambda3":
        return "\Lambda_3"
    if paramname == "Lambda4":
        return "\Lambda_4"
    if paramname == "Lambda12":
        return "\Lambda_{12}"
    if paramname == "km":
        return "k_M"
    if paramname == "kf":
        return "k_f"
    if paramname == "gamma_1":
        return "\\gamma_1"
    if paramname == "gamma_2":
        return "\\gamma_2"
    if paramname == "gamma_s1":
        return "\\gamma_{s1}"
    if paramname == "gamma_s2":
        return "\\gamma_{s2}"
    if paramname == "l4":
        return "\\bar\\ell_{4}"
    if paramname == "l3":
        return "\\bar\\ell_{3}"
    if paramname == "sigma13":
        return "\Sigma^{1/3}[\\bar{MS}]"
    if paramname == "f0":
        return "f_0"
    if paramname == "C1":
        return "C_1"
    if paramname == "C2":
        return "C_2"
    if paramname == "Fsqrtm_inf":
        return "F\sqrt{m}^\infty "
    if paramname == "eta":
        return " \\eta"
    if paramname == "gamma":
        return "\\gamma"
    if paramname == "mu":
        return "\\mu"
    return paramname

# def add_XI_LO_fit(axe, xran, values, errors, fill=False, save=False):
#     """
#     F = F_pi [ 1 + xi ln(M_pi^2 / Lambda_4^2) - 1/4 xi^2 (ln(m_pi^2/omega_f^2))^2 + xi^2cf ]
#     """

#     phys_mpisqr = (phys_pion)**2

#     LAMBDA4 = values["LAMBDA_4"]

#     x =  np.linspace(phys_mpisqr, xran[1])

#     XI = x/(8*(np.pi**2)*(values["F_PI"])**2)

#     XI_phys = phys_pion**2 / (phys_Fpi*8*np.pi**2)

#     XI_range = np.linspace(0, 0.2)

#     print XI_range

#     # y = values["F_PI"] / (1  + XI_range*np.log(phys_mpisqr/(LAMBDA4)**2))
#     y = values["F_PI"] * (1  - XI_range*np.log(phys_mpisqr/(LAMBDA4)**2))

#     p = axe.plot(XI_range,y, label="LO chiral fit", color='b', ls="--", lw=2)

#     return p


def add_mpisqrbymq_const_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    x = np.linspace(xran[0], xran[1], num=500)
    y = np.full_like(x, 2 * B)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "const fit: {}".format(paramstring)
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_xi_NLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    c3 = values["c3"]
    xi = np.linspace(xran[0], xran[1], num=500)

    y = 2 * B * (1.0 + 0.5 * xi * np.log(xi)) + c3 * xi

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NLO {}".format(paramstring)
    plabel = "NLO $ M_\pi<450$ MeV"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_xi_NNLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    F_0 = values["F_0"]
    c3 = values["c3"]
    c4 = values["c4"]
    alpha = values["alpha"]
    beta = values["beta"]
    ellphys = values["ellphys"]
    xi = np.linspace(xran[0], xran[1], num=500)

    y = 2 * B * (1.0 + 0.5 * xi * np.log(xi) + 7.0 / 8.0 * (xi * np.log(xi))**2 +
                 (c4 / F_0 - 1.0 / 3.0 * (ellphys + 16)) * np.log(xi) * xi**2) + c3 * xi * (1 - 5 * xi * np.log(xi)) + alpha * xi**2

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NNLO {}".format(paramstring)
    plabel = "NNLO"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_x_NLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    LAMBDA3 = values["Lambda3"]
    F_0 = values["F_0"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg = LAMBDA3**2 / Msqr

    y = 2 * B * (1.0 - 0.5 * x * np.log(arg))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NLO {}".format(paramstring)
    plabel = "NLO $M_\pi < 450$ MeV"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_x_NLO_all_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA3 = values["Lambda3"]
    LAMBDA4 = values["Lambda4"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]
    gamma_s2 = values["gamma_s2"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))

    arg1 = (LAMBDA4**2) / Msqr
    arg2 = (LAMBDA3**2) / Msqr

    y = 2 * B * (1.0 - 0.5 * x * np.log(arg2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NLO $Mss=0$ $a \\to 0$ fit: {}".format(paramstring)
    plabel = "NLO $ M_\pi<450$ MeV"

    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_X_NLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_0 * (1 + x*np.log(arg1))
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA4 = values["Lambda4"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg = LAMBDA4**2 / Msqr
    y = F_0 * (1 + x * np.log(arg))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NLO {}".format(paramstring)
    plabel = "NLO $M_\pi < 450$ MeV"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_X_NLO_all_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_0 * (1 + x*np.log(arg1))
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA3 = values["Lambda3"]
    LAMBDA4 = values["Lambda4"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]
    gamma_s2 = values["gamma_s2"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg1 = LAMBDA4**2 / Msqr
    arg2 = LAMBDA3**2 / Msqr
    y = F_0 * (1 + x * np.log(arg1))
    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<450$ MeV"
    plabel = "NLO $\Delta Mss=0$ fit: {}".format(paramstring)
    plabel = "NLO $ M_\pi<450$ MeV"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_XI_NLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_pi [ 1 + xi ln(M_pi^2 / Lambda_4^2) ]
    """
    F_0 = values["F_0"]
    c4 = values["c4"]
    xi = np.linspace(xran[0], xran[1], num=500)

    y = F_0 * (1 - xi * np.log(xi)) + c4 * xi

    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<${}".format(values[" M_\\pi<"])
    plabel = "NLO {}".format(paramstring)
    plabel = "NLO $ M_\pi<450$ MeV"

    plots = []
    addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel, "ls":"--", "lw":4})
    return plots


def add_XI_NNLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_pi [ 1 + xi ln(M_pi^2 / Lambda_4^2) ]
    """
    F_0 = values["F_0"]
    c4 = values["c4"]
    beta = values["beta"]
    ellphys = values["ellphys"]
    xi = np.linspace(xran[0], xran[1], num=500)

    y = F_0 * (1 - xi * np.log(xi) + 5.0 / 4.0 * (xi * np.log(xi))**2 + 1 / 6.0 * (ellphys + 53.0 / 2.0)
               * xi * xi * np.log(xi)) + c4 * xi * (1 - 5 * xi * np.log(xi)) + beta * xi**2

    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO {}".format(paramstring)
    plabel = "NNLO"

    plots = []
    addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel, "ls":"--", "lw":4})
    return plots


def add_X_NNLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_0 * (1 + x*np.log(arg1))
    """
    B = values["B"]
    F_0 = values["F_0"]

    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    # LAMBDA12 = values["Lambda12"]
    km = values["km"]
    kf = values["kf"]

    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr
    # arg12 = LAMBDA12**2 / Msqr

    l1 = -0.4
    l2 = 4.3

    Lambda1sqr = (phys_pion**2) * np.exp(l1)
    Lambda2sqr = (phys_pion**2) * np.exp(l2)

    lnLambda12sqr = (7.0 * np.log(Lambda1sqr) + 8.0 * np.log(Lambda2sqr)) / 15.0
    lambda12sqr = np.exp(lnLambda12sqr)

    arg12 = lambda12sqr / Msqr

    lm = 1.0 / 51.0 * (60.0 * np.log(arg12) - 9.0 * np.log(arg3) + 49.0)
    lf = 1.0 / 30.0 * (30.0 * np.log(arg12) + 6.0 * np.log(arg3) - 6.0 * np.log(arg4) + 23.0)

    y = F_0 * (1.0 + x * np.log(arg4) - 5.0 / 4.0 * (x**2) * (lf)**2 + kf * x**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO {}".format(paramstring)
    plabel = "NNLO"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_X_NNLO_all_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_0 * (1 + x*np.log(arg1))
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    # LAMBDA12 = values["Lambda12"]
    km = values["km"]
    kf = values["kf"]

    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr
    # arg12 = LAMBDA12**2 / Msqr

    l1 = -0.4
    l2 = 4.3

    Lambda1sqr = (phys_pion**2) * np.exp(l1)
    Lambda2sqr = (phys_pion**2) * np.exp(l2)

    lnLambda12sqr = (7.0 * np.log(Lambda1sqr) + 8.0 * np.log(Lambda2sqr)) / 15.0
    lambda12sqr = np.exp(lnLambda12sqr)

    arg12 = lambda12sqr / Msqr

    lm = 1.0 / 51.0 * (60.0 * np.log(arg12) - 9.0 * np.log(arg3) + 49.0)
    lf = 1.0 / 30.0 * (30.0 * np.log(arg12) + 6.0 * np.log(arg3) - 6.0 * np.log(arg4) + 23.0)

    y = F_0 * (1.0 + x * np.log(arg4) - 5.0 / 4.0 * (x**2) * (lf)**2 + kf * x**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO Mss=0 fit: {}".format(paramstring)
    plabel = "NNLO $a\\to 0$ $\Delta Mss=0$ "
    plabel = "NNLO"

    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_X_NNLO_fixa0_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_0 * (1 + x*np.log(arg1))
    """
    B = values["B"]
    F_0 = values["F_0"]

    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    # LAMBDA12 = values["Lambda12"]
    km = values["km"]
    kf = values["kf"]

    x = np.linspace(xran[0], xran[1], num=500)

    gamma_2 = values["gamma_2"]

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr
    # arg12 = LAMBDA12**2 / Msqr

    l1 = -0.4
    l2 = 4.3

    Lambda1sqr = (phys_pion**2) * np.exp(l1)
    Lambda2sqr = (phys_pion**2) * np.exp(l2)

    lnLambda12sqr = (7.0 * np.log(Lambda1sqr) + 8.0 * np.log(Lambda2sqr)) / 15.0
    lambda12sqr = np.exp(lnLambda12sqr)

    arg12 = lambda12sqr / Msqr

    lm = 1.0 / 51.0 * (60.0 * np.log(arg12) - 9.0 * np.log(arg3) + 49.0)
    lf = 1.0 / 30.0 * (30.0 * np.log(arg12) + 6.0 * np.log(arg3) - 6.0 * np.log(arg4) + 23.0)

    y = F_0 * (1.0 + x * np.log(arg4) - 5.0 / 4.0 * (x**2) * (lf)**2 + kf * x**2) / (1 + gamma_2 * (0.05**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO Mss=0 fit: {}".format(paramstring)
    plabel = "NNLO"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_x_NNLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    # l12 = values["l12"]
    km = values["km"]
    kf = values["kf"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))

    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr
    # # arg12 = LAMBDA12**2 / Msqr
    l1 = -0.4
    l2 = 4.3
    Lambda1sqr = (phys_pion**2) * np.exp(l1)
    Lambda2sqr = (phys_pion**2) * np.exp(l2)

    lnLambda12sqr = (7.0 * np.log(Lambda1sqr) + 8.0 * np.log(Lambda2sqr)) / 15.0
    lambda12sqr = np.exp(lnLambda12sqr)

    arg12 = lambda12sqr / Msqr

    lm = 1.0 / 51.0 * (60.0 * np.log(arg12) - 9.0 * np.log(arg3) + 49.0)
    lf = 1.0 / 30.0 * (30.0 * np.log(arg12) + 6.0 * np.log(arg3) - 6.0 * np.log(arg4) + 23.0)

    y = 2 * B * (1.0 - 0.5 * x * np.log(arg3) + 17.0 / 8.0 * (x**2) * (lm)**2 + km * x**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO {}".format(paramstring)
    plabel = "NNLO"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_mpisqrbymq_x_NNLO_all_fit(axe, xran, values, errors, fill=False, save=False):
    """
    M/mq = 2B*(1+...)
    """
    B = values["B"]
    F_0 = values["F_0"]
    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    # l12 = values["l12"]
    km = values["km"]
    kf = values["kf"]
    x = np.linspace(xran[0], xran[1], num=500)

    Msqr = x * (8 * (np.pi**2) * (F_0**2))
    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr
    # # arg12 = LAMBDA12**2 / Msqr

    l1 = -0.4
    l2 = 4.3
    Lambda1sqr = (phys_pion**2) * np.exp(l1)
    Lambda2sqr = (phys_pion**2) * np.exp(l2)

    lnLambda12sqr = (7.0 * np.log(Lambda1sqr) + 8.0 * np.log(Lambda2sqr)) / 15.0
    lambda12sqr = np.exp(lnLambda12sqr)

    arg12 = lambda12sqr / Msqr

    lm = 1.0 / 51.0 * (60.0 * np.log(arg12) - 9.0 * np.log(arg3) + 49.0)
    lf = 1.0 / 30.0 * (30.0 * np.log(arg12) + 6.0 * np.log(arg3) - 6.0 * np.log(arg4) + 23.0)

    y = 2 * B * (1.0 - 0.5 * x * np.log(arg3) + 17.0 / 8.0 * (x**2) * (lm)**2 + km * x**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO Mss=0 $a \\to 0$ fit: {}".format(paramstring)
    plabel = "NNLO $a\\to 0$ $\Delta Mss=0$ "
    plabel = "NNLO"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=x, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_XI_inverse_NNLO_fit(axe, xran, values, errors, fill=False, save=False):
    """
    F = F_pi [ 1 + xi ln(M_pi^2 / Lambda_4^2) ]
    """
    B = values["B"]
    F_0 = values["F_0"]

    LAMBDA4 = values["Lambda4"]
    LAMBDA3 = values["Lambda3"]
    l12 = values["l12"]
    cm = values["cm"]
    cf = values["cf"]

    xi = np.linspace(xran[0], xran[1], num=500)

    Msqr = xi * (8 * (np.pi**2) * (F_0**2))

    arg4 = LAMBDA4**2 / Msqr
    arg3 = LAMBDA3**2 / Msqr

    lambda12sqr = (phys_pion**2) * np.exp(l12)

    arg12 = lambda12sqr / Msqr

    lnOmegaM = 1.0 / 15.0 * (60.0 * np.log(arg12) - 33.0 * np.log(arg3) - 12.0 * np.log(arg4) + 52.0)
    lnOmegaF = 1.0 / 3.0 * (-15.0 * np.log(arg12) + 18.0 * np.log(arg4) - 29.0 / 2.0)

    y = F_0 / (1.0 - xi * np.log(arg4) - 1.0 / 4.0 * (xi * lnOmegaF)**2 + cf * (xi**2))

    del values["l12"]
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NNLO inverse fit: {}".format(paramstring)
    plabel = "NNLO"

    plots = []
    addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel, "ls":"--", "lw":4})

    # for i in [l12-1.0, l12+1.0]:
    #     lambda12sqr = (phys_pion**2)*np.exp(i)

    #     arg12 = lambda12sqr/Msqr

    #     lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
    #     lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

    #     y = F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

    #     paramstring = " ".join("${}={}$".format(format_parameters(k),print_paren_error(float(v),float(errors[k])))
    #                            for k,v in sorted(values.iteritems()) )
    #     plabel = "NNLO inverse fit l12={} : {}".format(i,paramstring)

    #     addplot(plots, axe, fill, save, x=xi, y=y, params={"label":plabel, "ls":"--", "lw":4})

    return plots


def add_fD_chiral(axe, xran, values, errors, fill=False, save=False):
    f_D0 = values["f_D0"]

    mu = values["mu"]
    c1 = values["c1"]
    g = values["g"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    factor = 3.0 * (1 + 3.0 * g**2) / 4.0
    F = 114.64
    arg = mpisqr / mu**2
    y = f_D0 * (1.0 - factor * (mpisqr / (8 * (np.pi**2) * (F**2))) * np.log(arg) + c1 * mpisqr)

    # print y[1]
    # exit(-1)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NLO {}".format(paramstring)
    plabel = "NLO $M_\pi <450$ MeV"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_fDsbyfD_chiral(axe, xran, values, errors, fill=False, save=False):
    f = values["f"]

    mu = values["mu"]
    c1 = values["c1"]
    k = values["k"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    arg = mpisqr / mu**2
    y = (1.0 + k * (mpisqr / (8 * (np.pi**2) * (f**2))) * np.log(arg) + c1 * mpisqr)

    # print y[1]
    # exit(-1)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "NLO {}".format(paramstring)
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4})

    return plots


def add_MD_linear_mpisqr(axe, xran, values, errors, fill=False, save=False):
    MDphys = values["MDphys"]

    b = values["b"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = MDphys * (1.0 + b * (mpisqr - phys_pion**2))
    y1 = (1 + gamma_1 * (a_beta417**2)) * MDphys * (1.0 + b * (mpisqr - phys_pion**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4})
    addplot(plots, axe, fill, save, x=mpisqr, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})
    # axe.errorbar(phys_pion**2, y=MDphys, yerr=errors["MDphys"], **plotsettings)

    return plots


def add_MDs_linear_mpisqr(axe, xran, values, errors, fill=False, save=False):
    MDsphys = values["MDsphys"]

    b = values["b"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = MDsphys * (1.0 + b * (mpisqr - phys_pion**2))
    y1 = (1 + gamma_1 * (a_beta417**2)) * MDsphys * (1.0 + b * (mpisqr - phys_pion**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4})
    addplot(plots, axe, fill, save, x=mpisqr, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})
    # axe.errorbar(phys_pion**2, y=MDsphys, yerr=errors["MDsphys"], **plotsettings)

    return plots


def add_FD_linear_mpisqr(axe, xran, values, errors, fill=False, save=False):
    FDphys = values["FDphys"]

    b = values["b"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = FDphys * (1.0 + b * (mpisqr - phys_pion**2))
    y1 = (1 + gamma_1 * (a_beta417**2)) * FDphys * (1.0 + b * (mpisqr - phys_pion**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit in continuum limit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4, "c":'k'})
    addplot(plots, axe, fill, save, x=mpisqr, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})
    #axe.errorbar(phys_pion**2, y=FDphys, yerr=errors["FDphys"], **plotsettings)

    return plots


def add_FDs_linear_mpisqr(axe, xran, values, errors, fill=False, save=False):
    FDsphys = values["FDsphys"]

    b = values["b"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = FDsphys * (1.0 + b * (mpisqr - phys_pion**2))
    y1 = (1 + gamma_1 * (a_beta417**2)) * FDsphys * (1.0 + b * (mpisqr - phys_pion**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit in continuum limit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4, "c":'k'})
    addplot(plots, axe, fill, save, x=mpisqr, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})
    #axe.errorbar(phys_pion**2, y=FDsphys, yerr=errors["FDsphys"], **plotsettings)

    return plots


def add_FDsbyFD_linear_mpisqr(axe, xran, values, errors, fill=False, save=False):
    FDsbyFDphys = values["FDsbyFDphys"]

    b = values["b"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mpisqr = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = FDsbyFDphys * (1.0 + b * (mpisqr - phys_pion**2))
    y1 = (1 + gamma_1 * (a_beta417**2)) * FDsbyFDphys * (1.0 + b * (mpisqr - phys_pion**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mpisqr, y=y, params={"label":plabel,  "ls":"--", "lw":4})
    addplot(plots, axe, fill, save, x=mpisqr, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})
    # axe.errorbar(phys_pion**2, y=FDsbyFDphys, yerr=errors["FDsbyFDphys"], **plotsettings)

    return plots


def add_Mhs_minus_Mhh(axe, xran, values, errors, fill=False, save=False):
    M_Bs = values["M_Bs"]

    alpha = values["alpha"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mHH_inv = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = (M_Bs + alpha * (mHH_inv))
    y1 = (1 + gamma_1 * (a_beta417**2)) * (M_Bs + alpha * (mHH_inv))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Linear fit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mHH_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4})
    addplot(plots, axe, fill, save, x=mHH_inv, y=y1, params={"label":"Linear fit $\\beta:4.17$", "ls":":", "lw":4})

    logging.info("Ploting point at x={}, y={} pm {}".format(1.0 / 9460.30, M_Bs + alpha *
                                                            (1.0 / 9460.30), errors["M_Bs"] + errors["alpha"] * (1.0 / 9460.30)))

    axe.errorbar(1.0 / 9460.30, y=M_Bs + alpha * (1.0 / 9460.30),
                 yerr=errors["M_Bs"] + errors["alpha"] * (1.0 / 9460.30), label="test", **plotsettings)
    # axe.errorbar(phys_pion**2, y=FDsbyFDphys, yerr=errors["FDsbyFDphys"], **plotsettings)

    return plots


def add_quad_Mhs_minus_Mhh(axe, xran, values, errors, fill=False, save=False):
    M_Bs = values["M_Bs"]

    alpha = values["alpha"]
    beta = values["beta"]
    gamma_1 = values["gamma_1"]
    gamma_s1 = values["gamma_s1"]

    mHH_inv = np.linspace(xran[0], xran[1], num=500)

    a_beta417 = 197.3269788 / scale["4.17"]

    y = (M_Bs + alpha * (mHH_inv) + beta * (mHH_inv)**2)
    y1 = (1 + gamma_1 * (a_beta417**2)) * (M_Bs + alpha * (mHH_inv) + beta * (mHH_inv)**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    addplot(plots, axe, fill, save, x=mHH_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4})
    addplot(plots, axe, fill, save, x=mHH_inv, y=y1, params={"label":"Quadratic fit $\\beta:4.17$", "ls":":", "lw":4})

    logging.info("Ploting point at x={}, y={} pm {}".format(1.0 / 9460.30, M_Bs + alpha * (1.0 / 9460.30) + beta *
                                                            (1.0 / 9460.30)**2, errors["M_Bs"] + errors["alpha"] * (1.0 / 9460.30) + errors["beta"] * (1.0 / 9460.30)**2))

    axe.errorbar(1.0 / 9460.30, y=M_Bs + alpha * (1.0 / 9460.30) + beta * (1.0 / 9460.30)**2, yerr=errors["M_Bs"] + errors[
                 "alpha"] * (1.0 / 9460.30) + errors["beta"] * (1.0 / 9460.30)**2, label="test", **plotsettings)
    # axe.errorbar(phys_pion**2, y=FDsbyFDphys, yerr=errors["FDsbyFDphys"], **plotsettings)

    return plots


def fdsqrtm(axe, xran, values, errors, fill=False, save=False):

    Fsqrtm_inf = values["Fsqrtm_inf"]


    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mD_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mD_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]

    print a_beta417, a_beta435, a_beta447

    y = Fsqrtm_inf * (1 + C1 * mD_inv + C2 * mD_inv**2)
    y1 = y*(1+ gamma * (m * a_beta417) ** 2 + eta * m * a_beta417**2 + mu * a_beta417**2)
    y2 = y*(1+ gamma * (m * a_beta435) ** 2 + eta * m * a_beta435**2 + mu * a_beta435**2)
    y3 = y*(1+ gamma * (m * a_beta447) ** 2 + eta * m * a_beta447**2 + mu * a_beta447**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mD_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mD_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots


def fdssqrtms(axe, xran, values, errors, fill=False, save=False):

    Fssqrtms_inf = values["Fssqrtms_inf"]

    # values["C1"] *= 1000.0
    # values["C2"] *= (1000.0)**2
    # values["gamma"] *= 1.0 / (10000.0)
    # values["eta"] *= 1.0 / (100.0)
    # values["mu"] *= 0.001

    # errors["C1"] *= 1000.0
    # errors["C2"] *= (1000.0)**2
    # errors["gamma"] *= 1.0 / (10000.0)
    # errors["eta"] *= 1.0 / (100.0)
    # errors["mu"] *= 0.001

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]

    print a_beta417, a_beta435, a_beta447

    y = Fssqrtms_inf * (1 + C1 * mDs_inv + C2 * mDs_inv**2)
    y1 =y*(1 + gamma * (m * a_beta417) ** 2 + eta * m * a_beta417**2 + mu * a_beta417**2)
    y2 =y*(1 + gamma * (m * a_beta435) ** 2 + eta * m * a_beta435**2 + mu * a_beta435**2)
    y3 =y*(1 + gamma * (m * a_beta447) ** 2 + eta * m * a_beta447**2 + mu * a_beta447**2)

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots

def fdssqrtms_noscale(axe, xran, values, errors, fill=False, save=False):

    Fssqrtms_inf = values["Fssqrtms_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fssqrtms_inf * (1 + C1 * mDs_inv + C2 * mDs_inv**2)
    y1 = Fssqrtms_inf * (1 + C1 * mDs_inv + C2 * mDs_inv**2)*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * (m * a_beta417) ** 2 )
    y2 = Fssqrtms_inf * (1 + C1 * mDs_inv + C2 * mDs_inv**2)*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma * (m * a_beta435) ** 2 )
    y3 = Fssqrtms_inf * (1 + C1 * mDs_inv + C2 * mDs_inv**2)*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots

def fdsqrtm_noscale(axe, xran, values, errors, fill=False, save=False):

    Fsqrtm_inf = values["Fsqrtm_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mD_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mD_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fsqrtm_inf * (1 + C1 * mD_inv + C2 * mD_inv**2)
    y1 = y*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * (m * a_beta417) ** 2 )
    y2 = y*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma * (m * a_beta435) ** 2 )
    y3 = y*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mD_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mD_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots

def fdsqrtm_noscale_alphas(axe, xran, values, errors, fill=False, save=False):

    Fsqrtm_inf = values["Fsqrtm_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mD_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mD_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fsqrtm_inf * (1 + C1 * mD_inv + C2 * mD_inv**2)
    y1 = y*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * get_alpha(scale["4.17"])* (m * a_beta417) ** 2 )
    y2 = y*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma* get_alpha(scale["4.35"]) * (m * a_beta435) ** 2 )
    y3 = y*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma* get_alpha(scale["4.47"]) * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mD_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mD_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mD_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots


def fdssqrtms_noscale_alphas(axe, xran, values, errors, fill=False, save=False):

    Fssqrtms_inf = values["Fssqrtms_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fssqrtms_inf * (1.0 + C1 * mDs_inv + C2 * mDs_inv**2)

    y1 = y*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * get_alpha(scale["4.17"])* (m * a_beta417) ** 2 )
    y2 = y*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma* get_alpha(scale["4.35"]) * (m * a_beta435) ** 2 )
    y3 = y*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma* get_alpha(scale["4.47"]) * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots


def fdssqrtms_noscale_pade(axe, xran, values, errors, fill=False, save=False):

    Fssqrtms_inf = values["Fssqrtms_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fssqrtms_inf * (1.0/(1.0 + C1 * mDs_inv + C2 * mDs_inv**2))

    y1 = y*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * (m * a_beta417) ** 2 )
    y2 = y*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma * (m * a_beta435) ** 2 )
    y3 = y*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit PADE"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots

def fdssqrtms_noscale_pade2(axe, xran, values, errors, fill=False, save=False):

    Fssqrtms_inf = values["Fssqrtms_inf"]

    C1 = values["C1"]
    C2 = values["C2"]

    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]


    y = Fssqrtms_inf * ((1.0+ C1*mDs_inv)/(1.0  + C2*mDs_inv))

    y1 = y*(1+ eta * m * a_beta417**2 + mu * a_beta417**2+ gamma * (m * a_beta417) ** 2 )
    y2 = y*(1 + eta * m * a_beta435**2 + mu * a_beta435**2 + gamma * (m * a_beta435) ** 2 )
    y3 = y*(1+ eta * m * a_beta447**2 + mu * a_beta447**2 + gamma * (m * a_beta447) ** 2 )

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    plabel = "Qaudratic fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit PADE2"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mDs_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    # mB_inv = 1.0/5279.0
    # axe.errorbar(mB_inv, y=Fsqrtm_inf*(1 + C1 * mB_inv + C2 * mB_inv**2 ),
    #              yerr=errors["Fsqrtm_inf"]*(1 + errors["C1"] * mB_inv + errors["C2"] * mB_inv**2 ),
    #              label="test", color='k', ecolor='k', mec='k', alpha=0.2, **plotsettings)

    return plots



def fdssqrtms_ratio(axe, xran, values, errors, fill=False, save=False):

    z = values["z"]
    z2 = values["z2"]
    gamma_1 = values["gamma_1"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]

    y = 1 + z / m + z2 / (m**2)
    y1 = (1 + gamma_1 * (a_beta417**2)) * (1 + z / m + z2 / (m**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})

    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})

    return plots


def fdssqrtms_mq_ratio(axe, xran, values, errors, fill=False, save=False):

    z = values["z"]
    z2 = values["z2"]
    gamma_A = values["gamma_A"]

    mq_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mq_inv

    a_beta417 = 197.3269788 / scale["4.17"]

    y = 1 + z / m + z2 / (m**2)
    y1 = (1 + gamma_A * (a_beta417**2)) * (1 + z / m + z2 / (m**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mq_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})

    addplot(plots, axe, fill, save, x=mq_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})

    return plots


def fdssqrtms_mq_ma_ratio(axe, xran, values, errors, fill=False, save=False):

    z = values["z"]
    z2 = values["z2"]
    gamma_A = values["gamma_A"]
    gamma_MA = values["gamma_MA"]
    gamma_MMA = values["gamma_MMA"]

    mq_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mq_inv

    a_beta417 = 197.3269788 / scale["4.17"]
    a_beta435 = 197.3269788 / scale["4.35"]
    a_beta447 = 197.3269788 / scale["4.47"]

    y = 1 + z / m + z2 / (m**2)
    y1 = (1 + gamma_A * (a_beta417**2) + gamma_MA * m * (a_beta417**2) +
          gamma_MMA * (m * a_beta417)**2) * (1 + z / m + z2 / (m**2))
    y2 = (1 + gamma_A * (a_beta435**2) + gamma_MA * m * (a_beta435**2) +
          gamma_MMA * (m * a_beta435)**2) * (1 + z / m + z2 / (m**2))
    y3 = (1 + gamma_A * (a_beta447**2) + gamma_MA * m * (a_beta447**2) +
          gamma_MMA * (m * a_beta447)**2) * (1 + z / m + z2 / (m**2))




    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
    plabel = "Continuum and Chiral Limit"
    addplot(plots, axe, fill, save, x=mq_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})

    addplot(plots, axe, fill, save, x=mq_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})
    addplot(plots, axe, fill, save, x=mq_inv, y=y2, params={"label":"fit $\\beta:4.35$",  "ls":"--", "lw":4, "color":'r'})
    addplot(plots, axe, fill, save, x=mq_inv, y=y3, params={"label":"fit $\\beta:4.47$",  "ls":"--", "lw":4, "color":'m'})

    return plots




def fdsqrtm_ratio(axe, xran, values, errors, fill=False, save=False):

    z = values["z"]
    z2 = values["z2"]
    gamma_A = values["gamma_A"]

    mDs_inv = np.linspace(xran[0], xran[1], num=500)
    m = 1 / mDs_inv

    a_beta417 = 197.3269788 / scale["4.17"]

    y = 1 + z / m + z2 / (m**2)
    y1 = (1 + gamma_A * (a_beta417**2)) * (1 + z / m + z2 / (m**2))

    plots = []
    paramstring = " ".join("${}={}$".format(format_parameters(k), print_paren_error(float(v), float(errors[k])))
                           for k, v in sorted(values.iteritems()))
    #paramstring = "$ M_\pi<{}$".format(values[" M_\pi<"])
    print paramstring
    plabel = "fit"

    plabel = paramstring.replace("$ \eta", "\n $ \eta")
    if "cutoff" in values:
        plabel += " $M_\pi < {}$".format(values["cutoff"])
        plabel = "Continuum and Chiral Limit"
        addplot(plots, axe, fill, save, x=mDs_inv, y=y, params={"label":plabel,  "ls":"--", "lw":4, "color":"k"})

    addplot(plots, axe, fill, save, x=mDs_inv, y=y1, params={"label":"fit $\\beta:4.17$",  "ls":"--", "lw":4, "color":'b'})

    return plots
