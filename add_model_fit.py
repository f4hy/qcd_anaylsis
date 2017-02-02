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
from physical_values import hbar_c

from commonplotlib.plot_helpers import print_paren_error

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi
from data_params import Zs, Zv

from commonplotlib.auto_key import auto_key

from ratio_methods import ratio_chain

from alpha_s import get_alpha

import new_fit_model

import inspect

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


def add_model_fit(axe, xran, boot_fit_file, options=None):

    header = boot_fit_file.readline().split(",")
    name = header[0].strip("# ")
    columns = [s.strip("\n ,") for s in header[1:]]

    df = pd.read_csv(boot_fit_file, sep=",", delimiter=",", names=columns)

    model = dict(inspect.getmembers(new_fit_model,inspect.isclass))[name]
    m = model([], options)
    x = np.linspace(xran[0], xran[1], num=100)
    means = df.mean()
    params = [means[i] for i in m.contlim_args]


    logging.info("plotting line with params {}".format(params))
    y = m.m(x, *params)
    plot_handles = axe.plot(x,y, color='k', lw=2, label=m.label)

    ys = []
    modelpoints = []
    for i,row in df.iterrows():
        p = [row[n] for n in m.contlim_args]
        ys.append(m.m(x,*p))
        if options.model_fit_point:
            modelpoints.append(m.m(options.model_fit_point,*p))

    ponesigma = np.percentile(ys, 84.1, axis=0)
    monesigma = np.percentile(ys, 15.9, axis=0)
    # axe.fill_between(x, ponesigma, monesigma, alpha=0.1, color="k")


    if options.model_fit_point:
        point_x = options.model_fit_point
        point_y = m.m(point_x, *params)
        py_upper = np.percentile(modelpoints, 84.1, axis=0) - point_y
        py_lower = point_y- np.percentile(modelpoints, 15.9, axis=0)

        logging.info("At model point specified, value is {} + {} - {}".format(point_y, py_upper, py_lower))
        axe.errorbar(point_x,point_y, yerr=[[py_lower],[py_upper]], color='k', ms=15, elinewidth=4, mew=2, capthick=2, capsize=8)
        # exit(-1)

    try:
        finite_beta_handles = []
        scale = {"4.17": 2453.1, "4.35": 3609.7, "4.47": 4496.1}
        colors = {"4.17":"b", "4.35":"r", "4.47":"m"}
        for beta in options.model_finite_fits:
        # for beta in ["4.17"]:
            logging.info("adding {}".format(beta))
            a_gev = 1000.0/(scale[beta])
            m.consts["a"] = a_gev
            m.consts["lat"] = hbar_c/scale[beta]
            m.consts["alphas"] = get_alpha(scale[beta])
            finbeta_params = [means[i] for i in m.finbeta_args]
            ybeta = m.m(x, *finbeta_params)
            h = axe.plot(x,ybeta, color=colors[beta], lw=2, label="fit at $\\beta$={}".format(beta))
            plot_handles.extend(h)
    except AttributeError as e:
        logging.warn("This model doesn't support plotting at finite beta")
    return plot_handles
