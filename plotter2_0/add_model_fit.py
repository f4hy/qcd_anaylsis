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

from  global_model2_0.global_fit_model2_0 import Model

from global_model2_0.fdssqrtms_models import * # noqa
from global_model2_0.single_heavy_fd_models import * # noqa
from global_model2_0.fd_models import * # noqa
from global_model2_0.pion_chiral_model import * # noqa
from global_model2_0.kaon_chiral_model import * # noqa
from global_model2_0.quark_mass_model import * # noqa

from itertools import cycle

import inspect

plotsettings = dict(linestyle="none", ms=12, elinewidth=4,
                    capsize=8, capthick=2, mew=2, c="k")

colorcycle = cycle(['k', 'c', 'g', 'y'])
styles = cycle(['--', '-.', ':', '-'])

meanplot = {}

#     addplot(plots, axe, fill, x=mDs_inv, y=y1, (label="fit $\\beta=4.17$",  ls="--", lw=2, color='b'))

def addplot(plots, axe, fill, save, x=None, y=None, params=None):

    if fill:
        axe.fill_between(x, meanplot[params["label"]], y, color=params["color"], alpha=0.01)
    else:
        if save:
            meanplot[params["label"]] = y
        plots.extend(axe.plot(x, y, **params))

def determine_eval_mode(modes, datanames):
    for m in modes:
        for name in datanames:
            if m.lower() in name.lower():
                logging.info("setting evaluation mode to {}".format(m))
                return m
    return None           # dont change if none match



def add_model_fit(axe, xran, boot_fit_file, options=None):
    c = colorcycle.next()
    lstyle = styles.next()
    plotsettings = dict(linestyle="none", ms=12, elinewidth=4,
                        capsize=8, capthick=2, mew=2, c=c)

    header = boot_fit_file.readline().split(",")
    name = header[0].strip("# ")
    columns = [s.strip("\n ,") for s in header[1:]]

    df = pd.read_csv(boot_fit_file, sep=",", delimiter=",", names=columns)

    valid_models = {m.__name__: m for m in Model.__subclasses__()}
    logging.debug("valid models available {}".format(valid_models.keys()))
    model = valid_models[name]

    m = model([], options)
    x = np.linspace(xran[0], xran[1], num=100)
    means = df.mean()
    params = [means[i] for i in m.contlim_args]
    np.seterr(divide='ignore', invalid='ignore')  # Ignore errors like this when plotting

    m.evalmode =  determine_eval_mode(m.evalmodes, options.ydata)


    logging.info("plotting line with params {}".format(params))
    y = m.eval_fit(x, *params)
    label = m.label
    if "cutoff" in boot_fit_file.name:
        try:
            cutoff = re.search("cutoff([0-9.]*[0-9])", boot_fit_file.name).group(1)
        except AttributeError:
            cutoff = np.nan
        if options.xdata in ["chiral_x", "xi"] and float(cutoff) < 600:
            label += ' ${}<{}$'.format("M_{\pi}", cutoff)
    plot_handles = axe.plot(x,y, color=c, lw=2, label=label, linestyle=lstyle)

    ys = []
    modelpoints = []
    for i,row in df.iterrows():
        p = [row[n] for n in m.contlim_args]
        ys.append(m.eval_fit(x,*p))
        if options.model_fit_point:
            modelpoints.append(m.eval_fit(options.model_fit_point,*p))

    ponesigma = np.percentile(ys, 84.1, axis=0)
    monesigma = np.percentile(ys, 15.9, axis=0)
    axe.fill_between(x, ponesigma, monesigma, alpha=0.05, color=c)

    if options.model_fit_point:
        point_x = options.model_fit_point
        point_y = m.eval_fit(point_x, *params)
        py_upper = np.percentile(modelpoints, 84.1, axis=0) - point_y
        py_lower = point_y- np.percentile(modelpoints, 15.9, axis=0)

        logging.info("At model point specified, value is {} + {} - {}".format(point_y, py_upper, py_lower))
        axe.errorbar(point_x,point_y, yerr=[[py_lower],[py_upper]], color=c, ms=15, elinewidth=4, mew=2, capthick=2, capsize=8)
        # withsys_err = [[np.sqrt((py_lower)**2 + (point_y*0.0168587)**2)],[np.sqrt((py_upper)**2 + (point_y*0.0168587)**2) ]]
        # axe.errorbar(point_x,point_y, yerr=withsys_err, color=c, ms=15, elinewidth=4, mew=2, capthick=2, capsize=8)

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
            ybeta = m.eval_fit(x, *finbeta_params)
            h = axe.plot(x,ybeta, color=colors[beta], lw=2, label="Fit at $\\beta$={}".format(beta))
            if options.model_fit_point:
                mfp = options.model_fit_point
                logging.info("Model point at {}: {}".format(mfp, m.eval_fit(mfp, *finbeta_params)))
            plot_handles.extend(h)
    except AttributeError as e:
        logging.warn("This model doesn't support plotting at finite beta")
    return plot_handles
