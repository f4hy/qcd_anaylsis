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

from plot_helpers import print_paren_error

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi
from data_params import Zs, Zv

from auto_key import auto_key

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
    m = model([], {})
    x = np.linspace(xran[0], xran[1], num=100)
    means = df.mean()
    params = [means[i] for i in m.contlim_args]

    logging.info("plotting line with params {}".format(params))
    y = m.m(x, *params)
    p = axe.plot(x,y)

    ys = []
    for i,row in df.iterrows():
        p = [row[n] for n in m.contlim_args]
        ys.append(m.m(x,*p))

    ponesigma = np.percentile(ys, 84.1, axis=0)
    monesigma = np.percentile(ys, 15.9, axis=0)
    axe.fill_between(x, ponesigma, monesigma, alpha=0.1)

    return p
