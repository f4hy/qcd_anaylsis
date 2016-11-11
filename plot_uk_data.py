#!/usr/bin/env python2
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import logging
import pandas as pd
import re
import numpy as np
from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from residualmasses import residual_mass
import glob
import os.path

from read_uk_data import uk_data_reader


colors = {"C": 'c', "M": "pink", "F":"purple"}

def add_uk_plot_data(axe, xdatatype, ydatatype, scaled=True):
    logging.info("plotting uk data")


    reader = uk_data_reader("/home/bfahy/data_Z2noise_runs/uk_data/UKQCD")

    added_plots = []
    xdata = reader.get_data(xdatatype)
    ydata = reader.get_data(ydatatype)

    scales = reader.get_data("ainv")

    scale = 1

    for xlabel,xd in xdata.iteritems():
        # if scaled:
        #     xd = xd * scales[xlabel.split("_")[0]]*1000
        for label,yd in [(k,v) for k,v in ydata.iteritems() if k.startswith(xlabel)]:
            # if scaled:
            #     yd = yd * scales[label.split("_")[0]]*1000
            mx = np.mean(xd)
            my = np.mean(yd)
            ex = np.std(xd)
            ey = np.std(yd)
            color = colors[label[0]]
            logging.info("ploting {}: {}, {}".format(label, mx, my))
            plotsettings = dict(linestyle="none", c=color, marker='s',
                                label=label, ms=15, elinewidth=4,
                                capsize=8, capthick=2, mec=color, mew=2,
                                aa=True, fmt='o', ecolor=color)

            plot = axe.errorbar(x=mx, y=my, yerr=ey, **plotsettings)
        added_plots.append(plot)

    logging.info("done plotting uk data")
    return added_plots
