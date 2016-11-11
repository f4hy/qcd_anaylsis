import logging
import argparse
import os
import numpy as np
from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon
from data_params import Zs, Zv
import matplotlib.pyplot as plt
from residualmasses import residual_mass, residual_mass_errors

from msbar_convert import get_matm
from alpha_s import get_Cmu_mbar

import inspect
import sys
import plot_light_masses
import plot_latparams
import plot_heavy_decays
import plot_heavy_masses
import plot_fdsqrtmd
import plot_fdssqrtmds
import plot_interpolated_fsqrtm

# function_list = inspect.getmembers(sys.modules["fitfunctions"], inspect.isclass)
# functions = {name: f for name, f in function_list}


class plot_data(object):

    def __init__(self, value, error, label=None, physical=None):
        self.value = value
        self.error = error
        self.label = label
        if label is None:
            self.label = ""
        self.physical = physical
        if physical is None:
            self.physical = {}


def package_heavies(datas):
    pass

def get_data(ed, data_type, options):

    # This is hackish but makes the whole thing work
    # We create a map of function names to their functions themselves
    # Mostly to break up these functions into many files
    modules_with_plot_fucntions = ("plot_latparams", # List of modules with the plot files
                                   "plot_light_masses",
                                   "plot_heavy_decays", "plot_heavy_masses",
                                   "plot_fdsqrtmd", "plot_fdssqrtmds", "plot_interpolated_fsqrtm")

    function_map = {}
    for m in modules_with_plot_fucntions:
        function_map.update(dict(inspect.getmembers(sys.modules[m], inspect.isfunction)))

    def dataindex():
        num = 0
        while num < 100:
            yield num
            num += 1

    if data_type in function_map:
        result = function_map[data_type](ed, options)
        # result will either be tuple and we make it a plot_data
        # or a dictionary where we make each element a plot_data
        if isinstance(result, tuple):
            return plot_data(*result)
        else:
            return {k: plot_data(*v) for k,v in result.iteritems()}
    else:
        raise RuntimeError("{} not supported as a data type yet".format(data_type))




    raise RuntimeError("{} not supported as a data type yet".format(data_type))
