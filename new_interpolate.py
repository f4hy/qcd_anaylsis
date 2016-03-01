#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass, residual_mass_errors

from ensamble_info import data_params, read_fit_mass, scale, phys_pion, phys_kaon, phys_Fpi
from ensamble_info import Zs, Zv

from ensemble_data import ensemble_data, MissingData

import inspect
import collections

from get_data import get_data

def read_files(files, fitdata, cutoff=None, hqm_cutoff=None):
    data = collections.OrderedDict()

    for f in files:
        logging.info("reading file {}".format(f))
        if "32x64x12" in f and "0.0035" in f:
            logging.warn("skipping file {}".format(f))
            continue

        dp = data_params(f)

        ed = ensemble_data(dp)

        if cutoff:
            if np.mean(ed.pion_mass(scaled=True).mean()) > (cutoff):
                continue
        if hqm_cutoff:
            if dp.heavyq_mass > hqm_cutoff:
                logging.info("dp {} has hqm {} > {}".format(dp,dp.heavyq_mass, hqm_cutoff))
                continue


        data[dp] = ed

    return data


class Model(object):

    def __init__(self, data, type_string):

        self.data = data


        self.type_string = type_string

        dps = self.data.keys()

        def safe_array(d):
            try:
                return np.array(d)
            except MissingData:
                return np.array(float("NAN"))


        def make_array(funname, **params):
            try:
                return np.array([getattr(data[dp], funname)(**params)
                                 for dp in dps])
            except MissingData:
                logging.warning("Missing {} data".format(funname))
                return None


        self.a = np.array([dp.latspacing for dp in dps])
        self.qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass) for dp in dps])
        self.renorm_qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass)/Zs[dp.beta] for
                                      dp in dps])
        self.res_err = np.array([data[dp].scale*residual_mass_errors(dp) for dp in dps])

        self.heavyq_mass = np.array([data[dp].scale*(dp.heavyq_mass) for dp in dps])

        self.m1 = np.array([dp.heavy_m1*data[dp].scale for dp in dps])
        self.m2 = np.array([dp.heavy_m2*data[dp].scale for dp in dps])

        self.mpisqr = make_array("pion_mass", scaled=True)**2

        self.mKsqr = make_array("kaon_mass", scaled=True)**2

        self.mD = make_array("D_mass", scaled=True)
        self.mDA = make_array("D_mass_axial", scaled=True)

        self.mDs = make_array("Ds_mass", scaled=True)

        self.mHH = make_array("HH_mass", scaled=True)

        self.fpi = make_array("fpi", scaled=True)

        self.fD = make_array("fD", scaled=True)

        self.fDA = make_array("fDA", scaled=True)

        self.fDA_div = make_array("fDA", scaled=True, renorm=True, div=True)
        self.mD_div = make_array("D_mass_div", scaled=True)

        self.fDs = make_array("fDs", scaled=True)


    def build_function(self):

        LAMBDA4_GUESS = 1100.0
        LAMBDA3_GUESS = 600.0

        B_GUESS = 3000.69
        c3_GUESS = 4.0
        c4_GUESS = 1.0
        F_0_GUESS = 120.0

        #colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1

        def paramdict(parameter, guess, err, limits=None, fix=False, fixzero=False):

            paramdict = {parameter: guess}
            paramdict["error_"+parameter] = err
            paramdict["fix_"+parameter] = fix
            if fixzero:
                paramdict[parameter] = 0.0
                paramdict["fix_"+parameter] = True
            if limits:
                paramdict["limit_"+parameter] = limits
            return paramdict

        if self.type_string == "mpisqrbymq_const":
            params = paramdict("B", 2000.0, 100.0)
            fun = self.mpisqrbymq_const

        elif self.type_string == "mpisqrbymq_xi_NLO":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("c3", c3_GUESS, c3_GUESS/10))
            fun = self.mpisqrbymq_xi_NLO

        elif self.type_string == "mpisqrbymq_xi_NLO_inverse":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            fun = self.mpisqrbymq_xi_NLO_inverse


        elif self.type_string == "combined_XI_NNLO":
            # """F_0, B, c3, c4, beta, ellphys"""
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0))
            params.update(paramdict("ellphys", -32.0, 4.3))
            params.update(paramdict("c3", c3_GUESS, c3_GUESS/10))
            params.update(paramdict("c4", c4_GUESS, c4_GUESS/10))
            params.update(paramdict("beta", 1.0, 1.0))
            params.update(paramdict("alpha", 1.0, 1.0))
            fun = self.combined_XI_NNLO


        elif self.type_string == "mpisqrbymq_x_NLO":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("F_0", 118.038, 4.30, fix=True))
            fun = self.mpisqrbymq_x_NLO

        elif self.type_string == "FPI_x_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("B", 2826.1, 68.66, fix=True))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_x_NLO_only

        elif self.type_string == "FPI_XI_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("c4", LAMBDA4_GUESS, LAMBDA4_GUESS/10))
            fun = self.FPI_XI_NLO_only

        elif self.type_string == "FPI_XI_NNLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("ellphys", -32.0, 4.3))
            params.update(paramdict("c4", np.mean(self.fpi), np.mean(self.fpi)/10))
            params.update(paramdict("beta", 8000.0, 8000.0/100, limits=(0, None)))
            fun = self.FPI_XI_NNLO_only


        elif self.type_string == "FPI_XI_NLO_inverse_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_XI_NLO_inverse_only

        elif self.type_string == "FPI_XI_NLO_inverse_phys":
            params = paramdict("F_P", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_XI_NLO_inverse_phys



        elif self.type_string == "combined_x_NLO":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0, limits=(0, None))
            params.update(paramdict("B", 2826.1, 68.66))
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.combined_x_NLO_only

        elif self.type_string == "combined_x_NLO_all":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0, limits=(0, None))
            params.update(paramdict("B", 2826.1, 68.66))
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))

            fun = self.combined_x_NLO_all


        elif self.type_string == "combined_x_NNLO":
            #colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            # params.update(paramdict("Lambda12", 3.4517, 0.1, limits=(0, None), fix=True))
            params.update(paramdict("km", 1.0, 0.01))
            params.update(paramdict("kf", 1.0, 0.01))
            fun = self.combined_x_NNLO_only

        elif self.type_string == "combined_x_NNLO_all":
            #colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            # params.update(paramdict("Lambda12", 3.4517, 0.1, limits=(0, None), fix=True))
            params.update(paramdict("km", 1.0, 0.01))
            params.update(paramdict("kf", 1.0, 0.01))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))
            fun = self.combined_x_NNLO_all

        elif self.type_string == "combined_x_NNLO_fixa0":
            #colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            # params.update(paramdict("Lambda12", 3.4517, 0.1, limits=(0, None), fix=True))
            params.update(paramdict("km", 1.0, 0.01))
            params.update(paramdict("kf", 1.0, 0.01))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))
            fun = self.combined_x_NNLO_fixa0



        elif self.type_string == "combined_XI_inverse_NNLO":
            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            l1, l2 = -0.4, 4.3
            l12_guess = (7.0*l1+8.0*l2)/15.0
            params.update(paramdict("l12", l12_guess, l12_guess*0.1, fix=True))
            params.update(paramdict("cm", 3.0, 0.01))
            params.update(paramdict("cf", 6.0, 0.01))
            fun = self.combined_XI_inverse_NNLO

        elif self.type_string == "combined_XI_inverse_NNLO_phys":
            params = paramdict("F_P", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            l1, l2 = 0.4, 4.3
            l12_guess = (7.0*l1+8.0*l2)/15.0
            params.update(paramdict("l12", l12_guess, l12_guess*0.1, fix=True))
            params.update(paramdict("cm", 3.0, 0.01))
            params.update(paramdict("cf", 6.0, 0.01))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))
            fun = self.combined_XI_inverse_NNLO_phys


        elif self.type_string == "combined_XI_inverse_NNLO_all":
            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            l1, l2 = 0.4, 4.3
            l12_guess = (7.0*l1+8.0*l2)/15.0
            #params.update(paramdict("l12", 3.0, 0.3, fix=True))
            params.update(paramdict("l12", l12_guess, l12_guess*0.1, fix=True))
            params.update(paramdict("cm", 3.0, 0.01))
            params.update(paramdict("cf", 6.0, 0.01))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))

            fun = self.combined_XI_inverse_NNLO_all

        elif self.type_string == "fD_chiral":
            params = paramdict("f_D0", 210, 10)
            params.update(paramdict("g", 0.59, 0.071, fix=True))
            params.update(paramdict("mu", 10.0, 100.0, limits=(0, None)))
            params.update(paramdict("c1", 0.0, 0.01))
            fun = self.fD_chiral

        elif self.type_string == "fDsbyfD_chiral":
            params = paramdict("mu", 100.0, 10.0, limits=(0, None))
            params.update(paramdict("k", 1.0, 0.1, limits=(0, None)))
            params.update(paramdict("f", 114.640, 6.26, fix=True))
            params.update(paramdict("c1", 0.0, 0.01))
            fun = self.fDsbyfD_chiral


        elif self.type_string == "MD_linear_mpisqr_asqr_mss":
            params = paramdict("MDphys", np.mean(self.mD), np.mean(self.mD_var), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.MD_linear_mpisqr_asqr_mss

        elif self.type_string == "MDs_linear_mpisqr_asqr_mss":
            params = paramdict("MDsphys", np.mean(self.mDs), np.mean(self.mDs_var), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.MDs_linear_mpisqr_asqr_mss

        elif self.type_string == "FD_linear_mpisqr_asqr_mss":
            params = paramdict("FDphys", np.mean(self.fD), np.mean(self.fD_var), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.FD_linear_mpisqr_asqr_mss

        elif self.type_string == "FDs_linear_mpisqr_asqr_mss":
            params = paramdict("FDsphys", np.mean(self.fDs), np.mean(self.fDs_var), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.FDs_linear_mpisqr_asqr_mss

        elif self.type_string == "FDsbyFD_linear_mpisqr_asqr_mss":
            params = paramdict("FDsbyFDphys", 1.2, 0.1, limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.FDsbyFD_linear_mpisqr_asqr_mss

        elif self.type_string == "Mhs_minus_Mhh":
            M_Bs_guess = 5366.79
            params = paramdict("M_Bs", M_Bs_guess, M_Bs_guess/100.0, limits=(0, None))

            params.update(paramdict("alpha", 0.0, 0.1))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.Mhs_minus_Mhh

        elif self.type_string == "quad_Mhs_minus_Mhh":
            M_Bs_guess = 5366.79
            params = paramdict("M_Bs", M_Bs_guess, M_Bs_guess/100.0, limits=(0, None))

            params.update(paramdict("alpha", 0.0, 0.1))
            params.update(paramdict("beta", 100.0, 0.1))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.quad_Mhs_minus_Mhh


        elif self.type_string == "fdsqrtm":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = -1.0
            gamma_guess = -0.001
            eta_guess = 1.0
            mu_guess = -1.0
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))


            fun = self.fdsqrtm

        elif self.type_string == "fdsqrtm_HQET":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = -1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -1.0
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            fun = self.fdsqrtm_HQET

        elif self.type_string == "fdsqrtm_dmss_HQET":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = -1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -1.0
            delta_S = -0.1
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdsqrtm_dmss_HQET

        else:
            logging.error("Function not supported yet")
            raise RuntimeError("Function {} not supported yet".format(self.type_string))

        return params, fun

    def MD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, MDphys):
        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* MDphys*(1.0+b*(self.mpisqr-phys_pion**2))

        data = self.mD.mean(1)
        var = self.mD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def MDs_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, MDsphys):
        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* MDsphys*(1.0+b*(self.mpisqr-phys_pion**2))

        data = self.mDs.mean(1)
        var = self.mDs.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def FD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDphys):
        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDphys*(1.0+b*(self.mpisqr-phys_pion**2))

        data = self.fD.mean(1)
        var = self.fD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def FDs_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDsphys):
        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDsphys*(1.0+b*(self.mpisqr-phys_pion**2))

        data = self.fDs.mean(1)
        var = self.fDs.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def FDsbyFD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDsbyFDphys):
        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDsbyFDphys*(1.0+b*(self.mpisqr-phys_pion**2))

        div = self.fDs/self.fD
        data = div.mean(1)
        var = div.var(1)

        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)



    def fD_chiral(self, f_D0, g, mu, c1):

        factor = 3.0*(1+3.0*g**2) / 4.0
        F = 114.64
        arg = self.mpisqr.mean(1) / mu**2
        M = f_D0*(1.0 -  factor*(self.mpisqr.mean(1)/(8*(np.pi**2)*(F**2)))*np.log(arg) + c1*self.mpisqr.mean(1)   )

        data = self.fD.mean(1)
        var = self.fD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def fDsbyfD_chiral(self, k, mu, c1, f):

        arg = self.mpisqr.mean(1) / mu**2
        M = (1.0 +  k*(self.mpisqr.mean(1)/(8*(np.pi**2)*(f**2)))*np.log(arg) + c1*self.mpisqr.mean(1)   )
        div = self.fDs/self.fD
        data = div.mean(1)
        var = div.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def mpisqrbymq_const(self, B):

        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        M = 2*B
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NLO(self, B, c3):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        M = 2*B*(1.0+0.5*self.xi*np.log(self.xi) ) + c3*self.xi
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NLO_inverse(self, B, Lambda3):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        arg = Lambda3**2 / self.mpisqr.mean(1)

        M = 2*B/(1.0+0.5*self.xi*np.log(arg) )
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NNLO(self, B, c3, c4, beta, ellphys):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi.mean(1)
        M = 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                 (c4/F_0 - 1.0/3.0 *(ellphys+16) )*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)


    def mpisqrbymq_x_NLO(self, B, F_0, Lambda3):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))

        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        arg1 = (Lambda3**2)/Msqr
        M = 2*B*(1.0-0.5*x*np.log(arg1))
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def combined_x_NLO_only(self, F_0, B, Lambda3, Lambda4):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        arg2 = (Lambda3**2)/Msqr

        data = self.mpisqr.mean(1) / self.renorm_qmass
        mpierr = self.mpisqr.std()
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


        M1 = 2*B*(1.0-0.5*x*np.log(arg2))

        M2 = F_0 * (1 + x*np.log(arg1))
        sqr_diff1 = (self.fpi - M2)**2
        sqr_diff2 = (data - M1)**2


        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)

    def combined_x_NLO_all(self, F_0, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        arg2 = (Lambda3**2)/Msqr

        data = self.mpisqr.mean(1) / self.renorm_qmass
        mpierr = self.mpisqr.std()
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        M1 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B*(1.0-0.5*x*np.log(arg2))

        M2 = (1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 * (1 + x*np.log(arg1))
        sqr_diff1 = (self.fpi - M2)**2
        sqr_diff2 = (data - M1)**2


        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)


    # def combined_x_NNLO_only(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf):
    #     Msqr = B*(self.renorm_qmass+self.renorm_qmass)
    #     x = Msqr/(8*(np.pi**2)*(F_0**2))
    #     arg4 = (Lambda4**2)/Msqr
    #     arg3 = (Lambda3**2)/Msqr

    #     arg12 = (Lambda12**2)/Msqr


    #     lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
    #     lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

    #     data = self.mpisqr / self.renorm_qmass
    #     mpierr = self.mpisqr.std()
    #     var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


    #     M1 = 2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

    #     M2 = F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2+kf*x**2)

    #     sqr_diff1 = (data - M1)**2
    #     sqr_diff2 = (self.fpi - M2)**2


    #     return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)

    def combined_x_NNLO_only(self, F_0, B, Lambda3, Lambda4, km, kf):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        # arg12 = (Lambda12**2)/Msqr

        l1 = -0.4
        l2 = 4.3
        #colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1


        Lambda1sqr = (phys_pion**2)*np.exp(l1)
        Lambda2sqr = (phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.mpisqr.mean(1) / self.renorm_qmass
        mpierr = self.mpisqr.std()
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


        M1 = 2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.fpi - M2)**2


        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)


    def combined_x_NNLO_all(self, F_0, B, Lambda3, Lambda4, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        # arg12 = (Lambda12**2)/Msqr

        l1 = -0.4
        l2 = 4.3
        #colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1


        Lambda1sqr = (phys_pion**2)*np.exp(l1)
        Lambda2sqr = (phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.mpisqr.mean(1) / self.renorm_qmass
        mpierr = self.mpisqr.std()
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.mKsqr) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        M1 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = (1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.fpi - M2)**2


        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)

    def combined_x_NNLO_fixa0(self, F_0, B, Lambda3, Lambda4, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        # arg12 = (Lambda12**2)/Msqr

        l1 = -0.4
        l2 = 4.3
        #colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1


        Lambda1sqr = (phys_pion**2)*np.exp(l1)
        Lambda2sqr = (phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.mpisqr.mean(1) / self.renorm_qmass
        mpierr = self.mpisqr.std()
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        denom1 = (1-gamma_1*((0.05)**2))
        denom2 = (1+gamma_2*((0.05)**2))

        M1 = ((1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)/denom1)*2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = ((1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)/denom2)*F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.fpi - M2)**2


        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)



    # def combined_x_NNLO_only(self, F_0, B, l1, l2, l3, l4, km, kf):
    #     Msqr = B*(self.renorm_qmass+self.renorm_qmass)
    #     x = Msqr/(8*(np.pi**2)*(F_0**2))
    #     # arg4 = (Lambda4**2)/Msqr
    #     # arg3 = (Lambda3**2)/Msqr

    #     # arg12 = (Lambda12**2)/Msqr
    #     # l1 = -0.4
    #     # l2 = 4.3
    #     #colangelo
    #     # l1 = -0.4 \pm 0.6
    #     # l2 = 4.3 \pm 0.1

    #     lm = 1.0/51.0 * (28*l1 + 32*l2 - 9.0*l3+49.0)
    #     lf = 1.0/30.0 * (14*l1 + 16*l2 + 6.0*l3-6.0*l4+23.0)

    #     data = self.mpisqr / self.renorm_qmass
    #     mpierr = self.mpisqr.std()
    #     var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


    #     M1 = 2*B*(1.0-0.5*x*l3+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

    #     M2 = F_0 * (1.0 + x*l4-5.0/4.0*(x**2)*(lf)**2+kf*x**2)

    #     sqr_diff1 = (data - M1)**2
    #     sqr_diff2 = (self.fpi - M2)**2


    #     return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)



    def combined_XI_NNLO(self, F_0, B, c3, c4, alpha, beta, ellphys):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi
        M1 = F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi) ) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2
        M2 = 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                  (c4/F_0 - 1.0/3.0 *(ellphys+16) )*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2
        sqr_diff1 = (self.fpi - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)


    def combined_XI_inverse_NNLO(self, F_0, B, Lambda3, Lambda4, l12, cm, cf):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi

        mpisqr = self.mpisqr.mean(1)
        arg3 = (Lambda3**2)/mpisqr
        arg4 = (Lambda4**2)/mpisqr

        # arg12 = (Lambda12**2)/Msqr

        lambda12sqr = (phys_pion**2)*np.exp(l12)

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1

        # l12 = (7.0*l1+8.0*l2)/15.0
        # print l12

        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)



        arg12 = lambda12sqr/mpisqr



        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        M1 = F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        M2 = 2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2) )

        sqr_diff1 = (self.fpi - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)



    def combined_XI_inverse_NNLO_all(self, F_0, B, Lambda3, Lambda4, l12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi

        mpisqr = self.mpisqr.mean(1)
        arg3 = (Lambda3**2)/mpisqr.mean(1)
        arg4 = (Lambda4**2)/mpisqr.mean(1)

        # arg12 = (Lambda12**2)/Msqr

        lambda12sqr = (phys_pion**2)*np.exp(l12)

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1

        # l12 = (7.0*l1+8.0*l2)/15.0
        # print l12

        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)



        arg12 = lambda12sqr/mpisqr

        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        M1 = (1-gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        M2 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2) )

        sqr_diff1 = (self.fpi - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)

    def combined_XI_inverse_NNLO_phys(self, F_P, B, Lambda3, Lambda4, l12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi

        mpisqr = self.mpisqr.mean(1)
        arg3 = (Lambda3**2)/mpisqr
        arg4 = (Lambda4**2)/mpisqr

        # arg12 = (Lambda12**2)/Msqr

        lambda12sqr = (phys_pion**2)*np.exp(l12)

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1

        # l12 = (7.0*l1+8.0*l2)/15.0
        # print l12

        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)



        arg12 = lambda12sqr/mpisqr

        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        xiphys = (phys_pion**2) / (8*np.pi**2 * (phys_Fpi**2))
        arg12phys = lambda12sqr/(phys_pion**2)
        arg3phys = (Lambda3**2)/(phys_pion**2)
        arg4phys = (Lambda4**2)/(phys_pion**2)

        lnOmegaMphys = 1.0/15.0 * (60.0*np.log(arg12phys) - 33.0*np.log(arg3phys) - 12.0*np.log(arg4phys)+52.0)
        lnOmegaFphys = 1.0/3.0 * (-15.0*np.log(arg12phys) + 18.0*np.log(arg4phys) - 29.0/2.0)

        F_0 = F_P * (1.0 - xiphys*np.log(arg4phys) - 1.0/4.0*(xi*lnOmegaFphys)**2 + cf*(xiphys**2))

        M1 = (1-gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        M2 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2) )

        sqr_diff1 = (self.fpi - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)


    def FPI_x_NLO_only(self, F_0, B, Lambda4):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_x_NNLO_only(self, F_0, B, Lambda4, k_f, LambdaF):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(4*np.pi*F_0)**2
        arg1 = (Lambda4**2)/Msqr
        arg2 = (LambdaF**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1) - (5.0/4.0)*(x**2)*(np.log(arg2))**2 + k_f*x**2)
        sqr_diff = (self.fpi/np.sqrt(2) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NLO_only(self, F_0, c4):
        M = F_0 * (1 - self.xi*np.log(self.xi) ) + c4*self.xi
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NNLO_only(self, F_0, c4, beta, ellphys):
        xi = self.xi
        xilnxi = xi*np.log(xi)
        M = F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi) ) + c4*xi*(1-5*xi*np.log(xi))  + beta*xi**2
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def FPI_XI_NLO_inverse_only(self, F_0, Lambda4):
        arg = self.mpisqr.mean(1)/(Lambda4**2)
        M = F_0 / (1 + self.xi*np.log(arg))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NLO_inverse_phys(self, F_P, Lambda4):
        arg = self.mpisqr.mean(1)/(Lambda4**2)
        xiphys = (phys_pion**2) / (8*np.pi**2 * (phys_Fpi**2))
        argphys = (phys_pion**2)/(Lambda4**2)
        F_0 = F_P * (1 + xiphys*np.log(argphys))
        M = F_0 / (1 + self.xi*np.log(arg))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def FPI_XI_NNLO_inverse_only(self, F_0, Lambda4, Omega_F, cF):
        arg1 = self.mpisqr.mean(1)/(Lambda4**2)
        arg2 = self.mpisqr.mean(1)/(Omega_F**2)
        XIs = self.xi
        M = F_0 / (1 + XIs*np.log(arg1) - (1.0/4.0)*(XIs*np.log(arg2))**2 - cF*(XIs**2))
        sqr_diff = (self.fpi - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def Mhs_minus_Mhh(self, M_Bs, alpha, gamma_1, gamma_s1):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi

        mpisqr = self.mpisqr.mean(1)


        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        M1 = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)*( M_Bs + alpha*(1.0/self.mHH) )



        Mhs_Mhh = self.mDs - (self.mHH)/2.0

        var = self.mDs_std + self.mHH_std

        sqr_diff1 = (Mhs_Mhh - M1)**2
        return np.sum(sqr_diff1/var)

    def quad_Mhs_minus_Mhh(self, M_Bs, alpha, beta, gamma_1, gamma_s1):
        mpierr = self.mpisqr.std()
        data = self.mpisqr.mean(1) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.xi

        mpisqr = self.mpisqr.mean(1)


        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        M1 = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)*( M_Bs + alpha*(1.0/self.mHH) + beta*(1.0/self.mHH)**2 )



        Mhs_Mhh = self.mDs - (self.mHH)/2.0

        var = self.mDs_std + self.mHH_std

        sqr_diff1 = (Mhs_Mhh - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm(self, Fsqrtm_inf, C1, C2, gamma, eta, mu):

        fdsqrm_data = self.fDA * np.sqrt(self.mDA)
        data = fdsqrm_data.mean(1)
        var = fdsqrm_data.var(1)

        m = self.mD.mean(1)

        # M1 = Fsqrtm_inf*( 1.0 + C1 / m + C2 / (m**2) + gamma *(m*self.a)**2 + eta*m*self.a*2 + mu*self.a**2)
        M1 = Fsqrtm_inf*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)


        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = fdsqrm_data.mean(1)
        var = fdsqrm_data.var(1)

        m = self.mD.mean(1) + self.m2 - self.m1

        M1 = Fsqrtm_inf*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_dmss_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, delta_S):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = fdsqrm_data.mean(1)
        var = fdsqrm_data.var(1)

        m = self.mD.mean(1) + self.m2 - self.m1

        Mss = (2.0*self.mKsqr.mean(1)) - self.mpisqr.mean(1)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)



def interpolate(data, model_str):

    logging.info("Fitting data")

    params, model = Model(data, model_str).build_function()

    ARGS = inspect.getargspec(model).args[1:]
    print params
    fixed_parms = [p for p in params if "fix" in p and params[p]]
    Nfree_params = len(ARGS) - len(fixed_parms)
    if model_str.startswith("combined"):
        dof = float(len(data)*2-Nfree_params)
    else:
        dof = float(len(data)-Nfree_params)
    # if "all" in model_str:
    #     dof = dof + 4

    logging.info("DOF {}".format(dof))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    m = Minuit(model, errordef=dof, print_level=0, pedantic=True, **params)
    m.set_strategy(2)
    results = m.migrad()

    logging.debug(results)

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(m.fval, dof, m.fval/dof))
    logging.info('covariance {}'.format(m.covariance))
    logging.info('fitted values {}'.format(m.values))
    logging.info('fitted errors {}'.format(m.errors))


    if not m.get_fmin().is_valid:
        logging.error("NOT VALID")
        exit(-1)

    return m


def write_data(fit_parameters, output_stub, suffix, model):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        chisqrbydof = fit_parameters.fval / fit_parameters.errordef
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fit_parameters.fval,
                                                                    fit_parameters.errordef,
                                                                    chisqrbydof))

        for name in fit_parameters.values.keys():
            ofile.write("{}, {} +/- {}\n".format(name, fit_parameters.values[name],
                                                 fit_parameters.errors[name]))


def interpolate_chiral_spacing(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    alldata = read_files(options.files, options.fitdata, cutoff=options.cutoff, hqm_cutoff=options.hqm_cutoff)

    fit_paramsters = interpolate(alldata, options.model)

    write_data(fit_paramsters, options.output_stub, ".fit", options.model)


if __name__ == "__main__":

    models = ["chiral_NLO_only", "chiral_NNLO_only", "chiral_NLO_all", "chiral_NNLOa_ll", "s_a_pi",
              "MPI_XI_NLO_only", "FPI_x_NLO_only", "FPI_XI_NLO_only", "FPI_XI_NLO_phys", "FPI_XI_NNLO_only",
              "FPI_XI_NLO_inverse_only", "FPI_XI_NLO_inverse_phys", "FPI_XI_NNLO_inverse_only", "mpisqrbymq_const",
              "mpisqrbymq_xi_NLO", "mpisqrbymq_xi_NLO_inverse", "mpisqrbymq_x_NLO", "combined_x_NLO", "combined_XI_NLO",  "combined_XI_NNLO", "combined_x_NNLO",
              "combined_XI_inverse_NNLO", "combined_x_NLO_all", "combined_x_NNLO_all", "combined_x_NNLO_fixa0", "combined_XI_inverse_NNLO_all" , "combined_XI_inverse_NNLO_phys",
              "fD_chiral",  "fDsbyfD_chiral",
              "MD_linear_mpisqr_asqr_mss", "MDs_linear_mpisqr_asqr_mss", "FD_linear_mpisqr_asqr_mss", "FDs_linear_mpisqr_asqr_mss", "FDsbyFD_linear_mpisqr_asqr_mss", "Mhs_minus_Mhh", "quad_Mhs_minus_Mhh", "fdsqrtm", "fdsqrtm_HQET", "fdsqrtm_dmss_HQET"]

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
    parser.add_argument("--hqm_cutoff", required=False, type=float,
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
