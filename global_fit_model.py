#!/usr/bin/env python2
import logging
import numpy as np

from residualmasses import residual_mass, residual_mass_errors

from ensamble_info import data_params, read_fit_mass, scale, phys_pion, phys_kaon, phys_Fpi
from ensamble_info import Zs, Zv
from ensamble_info import phys_pionplus

from ensemble_data import ensemble_data, MissingData

from msbar_convert import get_matm

class Model(object):

    def __init__(self, data, type_string, options):

        self.data = data

        self.type_string = type_string

        self.options = options

        dps = self.data.keys()

        self.bootstrap = None

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

        logging.info("buidling data")

        self.a = np.array([dp.latspacing for dp in dps])

        self.qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass) for dp in dps])
        self.renorm_qmass = np.array([data[dp].scale*(residual_mass(dp)+dp.ud_mass)/Zs[dp.beta] for
                                      dp in dps])
        self.res_err = np.array([data[dp].scale*residual_mass_errors(dp) for dp in dps])

        self.heavyq_mass = np.array([data[dp].scale*(dp.heavyq_mass) / Zs[dp.beta] for dp in dps])
        self.heavyq_mass_next = np.array([data[dp].scale*(dp.heavyq_mass_next) / Zs[dp.beta] for dp in dps])

        self.rho_mu = np.array([get_matm(hqm,hqm) for hqm in self.heavyq_mass])
        self.rho_mu_next = np.array([get_matm(hqmn,hqmn) for hqmn in self.heavyq_mass_next])

        self.m1 = np.array([dp.heavy_m1*data[dp].scale for dp in dps])
        self.m2 = np.array([dp.heavy_m2*data[dp].scale for dp in dps])



        self.mpisqr = make_array("pion_mass", scaled=True)**2
        self.mpi = make_array("pion_mass", scaled=True)

        self.mKsqr = make_array("kaon_mass", scaled=True)**2

        self.mD= make_array("D_mass", scaled=True)
        self.mDA = make_array("D_mass_axial", scaled=True)

        self.mDs = make_array("Ds_mass", scaled=True)
        self.mDsA = make_array("Ds_mass_axial", scaled=True)

        self.mHH = make_array("HH_mass", scaled=True)

        self.fpi = make_array("fpi", scaled=True)
        self.xi = make_array("xi", scaled=True)

        self.fD = make_array("fD", scaled=True)
        self.fD_div = make_array("fD", scaled=True, renorm=True, div=True)
        self.fD_matched = make_array("fD", scaled=True, renorm=True, div=True, matched=True)

        self.fDs = make_array("fDs", scaled=True)
        self.fDs_div = make_array("fDs", scaled=True, renorm=True, div=True)
        self.fDs_matched = make_array("fDs", scaled=True, renorm=True, div=True, matched=True)


        self.fDA = make_array("fDA", scaled=True)
        self.fDsA = make_array("fDsA", scaled=True)

        self.fDA_div = make_array("fDA", scaled=True, renorm=True, div=True)
        self.fDsA_div = make_array("fDsA", scaled=True, renorm=True, div=True)
        self.mD_div = make_array("D_mass_div", scaled=True)
        self.mDs_div = make_array("Ds_mass_div", scaled=True)




        self.D_mass_ratio = make_array("D_mass_ratio", scaled=False)
        self.D_mass_div_ratio = make_array("D_mass_ratio", scaled=False, div=True, corrected=False)
        self.D_mass_div_cor_ratio = make_array("D_mass_ratio", scaled=False, div=True, corrected=True)
        self.fD_ratio = make_array("fD_ratio", scaled=False)
        self.fD_div_ratio = make_array("fD_ratio", scaled=False, renorm=True, div=True)
        self.fD_matched_ratio = make_array("fD_ratio", scaled=False, renorm=True, div=True, matched=True)

        self.Ds_mass_ratio = make_array("Ds_mass_ratio", scaled=False)
        self.Ds_mass_div_ratio = make_array("Ds_mass_ratio", scaled=False, div=True, corrected=False)
        self.Ds_mass_div_cor_ratio = make_array("Ds_mass_ratio", scaled=False, div=True, corrected=True)
        self.fDs_ratio = make_array("fDs_ratio", scaled=False)
        self.fDs_div_ratio = make_array("fDs_ratio", scaled=False, renorm=True, div=True)
        self.fDs_matched_ratio = make_array("fDs_ratio", scaled=False, renorm=True, div=True, matched=True)

        logging.info("Data read")


    def bstrapdata(self, d):
        if self.bootstrap is None or self.bootstrap == "mean":
            return d.mean(1)
        else:
            return d[:, self.bootstrap]

    def set_bootstrap(self, b):
        self.bootstrap = b

    def build_function(self):

        LAMBDA4_GUESS = 1100.0
        LAMBDA3_GUESS = 600.0

        B_GUESS = 3000.69
        c3_GUESS = 4.0
        c4_GUESS = 1.0
        F_0_GUESS = 130.0

        l12_GUESS = 4.0
        l3_GUESS = 3.0
        l4_GUESS = 4.0

        # colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1

        def paramdict(parameter, guess, err, limits=None, fix=False, fixzero=False):

            paramdict = {parameter: guess}
            paramdict["error_"+parameter] = err
            paramdict["fix_"+parameter] = fix
            if parameter in self.options.zero:
                logging.info("option passed to set {} to zero".format(parameter))
                logging.info("zero {self.options.zero}")
            if fixzero or parameter in self.options.zero:
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
            """F_0, B, c3, c4, beta, ellphys"""
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0))
            params.update(paramdict("ellphys", -32.0, 4.3, fix=True))
            params.update(paramdict("c3", c3_GUESS, c3_GUESS/10))
            params.update(paramdict("c4", c4_GUESS, c4_GUESS/10))
            params.update(paramdict("beta", 1.0, 1.0))
            params.update(paramdict("alpha", 1.0, 1.0))
            fun = self.combined_XI_NNLO

        elif self.type_string == "mpisqrbymq_x_NLO":
            params = paramdict("B", B_GUESS, 50)
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("F_0", 118.038, 4.30))
            fun = self.mpisqrbymq_x_NLO

        elif self.type_string == "FPI_x_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("B", 2826.1, 68.66))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10, limits=(0, None)))
            fun = self.FPI_x_NLO_only

        elif self.type_string == "FPI_XI_NLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("c4", LAMBDA4_GUESS, LAMBDA4_GUESS/10))
            fun = self.FPI_XI_NLO_only

        elif self.type_string == "FPI_XI_NNLO_only":
            params = paramdict("F_0", np.mean(self.fpi), np.mean(self.fpi)/10.0)
            params.update(paramdict("ellphys", -32.0, 4.3, fix=True))
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
            params.update(paramdict("B", 2826.1, 68.66, limits=(0, None)))
            params.update(paramdict("Lambda3", LAMBDA3_GUESS, LAMBDA3_GUESS/10.0, limits=(0, None)))
            params.update(paramdict("Lambda4", LAMBDA4_GUESS, LAMBDA4_GUESS/10.0, limits=(0, None)))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))

            fun = self.combined_x_NLO_all

        elif self.type_string == "combined_x_NNLO":
            # colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0, limits=(0, None)))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            params.update(paramdict("Lambda12", 20.0, 0.1, limits=(0, None), fix=True))
            params.update(paramdict("km", 1.0, 0.01))
            params.update(paramdict("kf", 1.0, 0.01))
            fun = self.combined_x_NNLO_only

        elif self.type_string == "combined_x_NNLO_LEC":

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/100.0, limits=(0, None)))

            params.update(paramdict("l3", 3.0, 1.0))
            params.update(paramdict("l4", 4.0, 1.0))
            params.update(paramdict("l12", 2.0, 2.0, fix=True))
            params.update(paramdict("km", 1.0, 2.0))
            params.update(paramdict("kf", 1.0, 2.0))
            fun = self.combined_x_NNLO_LEC

        elif self.type_string == "combined_x_NNLO_all":
            # colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            params.update(paramdict("Lambda12", 20.0, 0.1, limits=(0, None)))
            params.update(paramdict("km", 1.0, 0.01))
            params.update(paramdict("kf", 1.0, 0.01))

            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_2", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))
            params.update(paramdict("gamma_s2", 0.0, 0.1))
            fun = self.combined_x_NNLO_all

        elif self.type_string == "combined_x_NNLO_fixa0":
            # colangelo
            # l1 = -0.4 \pm 0.6
            # l2 = 4.3 \pm 0.1

            params = paramdict("F_0", F_0_GUESS, F_0_GUESS/100.0, limits=(0, None))
            params.update(paramdict("B", B_GUESS, B_GUESS/10.0))

            params.update(paramdict("Lambda3", 609.7, 146.2, limits=(0, None)))
            params.update(paramdict("Lambda4", 1169.7, 140.55, limits=(0, None)))
            params.update(paramdict("Lambda12", 20.0, 0.1, limits=(0, None)))
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
            l1, l2 = -0.4, 4.3
            l12_guess = (7.0*l1+8.0*l2)/15.0
            # params.update(paramdict("l12", 3.0, 0.3, fix=True))
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
            params = paramdict("MDphys", np.mean(self.mD.mean(1)), np.mean(self.mD.var(1)), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.MD_linear_mpisqr_asqr_mss

        elif self.type_string == "MDs_linear_mpisqr_asqr_mss":
            params = paramdict("MDsphys", np.mean(self.mDs.mean(1)), np.mean(self.mDs.var(1)), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.MDs_linear_mpisqr_asqr_mss

        elif self.type_string == "FD_linear_mpisqr_asqr_mss":
            params = paramdict("FDphys", np.mean(self.fD.mean(1)), np.mean(self.fD.std(1)), limits=(0, None))

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.0, 0.1))
            params.update(paramdict("gamma_s1", 0.0, 0.1))

            fun = self.FD_linear_mpisqr_asqr_mss

        elif self.type_string == "FDA_linear_mpisqr_asqr_mss":
            params = paramdict("FDphys", np.mean(self.fD.mean(1)), np.mean(self.fD.std(1))/10.0)

            params.update(paramdict("b", 0.0, 0.1))
            params.update(paramdict("gamma_1", 0.01, 0.1))
            params.update(paramdict("gamma_s1", 0.01, 0.1))

            fun = self.FDA_linear_mpisqr_asqr_mss

        elif self.type_string == "FDs_linear_mpisqr_asqr_mss":
            params = paramdict("FDsphys", np.mean(self.fDs.mean(1)), np.mean(self.fDs.var(1)), limits=(0, None))

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

        elif self.type_string == "fdsqrtm_chiral":
            Fsqrtm_inf_guess = 28000.0
            C1_guess = -0.6
            C2_guess = -0.6
            gamma_guess = -0.07
            eta_guess = 1.0
            mu_guess = 28332.0
            b_guess = 0.0000004
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))
            params.update(paramdict("b", b_guess, b_guess/2))

            fun = self.fdsqrtm_chiral

        elif self.type_string == "fdsqrtm_chiral_dmss":
            Fsqrtm_inf_guess = 28000.0
            C1_guess = -0.6
            C2_guess = -0.6
            gamma_guess = -0.07
            eta_guess = 1.0
            mu_guess = 28332.0
            b_guess = 0.0000004
            delta_S = 1.0

            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))
            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdsqrtm_chiral_dmss

        elif self.type_string == "fdssqrtms_chiral_dmss":
            Fssqrtms_inf_guess = 28000.0
            C1_guess = -0.6
            C2_guess = -0.6
            gamma_guess = -0.07
            eta_guess = 1.0
            mu_guess = 28332.0
            b_guess = 0.0000004
            delta_S = 1.0

            params = paramdict("Fssqrtms_inf", Fssqrtms_inf_guess, Fssqrtms_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))
            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdssqrtms_chiral_dmss

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

        elif self.type_string == "fdsqrtm_chiral_HQET":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = -1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -1.0
            b_guess = 0.0000004
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))

            fun = self.fdsqrtm_chiral_HQET

        elif self.type_string == "fdsqrtm_chiral_dmss_HQET":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = 1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -9000.0
            b_guess = 0.0000004
            delta_S = 0.01
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdsqrtm_chiral_dmss_HQET

        elif self.type_string == "fdsqrtm_HQET_matched":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = 1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -9000.0
            b_guess = 0.0000004
            delta_S = 0.01
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdsqrtm_HQET_matched

        elif self.type_string == "fdsqrtm_HQET_matched_nom2":
            Fsqrtm_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = 1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -9000.0
            b_guess = 0.0000004
            delta_S = 0.01
            params = paramdict("Fsqrtm_inf", Fsqrtm_inf_guess, Fsqrtm_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdsqrtm_HQET_matched_nom2

        elif self.type_string == "fdssqrtms_HQET_matched_nom2":
            Fssqrtms_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = 1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -9000.0
            b_guess = 0.0000004
            delta_S = 0.01
            params = paramdict("Fssqrtms_inf", Fssqrtms_inf_guess, Fssqrtms_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdssqrtms_HQET_matched_nom2


        elif self.type_string == "fdssqrtms_chiral_dmss_HQET":
            Fssqrtms_inf_guess = 20000.0
            C1_guess = -1.0
            C2_guess = 1.0
            gamma_guess = 0.001
            eta_guess = -1.0
            mu_guess = -9000.0
            b_guess = 0.0000004
            delta_S = 0.01
            params = paramdict("Fssqrtms_inf", Fssqrtms_inf_guess, Fssqrtms_inf_guess/10.0, limits=(0, None))
            params.update(paramdict("C1", C1_guess, C1_guess/2))
            params.update(paramdict("C2", C2_guess, C2_guess/2))

            params.update(paramdict("gamma", gamma_guess, gamma_guess/2))
            params.update(paramdict("eta", eta_guess, eta_guess/2, fixzero=True))
            params.update(paramdict("mu", mu_guess, mu_guess/2))

            params.update(paramdict("b", b_guess, b_guess/2))
            params.update(paramdict("delta_S", delta_S, delta_S/2))

            fun = self.fdssqrtms_chiral_dmss_HQET

        elif self.type_string == "fdssqrtms_ratio":

            z_guess = 1.0
            z2_guess = 1.0
            gamma_guess = 1.0

            params = paramdict("z", z_guess, z_guess)
            params.update(paramdict("z2", z2_guess, z2_guess))
            params.update(paramdict("gamma_1", gamma_guess, gamma_guess))

            fun = self.fdssqrtms_ratio

        elif self.type_string == "fdssqrtms_mq_ratio":

            z_guess = 1.0
            z2_guess = 1.0
            gammaA_guess = 1.0
            gammaS_guess = 1.0
            gammaP_guess = 1.0

            params = paramdict("z", z_guess, z_guess)
            params.update(paramdict("z2", z2_guess, z2_guess))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            fun = self.fdssqrtms_mq_ratio

        elif self.type_string == "fdssqrtms_mq_ma_ratio":

            z_guess = 200.0
            z2_guess = -100000.0
            gammaA_guess = 11.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.1
            gammaS_guess = 2.0e-8
            gammaP_guess = 2.0e-8

            params = paramdict("z", z_guess, z_guess/2)
            params.update(paramdict("z2", z2_guess, z2_guess/2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.fdssqrtms_mq_ma_ratio

        elif self.type_string == "fdsqrtm_mq_ma_ratio":

            z_guess = 200.0
            z2_guess = -100000.0
            gammaA_guess = 11.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.1
            gammaS_guess = 2.0e-8
            gammaP_guess = 2.0e-8

            params = paramdict("z", z_guess, z_guess/2)
            params.update(paramdict("z2", z2_guess, z2_guess/2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.fdsqrtm_mq_ma_ratio

        elif self.type_string == "fdsqrtmd_matched_ratio":

            z_guess = 200.0
            z2_guess = -100000.0
            gammaA_guess = 11.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.1
            gammaS_guess = 2.0e-8
            gammaP_guess = 2.0e-8

            params = paramdict("z", z_guess, z_guess/2)
            params.update(paramdict("z2", z2_guess, z2_guess/2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.fdsqrtmd_matched_ratio


        elif self.type_string == "fdsqrtm_ratio":

            z_guess = 1.0
            z2_guess = 1.0
            gamma1_guess = 1.0

            params = paramdict("z", z_guess, z_guess)
            params.update(paramdict("z2", z2_guess, z2_guess))
            params.update(paramdict("gamma_1", gamma1_guess, gamma1_guess))

            fun = self.fdsqrtm_ratio

        elif self.type_string == "ms_mq_ma_ratio":

            z_guess = 200.0
            z2_guess = -100000.0
            gammaA_guess = 11.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.1
            gammaS_guess = 2.0e-8
            gammaP_guess = 2.0e-8

            params = paramdict("z", z_guess, z_guess/2)
            params.update(paramdict("z2", z2_guess, z2_guess/2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.ms_mq_ma_ratio

        elif self.type_string == "m_mq_ma_ratio":

            z_guess = 200.0
            z2_guess = -100000.0
            gammaA_guess = 11.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.1
            gammaS_guess = 2.0e-8
            gammaP_guess = 2.0e-8

            params = paramdict("z", z_guess, z_guess/2)
            params.update(paramdict("z2", z2_guess, z2_guess/2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.m_mq_ma_ratio

        elif self.type_string == "mD_corrected_pole_ratio":

            z_guess = 200.0
            z2_guess = 100.0
            gammaA_guess = 5.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.00001
            gammaS_guess = 2.0e-8
            gammaP_guess = -2.0e-8

            params = paramdict("z", z_guess, z_guess*2)
            params.update(paramdict("z2", z2_guess, z2_guess*2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.mD_corrected_pole_ratio


        elif self.type_string == "mDs_corrected_pole_ratio":

            z_guess = 200.0
            z2_guess = 100.0
            gammaA_guess = 5.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.00001
            gammaS_guess = 2.0e-8
            gammaP_guess = -2.0e-8

            params = paramdict("z", z_guess, z_guess*2)
            params.update(paramdict("z2", z2_guess, z2_guess*2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.mDs_corrected_pole_ratio

        elif self.type_string == "mD_pole_ratio":

            z_guess = 200.0
            z2_guess = 100.0
            gammaA_guess = 50.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.00001
            gammaS_guess = 2.0e-8
            gammaP_guess = -2.0e-8

            params = paramdict("z", z_guess, z_guess*2)
            params.update(paramdict("z2", z2_guess, z2_guess*2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.mD_pole_ratio


        elif self.type_string == "mDs_pole_ratio":

            z_guess = 200.0
            z2_guess = 100.0
            gammaA_guess = 50.0
            gammaMA_guess = 0.01
            gammaMMA_guess = 0.00001
            gammaS_guess = 2.0e-8
            gammaP_guess = -2.0e-8

            params = paramdict("z", z_guess, z_guess*2)
            params.update(paramdict("z2", z2_guess, z2_guess*2))
            params.update(paramdict("gamma_A", gammaA_guess, gammaA_guess))
            params.update(paramdict("gamma_S", gammaS_guess, gammaS_guess))
            params.update(paramdict("gamma_P", gammaP_guess, gammaP_guess))

            params.update(paramdict("gamma_MA", gammaMA_guess, gammaMA_guess))
            params.update(paramdict("gamma_MMA", gammaMMA_guess, gammaMMA_guess))

            fun = self.mDs_pole_ratio


        else:
            logging.error("Function not supported yet")
            raise RuntimeError("Function {} not supported yet".format(self.type_string))

        return params, fun

    def MD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, MDphys):
        Mss = (2.0*self.bstrapdata(self.mKsqr) - self.bstrapdata(self.mpisqr))
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* MDphys*(1.0+b*(self.bstrapdata(self.mpisqr)-phys_pion**2))

        data = self.bstrapdata(self.mD)
        var = self.mD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def MDs_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, MDsphys):
        Mss = (2.0*self.bstrapdata(self.mKsqr) - self.bstrapdata(self.mpisqr))
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* MDsphys*(1.0+b*(self.bstrapdata(self.mpisqr)-phys_pion**2))

        data = self.bstrapdata(self.mDs)
        var = self.mDs.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def FD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDphys):
        Mss = (2.0*self.bstrapdata(self.mKsqr) - self.bstrapdata(self.mpisqr))
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss

        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDphys*(1.0+b*(self.bstrapdata(self.mpisqr)-phys_pion**2))

        data = self.bstrapdata(self.fD)
        var = self.fD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def FDA_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDphys):
        Mss = (2.0*self.bstrapdata(self.mKsqr) - self.bstrapdata(self.mpisqr))
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss


        M = (1.0+gamma_1*(self.a**2))*(1.0+gamma_s1*delta_Mss)* FDphys*(1.0+b*(self.bstrapdata(self.mpisqr)-phys_pion**2))

        data = self.bstrapdata(self.fDA)
        var = self.fDA.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)


    def FDs_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDsphys):
        mpisqr = self.bstrapdata(self.mpisqr)
        Mss = (2.0*self.bstrapdata(self.mKsqr) - mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss

        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDsphys*(1.0+b*(mpisqr-phys_pion**2))


        data = self.bstrapdata(self.fDs)
        var = self.fDs.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)



    def FDsbyFD_linear_mpisqr_asqr_mss(self, b, gamma_1, gamma_s1, FDsbyFDphys):
        Mss = (2.0*self.bstrapdata(self.mKsqr) - self.bstrapdata(self.mpisqr))
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)
        delta_Mss = Mss - phys_Mss
        M = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)* FDsbyFDphys*(1.0+b*(self.bstrapdata(self.mpisqr)-phys_pion**2))

        div = self.fDs/self.fD
        data = self.bstrapdata(div)
        var = div.var(1)

        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)



    def fD_chiral(self, f_D0, g, mu, c1):

        factor = 3.0*(1+3.0*g**2) / 4.0
        F = 114.64
        arg = self.bstrapdata(self.mpisqr) / mu**2
        M = f_D0*(1.0 -  factor*(self.bstrapdata(self.mpisqr)/(8*(np.pi**2)*(F**2)))*np.log(arg) + c1*self.bstrapdata(self.mpisqr)   )

        data = self.bstrapdata(self.fD)
        var = self.fD.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def fDsbyfD_chiral(self, k, mu, c1, f):

        arg = self.bstrapdata(self.mpisqr) / mu**2
        M = (1.0 +  k*(self.bstrapdata(self.mpisqr)/(8*(np.pi**2)*(f**2)))*np.log(arg) + c1*self.bstrapdata(self.mpisqr)   )
        div = self.fDs/self.fD
        data = self.bstrapdata(div)
        var = div.var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def mpisqrbymq_const(self, B):

        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        M = 2*B
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NLO(self, B, c3):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)
        M = 2*B*(1.0+0.5*xi*np.log(xi) ) + c3*xi
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NLO_inverse(self, B, Lambda3):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)

        arg = Lambda3**2 / self.bstrapdata(self.mpisqr)

        M = 2*B/(1.0+0.5*xi*np.log(arg) )
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)

    def mpisqrbymq_xi_NNLO(self, B, c3, c4, beta, ellphys):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)
        M = 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                 (c4/F_0 - 1.0/3.0 *(ellphys+16) )*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff/var)


    def mpisqrbymq_x_NLO(self, B, F_0, Lambda3):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))

        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
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

        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        mpierr = self.mpisqr.std(1)
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


        M1 = 2*B*(1.0-0.5*x*np.log(arg2))

        M2 = F_0 * (1 + x*np.log(arg1))
        sqr_diff1 = (self.bstrapdata(self.fpi) - M2)**2
        sqr_diff2 = (data - M1)**2


        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)

    def combined_x_NLO_all(self, F_0, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        arg2 = (Lambda3**2)/Msqr

        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        mpierr = self.mpisqr.std(1)
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        M1 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B*(1.0-0.5*x*np.log(arg2))

        M2 = (1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 * (1 + x*np.log(arg1))
        sqr_diff1 = (self.bstrapdata(self.fpi) - M2)**2
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
    #     mpierr = self.mpisqr.std(1)
    #     var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


    #     M1 = 2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

    #     M2 = F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2+kf*x**2)

    #     sqr_diff1 = (data - M1)**2
    #     sqr_diff2 = (self.fpi - M2)**2


    #     return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)

    def combined_x_NNLO_only(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        arg12 = (Lambda12**2)/Msqr

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1


        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)

        # arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        mpierr = self.mpisqr.std(1)
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


        M1 = 2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.bstrapdata(self.fpi) - M2)**2


        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)

    def combined_x_NNLO_LEC(self, F_0, B, l3, l4, l12, km, kf):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))

        Lambda3 = np.sqrt((phys_pionplus**2)*np.exp(l3))
        Lambda4 = np.sqrt((phys_pionplus**2)*np.exp(l4))
        Lambda12 = np.sqrt((phys_pionplus**2)*np.exp(l12))

        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        arg12 = (Lambda12**2)/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        mpisqrq = np.array([self.mpisqr[i] / self.renorm_qmass[i] for i in range(len(self.renorm_qmass))])

        M1 = 2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*(x**2) )

        M2 = F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (self.bstrapdata(mpisqrq) - M1)**2
        sqr_diff2 = (self.bstrapdata(self.fpi) - M2)**2

        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/mpisqrq.var(1))



    def combined_x_NNLO_all(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        arg12 = (Lambda12**2)/Msqr

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1


        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)

        # arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        mpierr = self.mpisqr.std(1)
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        M1 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = (1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.bstrapdata(self.fpi) - M2)**2


        return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)

    def combined_x_NNLO_fixa0(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        arg12 = (Lambda12**2)/Msqr

        # l1 = -0.4
        # l2 = 4.3
        # #colangelo
        # # l1 = -0.4 \pm 0.6
        # # l2 = 4.3 \pm 0.1


        # Lambda1sqr = (phys_pion**2)*np.exp(l1)
        # Lambda2sqr = (phys_pion**2)*np.exp(l2)

        # lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        # lambda12sqr = np.exp(lnLambda12sqr)

        # arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        mpierr = self.mpisqr.std(1)
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        denom1 = (1-gamma_1*((0.05)**2))
        denom2 = (1+gamma_2*((0.05)**2))

        M1 = ((1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)/denom1)*2*B*(1.0-0.5*x*np.log(arg3)+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

        M2 = ((1+gamma_2*(self.a**2)+gamma_s2*delta_Mss)/denom2)*F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        sqr_diff1 = (data - M1)**2
        sqr_diff2 = (self.bstrapdata(self.fpi) - M2)**2


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
    #     mpierr = self.mpisqr.std(1)
    #     var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2


    #     M1 = 2*B*(1.0-0.5*x*l3+17.0/8.0*(x**2)*(lm)**2 + km*x**2 )

    #     M2 = F_0 * (1.0 + x*l4-5.0/4.0*(x**2)*(lf)**2+kf*x**2)

    #     sqr_diff1 = (data - M1)**2
    #     sqr_diff2 = (self.fpi - M2)**2


    #     return np.sum(sqr_diff2/self.fpi.var(1))+np.sum(sqr_diff1/var)



    def combined_XI_NNLO(self, F_0, B, c3, c4, alpha, beta, ellphys):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)
        M1 = F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi) ) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2
        M2 = 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                  (c4/F_0 - 1.0/3.0 *(ellphys+16) )*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2
        sqr_diff1 = (self.bstrapdata(self.fpi) - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)


    def combined_XI_inverse_NNLO(self, F_0, B, Lambda3, Lambda4, l12, cm, cf):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)

        mpisqr = self.bstrapdata(self.mpisqr)
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

        sqr_diff1 = (self.bstrapdata(self.fpi) - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)



    def combined_XI_inverse_NNLO_all(self, F_0, B, Lambda3, Lambda4, l12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2

        xi = self.bstrapdata(self.xi)

        mpisqr = self.bstrapdata(self.mpisqr)
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

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        M1 = (1-gamma_2*(self.a**2)+gamma_s2*delta_Mss)*F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        M2 = (1-gamma_1*(self.a**2)+gamma_s1*delta_Mss)*2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2) )

        sqr_diff1 = (self.bstrapdata(self.fpi) - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)

    def combined_XI_inverse_NNLO_phys(self, F_P, B, Lambda3, Lambda4, l12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)

        mpisqr = self.bstrapdata(self.mpisqr)
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

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
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

        sqr_diff1 = (self.bstrapdata(self.fpi) - M1)**2
        sqr_diff2 = (data - M2)**2
        return np.sum(sqr_diff1/self.fpi.var(1))+np.sum(sqr_diff2/var)


    def FPI_x_NLO_only(self, F_0, B, Lambda4):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg1 = (Lambda4**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1))
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_x_NNLO_only(self, F_0, B, Lambda4, k_f, LambdaF):
        Msqr = B*(self.renorm_qmass+self.renorm_qmass)
        x = Msqr/(4*np.pi*F_0)**2
        arg1 = (Lambda4**2)/Msqr
        arg2 = (LambdaF**2)/Msqr
        M = F_0 * (1 + x*np.log(arg1) - (5.0/4.0)*(x**2)*(np.log(arg2))**2 + k_f*x**2)
        sqr_diff = (self.bstrapdata(self.fpi)/np.sqrt(2) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NLO_only(self, F_0, c4):
        xi = self.bstrapdata(self.xi)
        M = F_0 * (1 - xi*np.log(xi) ) + c4*xi
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NNLO_only(self, F_0, c4, beta, ellphys):
        xi = self.bstrapdata(self.xi)
        xilnxi = xi*np.log(xi)
        M = F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi) ) + c4*xi*(1-5*xi*np.log(xi))  + beta*xi**2
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def FPI_XI_NLO_inverse_only(self, F_0, Lambda4):
        arg = self.bstrapdata(self.mpisqr)/(Lambda4**2)
        xi = self.bstrapdata(self.xi)
        M = F_0 / (1 + xi*np.log(arg))
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))

    def FPI_XI_NLO_inverse_phys(self, F_P, Lambda4):
        arg = self.bstrapdata(self.mpisqr)/(Lambda4**2)
        xiphys = (phys_pion**2) / (8*np.pi**2 * (phys_Fpi**2))
        argphys = (phys_pion**2)/(Lambda4**2)
        xi = self.bstrapdata(self.xi)
        F_0 = F_P * (1 + xiphys*np.log(argphys))
        M = F_0 / (1 + xi*np.log(arg))
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def FPI_XI_NNLO_inverse_only(self, F_0, Lambda4, Omega_F, cF):
        arg1 = self.bstrapdata(self.mpisqr)/(Lambda4**2)
        arg2 = self.bstrapdata(self.mpisqr)/(Omega_F**2)
        XIs = self.bstrapdata(self.xi)
        M = F_0 / (1 + XIs*np.log(arg1) - (1.0/4.0)*(XIs*np.log(arg2))**2 - cF*(XIs**2))
        sqr_diff = (self.bstrapdata(self.fpi) - M)**2
        return np.sum(sqr_diff/self.fpi.var(1))


    def Mhs_minus_Mhh(self, M_Bs, alpha, gamma_1, gamma_s1):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)

        mpisqr = self.bstrapdata(self.mpisqr)


        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        mHH = self.bstrapdata(self.mHH)
        M1 = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)*( M_Bs + alpha*(1.0/mHH) )



        Mhs_Mhh = (self.bstrapdata(self.mDs - (self.mHH)/2.0))

        var = (self.mDs + self.mHH).var(1)

        sqr_diff1 = (Mhs_Mhh - M1)**2
        return np.sum(sqr_diff1/var)

    def quad_Mhs_minus_Mhh(self, M_Bs, alpha, beta, gamma_1, gamma_s1):
        mpierr = self.mpisqr.std(1)
        data = self.bstrapdata(self.mpisqr) / self.renorm_qmass
        var = (mpierr/self.renorm_qmass)**2 + (self.res_err*data/(self.qmass))**2
        xi = self.bstrapdata(self.xi)

        mpisqr = self.bstrapdata(self.mpisqr)


        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss

        mHH = self.bstrapdata(self.mHH)
        M1 = (1+gamma_1*(self.a**2)+gamma_s1*delta_Mss)*( M_Bs + alpha*(1.0/mHH) + beta*(1.0/mHH)**2 )

        Mhs_Mhh = self.mDs - (self.mHH)/2.0
        data = self.bstrapdata(Mhs_Mhh)
        var = Mhs_Mhh.var(1)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm(self, Fsqrtm_inf, C1, C2, gamma, eta, mu):

        fdsqrm_data = self.fDA * np.sqrt(self.mDA)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD)

        # M1 = Fsqrtm_inf*( 1.0 + C1 / m + C2 / (m**2) + gamma *(m*self.a)**2 + eta*m*self.a*2 + mu*self.a**2)
        M1 = Fsqrtm_inf*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)


        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_chiral(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b):

        fdsqrm_data = self.fDA * np.sqrt(self.mDA)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD)
        mpisqr = self.bstrapdata(self.mpisqr)

        # M1 = Fsqrtm_inf*( 1.0 + C1 / m + C2 / (m**2) + gamma *(m*self.a)**2 + eta*m*self.a*2 + mu*self.a**2)
        M1 = Fsqrtm_inf*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)


        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_chiral_dmss(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdsqrm_data = self.fDA * np.sqrt(self.mDA)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD)
        mpisqr = self.bstrapdata(self.mpisqr)
        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        # M1 = Fsqrtm_inf*( 1.0 + C1 / m + C2 / (m**2) + gamma *(m*self.a)**2 + eta*m*self.a*2 + mu*self.a**2)
        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)


        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdssqrtms_chiral_dmss(self, Fssqrtms_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdssqrms_data = self.fDsA * np.sqrt(self.mDsA)
        data = self.bstrapdata(fdssqrms_data)
        var = fdssqrms_data.var(1)

        m = self.bstrapdata(self.mDs)
        mpisqr = self.bstrapdata(self.mpisqr)
        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        # M1 = Fsqrtm_inf*( 1.0 + C1 / m + C2 / (m**2) + gamma *(m*self.a)**2 + eta*m*self.a*2 + mu*self.a**2)
        M1 = Fssqrtms_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)


        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)



    def fdsqrtm_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD) + self.m2 - self.m1

        M1 = Fsqrtm_inf*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_dmss_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, delta_S):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD) + self.m2 - self.m1

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - self.bstrapdata(self.mpisqr)
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000

        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_chiral_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD) + self.m2 - self.m1

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = Mss - phys_Mss


        M1 = Fsqrtm_inf*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a*2 + (mu*0.001)*self.a**2)



        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_chiral_dmss_HQET(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdsqrm_data = self.fDA_div * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD) + self.m2 - self.m1

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a**2 + (mu*0.001)*self.a**2)



        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_HQET_matched(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdsqrm_data = self.fD_matched * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)


        m = self.bstrapdata(self.mD_div) + self.m2 - self.m1

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a**2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdsqrtm_HQET_matched_nom2(self, Fsqrtm_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdsqrm_data = self.fD_matched * np.sqrt(self.mD_div)
        data = self.bstrapdata(fdsqrm_data)
        var = fdsqrm_data.var(1)

        m = self.bstrapdata(self.mD_div)

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        M1 = Fsqrtm_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a**2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdssqrtms_HQET_matched_nom2(self, Fssqrtms_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdssqrms_data = self.fDs_matched * np.sqrt(self.mDs_div)
        data = self.bstrapdata(fdssqrms_data)
        var = fdssqrms_data.var(1)

        m = self.bstrapdata(self.mDs_div)

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        M1 = Fssqrtms_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a**2 + (mu*0.001)*self.a**2)

        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)


    def fdssqrtms_chiral_dmss_HQET(self, Fssqrtms_inf, C1, C2, gamma, eta, mu, b, delta_S):

        fdssqrms_data = self.fDsA_div * np.sqrt(self.mDs_div)
        data = self.bstrapdata(fdssqrms_data)
        var = fdssqrms_data.var(1)

        m = self.bstrapdata(self.mDs) + self.m2 - self.m1

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)/10000000


        M1 = Fssqrtms_inf*(1+delta_S*delta_Mss)*(1.0+b*(mpisqr-phys_pion**2))*( 1.0 + C1*1000.0 / m + C2*1000000 / (m**2) + (gamma/10000.0) *(m*self.a)**2 + (eta/100.0)*m*self.a**2 + (mu*0.001)*self.a**2)



        sqr_diff1 = (data - M1)**2
        return np.sum(sqr_diff1/var)

    def fdssqrtms_ratio(self, z, z2, gamma_1):

        data = self.fDs_div_ratio * np.sqrt(self.Ds_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]



        m = self.bstrapdata(self.mDs) + self.m2 - self.m1
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_1*(A**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def fdssqrtms_mq_ratio(self, z, z2, gamma_A, gamma_S, gamma_P):

        data = self.fDs_div_ratio * np.sqrt(self.Ds_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def fdssqrtms_mq_ma_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.fDs_div_ratio * np.sqrt(self.Ds_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def fdsqrtm_mq_ma_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.fD_div_ratio * np.sqrt(self.D_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)


    def ms_mq_ma_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.Ds_mass_div_ratio / 1.25

        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mD_mq_ma_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.D_mass_div_ratio / 1.25

        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mD_corrected_pole_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.D_mass_div_ratio

        Lambda = (self.rho_mu_next / self.rho_mu)


        data = np.array([data[i] / Lambda[i] for i in range(data.shape[0]) ])

        datameans = self.bstrapdata(data)

        pdatameans = datameans[~np.isnan(datameans)]


        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)


    def mDs_mq_ma_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.Ds_mass_div_ratio / 1.25

        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mD_corrected_pole_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.D_mass_div_cor_ratio

        Lambda = (self.rho_mu_next / self.rho_mu)

        data = np.array([data[i] / Lambda[i] for i in range(data.shape[0]) ])


        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]


        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mDs_corrected_pole_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.Ds_mass_div_cor_ratio

        Lambda = (self.rho_mu_next / self.rho_mu)

        data = np.array([data[i] / Lambda[i] for i in range(data.shape[0]) ])

        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]

        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mD_pole_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.D_mass_div_ratio

        Lambda = (self.rho_mu_next / self.rho_mu)

        data = np.array([data[i] / Lambda[i] for i in range(data.shape[0]) ])


        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]


        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)

    def mDs_pole_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.Ds_mass_div_ratio

        Lambda = (self.rho_mu_next / self.rho_mu)

        data = np.array([data[i] / Lambda[i] for i in range(data.shape[0]) ])

        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]

        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata((self.mKsqr))) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))

        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)



    def fdsqrtmd_matched_ratio(self, z, z2, gamma_A, gamma_S, gamma_P, gamma_MA, gamma_MMA):

        data = self.fD_matched_ratio * np.sqrt(self.D_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        mpisqr = self.bstrapdata(self.mpisqr)

        Mss = (2.0*self.bstrapdata(self.mKsqr)) - mpisqr
        phys_Mss = (2.0*(phys_kaon**2)) - (phys_pion**2)

        delta_Mss = (Mss - phys_Mss)

        mpisqr = mpisqr[~np.isnan(datameans)]
        delta_Mss = delta_Mss[~np.isnan(datameans)]

        m = self.heavyq_mass
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_S*delta_Mss+gamma_P*(mpisqr-phys_pion**2)+gamma_A*(A**2)+gamma_MA*(m*A**2)+gamma_MMA*((m*A)**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)



    def fdsqrtm_ratio(self, z, z2, gamma_1):

        data =  self.fD_div_ratio * np.sqrt(self.D_mass_div_ratio)
        datameans = self.bstrapdata(data)
        pdatameans = datameans[~np.isnan(datameans)]
        datavar = data.var(1)
        pdatavar = datavar[~np.isnan(datameans)]

        m = self.bstrapdata(self.mD) + self.m2 - self.m1
        m = m[~np.isnan(datameans)]

        A = self.a[~np.isnan(datameans)]

        M1 = (1+gamma_1*(A**2)) * (1 + z/m + z2/(m**2))


        sqr_diff1 = (pdatameans - M1)**2
        return np.sum(sqr_diff1/pdatavar)
