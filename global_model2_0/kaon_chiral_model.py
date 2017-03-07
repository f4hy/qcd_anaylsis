from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging
import inspect


class fK_linear_mpisqr(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("F_K0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_1", 0.0, 0.0001, fixzero=True)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, x, F_K0, alpha, gamma_1=0, gamma_s1=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_1*asqr + gamma_s1*delta_Mss)
        Fpi = delta_F * F_K0 * (1.0 + alpha * x )

        return Fpi

    def sqr_diff(self, F_K0, alpha, gamma_1=0, gamma_s1=0):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, F_K0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)

class fKbyfpi_linear_mpisqr(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK")
        self.data["fpi"] = self.make_array("fpi")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("ratio0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_1", 0.0, 0.0001, fixzero=True)

        self.contlim_args = ["ratio0", "alpha"]
        self.finbeta_args = ["ratio0", "alpha", "gamma_1"]

    def m(self, x, ratio0, alpha, gamma_1=0, gamma_s1=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_1*asqr + gamma_s1*delta_Mss)
        Fpi = delta_F * ratio0 * (1.0 + alpha * x )

        return Fpi

    def sqr_diff(self, ratio0, alpha, gamma_1=0, gamma_s1=0):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, ratio0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK")/self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff / self.var)
