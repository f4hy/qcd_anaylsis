from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging


class fdsqrtm_chiral_dmss(Model):
    """
    poorly named bu leaving it for now. Raw fsqrtm fit
    """
    def __init__(self, ensemble_datas, options, hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, each_heavy=True, **kargs)
        self.data["mhl"] = self.make_array("get_mass", flavor="heavy-ud", div=hqet)
        self.data["fhl"] = self.make_array("fhl", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["fhl"] * np.sqrt(self.data["mhl"])).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", -1.0e-9, 0.01)
        self.update_paramdict("eta", 0.0, 0.01, fixzero=True)
        self.update_paramdict("mu", -1.0e-4, 0.01)
        self.update_paramdict("b", 1.0e-8, 0.01)
        self.update_paramdict("delta_S", 1.0e-8, 0.01)
        self.update_paramdict("Fsqrtm_inf", 18000.0, 0.01)

        self.contlim_args = ["Fsqrtm_inf", "C1", "C2"]
        self.finbeta_args = ["Fsqrtm_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fsqrtm_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)

        asqr = (self.consts["a"]**2)
        asqr = (self.consts["lat"]**2)
        deltas = (1.0 + delta_S * delta_Mss + b * delta_mpisqr +
                  mu * asqr + eta * asqr / x + gamma * (asqr) / (x**2))
        # deltas = 1.0
        poly = Fsqrtm_inf * (1.0 + C1 * x + C2 * x**2)
        M = deltas * poly
        return M

    def sqr_diff(self, Fsqrtm_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):

        x = 1.0 / self.bstrapdata("mhl")
        M = self.m(x, Fsqrtm_inf, C1, C2, mu, eta, gamma, b, delta_S)
        data = self.bstrapdata("fhl") * np.sqrt(self.bstrapdata("mhl"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class fdssqrtms_chiral_dmss(Model):
    """
    poorly named bu leaving it for now. Raw fsqrtm fit
    """

    def __init__(self, ensemble_datas, options, hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, each_heavy=True, **kargs)
        self.data["mhs"] = self.make_array("get_mass", flavor="heavy-s", div=hqet)
        self.data["fhs"] = self.make_array("fhs", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", -1.0e-9, 0.01)
        self.update_paramdict("eta", 0.0, 0.01, fixzero=True)
        self.update_paramdict("mu", -1.0e-4, 0.01)
        self.update_paramdict("b", 1.0e-8, 0.01)
        self.update_paramdict("delta_S", 1.0e-8, 0.01)
        self.update_paramdict("Fssqrtms_inf", 18000.0, 0.01)

        self.contlim_args = ["Fssqrtms_inf", "C1", "C2"]
        self.finbeta_args = ["Fssqrtms_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fssqrtms_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)

        delta_Mss = Mss - phys_Mss
        asqr = (self.consts["a"]**2)
        asqr = (self.consts["lat"]**2)
        deltas = (1.0 + delta_S * delta_Mss + b * delta_mpisqr + mu * asqr + eta * asqr / x + gamma * (asqr) / (x**2))
        # deltas = 1
        poly = Fssqrtms_inf * (1.0 + C1 * x + C2 * x**2)
        M = deltas * poly
        return M

    def sqr_diff(self, Fssqrtms_inf, C1, C2, mu, eta, gamma, b, delta_S):

        x = 1.0 / self.bstrapdata("mhs")
        M = self.m(x, Fssqrtms_inf, C1, C2, mu, eta, gamma, b, delta_S)
        data = self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class fdsqrtm_HQET_matched_alphas(Model):
    """
    HQET fsqrtm global fit with Cmu and using alphas
    """
    def __init__(self, ensemble_datas, options, hqet=True, **kargs):

        Model.__init__(self, ensemble_datas, options, each_heavy=True, **kargs)
        self.data["fhl"] = self.make_array("fhl", div=hqet, matched=True)
        self.data["mhl"] = self.make_array("get_mass", flavor="heavy-ud", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        # Specify once so it is not called each bootstrap
        try:
            self.var = (self.data["fhl"] * np.sqrt(self.data["mhl"])).var(1)
        except IndexError:
            logging.info("run without data set var to NaN")
            self.var = np.NaN

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", 0.0, 0.01)
        self.update_paramdict("eta", 0.0, 0.01, fixzero=True)
        self.update_paramdict("mu", 0.0, 0.01)
        self.update_paramdict("b", 0.0, 0.01)
        self.update_paramdict("delta_S", 0.0, 0.01)
        self.update_paramdict("Fsqrtm_inf", 0.0, 0.01)

        self.contlim_args = ["Fsqrtm_inf", "C1", "C2"]
        self.finbeta_args = ["Fsqrtm_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fsqrtm_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)

        asqr = (self.consts["a"]**2)
        asqr = (self.consts["lat"]**2)
        alphas = self.consts["alphas"]
        deltas = (1.0 + delta_S * delta_Mss + b * delta_mpisqr +
                  mu * asqr + eta * asqr / x + gamma * alphas * (asqr) / (x**2))
        # deltas = 1
        poly = Fsqrtm_inf * (1.0 + C1 * x + C2 * x**2)
        M = deltas * poly
        return M

    def sqr_diff(self, Fsqrtm_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):

        x = 1.0 / (self.bstrapdata("mhl") + self.consts["m2"] - self.consts["m1"])
        M = self.m(x, Fsqrtm_inf, C1, C2, mu, eta, gamma, b, delta_S)
        data = self.bstrapdata("fhl") * np.sqrt(self.bstrapdata("mhl"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class fdssqrtms_HQET_matched_alphas(Model):
    """
    HQET fsqrtm global fit with Cmu and using alphas
    """
    def __init__(self, ensemble_datas, options, hqet=True, **kargs):

        hqet = True
        Model.__init__(self, ensemble_datas, options, each_heavy=True, **kargs)
        self.data["fhs"] = self.make_array("fhs", div=hqet, matched=True)
        self.data["mhs"] = self.make_array("get_mass", flavor="heavy-s", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", 0.0, 0.01)
        self.update_paramdict("eta", 0.0, 0.01, fixzero=True)
        self.update_paramdict("mu", 0.0, 0.01)
        self.update_paramdict("b", 0.0, 0.0001)
        self.update_paramdict("delta_S", 0.0, 0.00001)
        self.update_paramdict("Fssqrtms_inf", 180000.0, 0.01)

        logging.debug("Created model with inital parameters: {}".format(self.params))

        self.contlim_args = ["Fssqrtms_inf", "C1", "C2"]
        self.finbeta_args = ["Fssqrtms_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fssqrtms_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss
        asqr = (self.consts["a"]**2)
        asqr = (self.consts["lat"]**2)
        alphas = self.consts["alphas"]
        deltas = (1.0 + delta_S * delta_Mss + b * delta_mpisqr +
                  mu * asqr + eta * asqr / x + gamma * alphas * (asqr) / (x**2))
        # deltas = 1
        poly = Fssqrtms_inf * (1.0 + C1 * x + C2 * x**2)
        M = deltas * poly
        return M

    def sqr_diff(self, Fssqrtms_inf, C1, C2, mu, eta, gamma, b, delta_S):

        x = 1.0 / (self.bstrapdata("mhs") + self.consts["m2"] - self.consts["m1"])
        M = self.m(x, Fssqrtms_inf, C1, C2, mu, eta, gamma, b, delta_S)
        data = self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs"))
        sqr_diff = (data - M)**2

        return np.sum(sqr_diff / self.var)
