from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging
import inspect


class mpisqr_mud(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (self.data["mpi"]**2).var(1)
        except IndexError:
            self.var1 = np.NaN

        self.label = "NLO"

        self.update_paramdict("slope", 5000, 2.0, limits=(0, None))
        self.update_paramdict("gamma_s", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma", -2.0, 0.1)

        self.contlim_args = ["slope"]
        self.finbeta_args = ["slope", "gamma_2"]

    def m(self, x, slope, gamma=0, gamma_s=0):
        qm = x

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta = (1+gamma*asqr+gamma_s*delta_Mss)
        mpisqr = delta * (0.0 + slope*x)

        return mpisqr

    def sqr_diff(self, slope, gamma, gamma_s):
        x = self.consts["renorm_qmass"]
        mpisqr = self.m(x, slope, gamma, gamma_s)
        sqr_diff = (self.bstrapdata("mpi")**2 - mpisqr)**2
        return np.sum(sqr_diff / self.var1)

class mud_mpisqr(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (self.consts["residual_error/Z"]).var()
        except IndexError as e:
            self.var1 = np.NaN
        except AttributeError:
            self.var1 = np.NaN

        self.label = "NLO"

        self.update_paramdict("slope", 0.0000001, 0.0000001)
        self.update_paramdict("gamma_s", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma", 1.0e-7, 1.0e-8)

        self.contlim_args = ["slope"]
        self.finbeta_args = ["slope", "gamma_2"]

    def m(self, x, slope, gamma=0, gamma_s=0):
        mpisqr = x

        Mss = (2.0 * self.bstrapdata("mK")**2 - mpisqr)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta = (1+gamma*asqr+gamma_s*delta_Mss)
        qm = delta * (0.0 + slope*mpisqr)

        return qm

    def sqr_diff(self, slope, gamma, gamma_s):
        x = self.bstrapdata("mpi")**2
        qm = self.m(x, slope, gamma, gamma_s)
        sqr_diff = (self.consts["renorm_qmass"] - qm)**2
        return np.sum(sqr_diff / self.var1)
