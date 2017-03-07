from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging


class linear_fD_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fhl"] = self.make_array("fD", heavy=heavy, div=hqet)
        self.data["mhl"] = self.make_array("D_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["fhl"]).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear continuum fit"

        logging.info(self.bstrapdata("fhl") * np.sqrt(self.bstrapdata("mhl")))
        self.update_paramdict("FDphys", 200.0, 1.0)
        self.update_paramdict("b", 0.1, 0.1)
        self.update_paramdict("gamma_1", -4.0, 0.1)
        self.update_paramdict("gamma_s1", 0.01, 0.1)
        self.contlim_args = ["FDphys", "b"]
        self.finbeta_args = ["FDphys", "b", "gamma_1"]

    def m(self, x, FDphys, b, gamma_1=0.0, gamma_s1=0.0):
        delta_mpisqr = (x) - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - x)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["a"]**2)
        linear = FDphys * (1.0 + b * delta_mpisqr)
        deltas = (1.0 + gamma_1 * asqr + gamma_s1 * delta_Mss)
        M = deltas * linear
        return M

    def sqr_diff(self, FDphys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, FDphys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("fhl")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class linear_fDs_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["fhs"]).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear continuum fit"

        self.update_paramdict("FDsphys", 200.0, 20.0)
        self.update_paramdict("b", 0.0, 0.1)
        self.update_paramdict("gamma_1", 0.0, 0.01)
        self.update_paramdict("gamma_s1", 0.0, 0.01)
        self.contlim_args = ["FDsphys", "b"]
        self.finbeta_args = ["FDsphys", "b", "gamma_1"]

    def m(self, x, FDsphys, b, gamma_1=0.0, gamma_s1=0.0):
        delta_mpisqr = (x) - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - x)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss
        asqr = (self.consts["a"]**2)
        linear = FDsphys * (1.0 + b * delta_mpisqr)
        deltas = (1.0 + gamma_1 * asqr + gamma_s1 * delta_Mss)
        M = deltas * linear
        return M

    def sqr_diff(self, FDsphys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, FDsphys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("fhs")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)

class linear_mD_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mhl"] = self.make_array("D_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["mhl"]).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear continuum fit"

        self.update_paramdict("mD_phys", 1900.0, 1.0)
        self.update_paramdict("b", 0.0, 0.1)
        self.update_paramdict("gamma_1", 0.0, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.1)
        self.contlim_args = ["mD_phys", "b"]
        self.finbeta_args = ["mD_phys", "b", "gamma_1"]

    def m(self, x, mD_phys, b, gamma_1=0.0, gamma_s1=0.0):
        delta_mpisqr = (x) - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - x)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["a"]**2)
        linear = mD_phys * (1.0 + b * delta_mpisqr)
        deltas = (1.0 + gamma_1 * asqr + gamma_s1 * delta_Mss)
        M = deltas * linear
        return M

    def sqr_diff(self, mD_phys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, mD_phys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("mhl")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)

class linear_mDs_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["mhs"]).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear continuum fit"

        self.update_paramdict("mDs_phys", 2000.0, 1.0)
        self.update_paramdict("b", 0.0, 0.1)
        self.update_paramdict("gamma_1", 0.0, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.1)
        self.contlim_args = ["mDs_phys", "b"]
        self.finbeta_args = ["mDs_phys", "b", "gamma_1"]

    def m(self, x, mDs_phys, b, gamma_1=0.0, gamma_s1=0.0):
        delta_mpisqr = (x) - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - x)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["a"]**2)
        linear = mDs_phys * (1.0 + b * delta_mpisqr)
        deltas = (1.0 + gamma_1 * asqr + gamma_s1 * delta_Mss)
        M = deltas * linear
        return M

    def sqr_diff(self, mDs_phys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, mDs_phys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("mhs")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)

class linear_mDsplit_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mhl"] = self.make_array("D_mass", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = (self.data["mhs"]).var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear continuum fit"

        self.update_paramdict("mDsplit_phys", 2000.0, 1.0)
        self.update_paramdict("b", 0.0, 0.1)
        self.update_paramdict("gamma_1", 0.0, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.1)
        self.contlim_args = ["mDsplit_phys", "b"]
        self.finbeta_args = ["mDsplit_phys", "b", "gamma_1"]

    def m(self, x, mDsplit_phys, b, gamma_1=0.0, gamma_s1=0.0):
        delta_mpisqr = (x) - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - x)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["a"]**2)
        linear = mDsplit_phys * (1.0 + b * delta_mpisqr)
        deltas = (1.0 + gamma_1 * asqr + gamma_s1 * delta_Mss)
        M = deltas * linear
        return M

    def sqr_diff(self, mDsplit_phys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, mDsplit_phys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("mhs")-self.bstrapdata("mhl")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)
