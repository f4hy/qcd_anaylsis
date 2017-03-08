from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np


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
        self.update_paramdict("gamma_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, x, F_K0, alpha, gamma_1=0, gamma_s1=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_1*asqr + gamma_s1*delta_Mss)
        FK = delta_F * F_K0 * (1.0 + alpha * x)

        return FK

    def sqr_diff(self, F_K0, alpha, gamma_1=0, gamma_s1=0):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, F_K0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)


class fK_chiral_mud(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["FK"] = self.make_array("fK") / np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = self.data["FK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("F_0", 80.0, 1.0, fix=True)
        self.update_paramdict("B_0", 1000.0, 100.0, limits=[0, None])
        self.update_paramdict("L4", 0.0005, 0.0005)
        self.update_paramdict("L5", 0.02, 0.001)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_1", 0.0, 0.0001, fixzero=True)

        self.contlim_args = ["F_0", "B_0", "L4", "L5"]
        self.finbeta_args = ["F_0", "B_0", "L4", "L5", "gamma_1"]

    def m(self, x, F_0, B_0, L4, L5, gamma_1=0, gamma_s1=0):
        mud = x
        ms = self.consts["renorm_qs"]

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_1*asqr + gamma_s1*delta_Mss)

        mpsqr = 2 * B_0 * mud
        mksqr = B_0 * (ms + mud)
        metasqr = (2.0/3.0) * B_0 * (2*ms + mud)

        denom = (32 * (F_0**2) * np.pi**2)

        musqr = 770.0**2

        mup = (mpsqr / denom) * np.log(mpsqr / musqr)
        muk = (mksqr / denom) * np.log(mksqr / musqr)
        mueta = (metasqr / denom) * np.log(metasqr / musqr)

        FK = delta_F * F_0 * (1.0 - (3.0/4.0)*mup - (3.0/2.0)*muk - (3.0/4.0)*mueta
                              + (B_0 / (F_0**2))*(4*(ms+mud)*L5 + 8*(ms+2*mud)*L4))
        # print (3.0/4.0)*mup - (3.0/2.0)*muk - (3.0/4.0)*mueta
        # print (3.0/4.0)*mup, (3.0/2.0)*muk, (3.0/4.0)*mueta
        # print mup, muk, mueta
        # exit(-1)
        return FK

    def eval_fit(self, x, F_0, B_0, L4, L5, gamma_1=0, gamma_s1=0):
        self.consts["renorm_qs"] = 80.0
        return self.m(x, F_0, B_0, L4, L5, gamma_1=0, gamma_s1=0)

    def sqr_diff(self, F_0, B_0, L4, L5, gamma_1=0, gamma_s1=0):
        x = self.consts["renorm_qmass"]
        F = self.m(x, F_0, B_0, L4, L5, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("FK") - F)**2
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
        Fratio = delta_F * ratio0 * (1.0 + alpha * x)

        return Fratio

    def sqr_diff(self, ratio0, alpha, gamma_1=0, gamma_s1=0):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, ratio0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK")/self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff / self.var)
