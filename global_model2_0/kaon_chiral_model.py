from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np


class fK_linear_mpisqr(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK") / np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"
        self.evalmodes = ["fk", "F_K0", "F_F_0", ]
        self.update_paramdict("F_K0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gammaK_s1", 0.0, 0.0001)
        self.update_paramdict("gammaK_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, x, F_K0, alpha, gammaK_1=0, gammaK_s1=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_F * F_K0 * (1.0 + alpha * x)

        return FK


    def sqr_diff(self, F_K0, alpha, gammaK_1=0, gammaK_s1=0):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, F_K0, alpha, gammaK_1, gammaK_s1)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)

class fK_linear_xi(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("F_K0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gammaK_s1", 0.0, 0.0001)
        self.update_paramdict("gammaK_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, xi, F_K0, alpha, gammaK_1=0, gammaK_s1=0):

        asqr = (self.consts["lat"]**2)

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        delta_FK = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_FK * F_K0 * (1.0 + alpha * xi)

        return FK

    def sqr_diff(self, F_K0, alpha, gammaK_1, gammaK_s1):
        xi = self.bstrapdata("xi")
        F = self.m(xi, F_K0, alpha, gammaK_1, gammaK_s1)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)


class fK_linear_x(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK") / np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"
        self.evalmodes = ["fk"]
        self.evalmodes = ["fk", "F_K0", "F_F_0", ]

        self.update_paramdict("F_K0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gammaK_s1", 0.0, 0.0001)
        self.update_paramdict("gammaK_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, F_K0, alpha, gammaK_1=0, gammaK_s1=0):

        asqr = (self.consts["lat"]**2)

        qm = self.consts["renorm_qmass"]
        B=2808
        Msqr = B*(qm+qm)
        F=84.04
        x = Msqr/(16*(np.pi**2)*(F**2))


        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        delta_FK = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_FK * F_K0 * (1.0 + alpha * x)

        return FK

    def eval_fit(self, x, F_K0, alpha):
        FK = F_K0 * (1.0 + alpha * x)
        if self.evalmode == "F_K0":
            return F_K0
        if self.evalmode == "fk":
            return FK
        if self.evalmode == "F_F_0":
            return FK / F_K0


    def sqr_diff(self, F_K0, alpha, gammaK_1, gammaK_s1):
        xi = self.bstrapdata("xi")
        F = self.m(F_K0, alpha, gammaK_1, gammaK_s1)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)




class fK_fpi_linear_chiral_x(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var = self.data["fK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("F_K0", 155.0, 1.0)
        self.update_paramdict("alpha", 0.1, 0.1)
        self.update_paramdict("gammaK_s1", 0.0, 0.0001)
        self.update_paramdict("gammaK_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0", "alpha"]
        self.finbeta_args = ["F_K0", "alpha", "gamma_1"]

    def m(self, xi, F_K0, alpha, gammaK_1, gammaK_s1, F, B, Lambda3,
          Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        asqr = (self.consts["lat"]**2)

        mpisqr = self.bstrapdata("mpi")**2
        arg3 = (Lambda3**2)/mpisqr
        arg4 = (Lambda4**2)/mpisqr

        arg12 = (Lambda12**2)/mpisqr

        l1 = -0.4
        l2 = 4.3
        # colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1

        Lambda1sqr = (pv.phys_pion**2)*np.exp(l1)
        Lambda2sqr = (pv.phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/mpisqr

        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        # lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        # lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)
        Mpisqr = delta_M * 2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2))

        Fpi = delta_F * F / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        delta_FK = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_F * F_K0 * (1.0 + alpha * xi)

        return FK / Fpi

    def sqr_diff(self, F_K0, alpha, gammaK_1, gammaK_s1, F, B,
                 Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1,
                 gamma_s2):
        xi = self.bstrapdata("xi")
        F = self.m(xi, F_K0, alpha, gammaK_1, gammaK_s1, F, B,
                   Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2,
                   gamma_s1, gamma_s2)
        sqr_diff = (self.bstrapdata("fK") - F)**2
        return np.sum(sqr_diff / self.var)



class fK_chiral_mud(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["FK"] = self.make_array("fK") / np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["meta"] = self.make_array("eta_mass")

        try:
            self.var = self.data["FK"].var(1)
        except IndexError:
            self.var = np.NaN

        self.label = "Linear fit"

        self.update_paramdict("F_0", 98.0, 1.0)
        self.update_paramdict("B_0", 2000.0, 100.0, limits=[500, 4000])
        self.update_paramdict("L4", 0.08, 0.2)
        self.update_paramdict("L5", 1.5, 0.5)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_1", -1.7, 0.0001)

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
                              + (B_0 / (F_0**2))*(4*(ms+mud)*L5/1000 + 8*(ms+2*mud)*L4/1000))

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
        self.update_paramdict("gamma_1", 0.0, 0.0001)

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


class fKbyfpi_linear_chiral_mpisqr(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK") / np.sqrt(2)
        self.data["fpi"] = self.make_array("fpi") / np.sqrt(2)
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
        self.update_paramdict("gamma_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0","alpha","F","B","Lambda3","Lambda4","Lambda12","cm","cf"]
        self.finbeta_args = ["F_K0","alpha","F","B","Lambda3","Lambda4","Lambda12","cm","cf","gamma_1", "gamma_2", "gammaK_1"]

    def m(self, x, F_K0, alpha, F, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1=0, gamma_2=0, gamma_s1=0, gamma_s2=0, gammaK_1=0, gammaK_s1=0):

        mpisqr = x
        arg3 = (Lambda3**2)/mpisqr
        arg4 = (Lambda4**2)/mpisqr

        arg12 = (Lambda12**2)/mpisqr

        l1 = -0.4
        l2 = 4.3
        # colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1

        Lambda1sqr = (pv.phys_pion**2)*np.exp(l1)
        Lambda2sqr = (pv.phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/mpisqr

        lnOmegaM = 1.0/15.0 * (60.0*np.log(arg12) - 33.0*np.log(arg3) - 12.0*np.log(arg4)+52.0)
        lnOmegaF = 1.0/3.0 * (-15.0*np.log(arg12) + 18.0*np.log(arg4) - 29.0/2.0)

        # lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        # lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        qm = (0.00018746525793*mpisqr)
        Msqr = B*(qm+qm)


        xi = mpisqr / (16 * (np.pi**2) * (delta_F * F * (1.0 + (Msqr/(16*(np.pi**2)*(F**2)))*np.log(arg4)))**2)


        Mpisqr = delta_M * 2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2))

        Fpi = delta_F * F / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))


        delta_FK = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_FK * F_K0 * (1.0 + alpha * x)

        return FK/Fpi

    def sqr_diff(self, ratio0, alpha, gamma_1=0, gamma_s1=0):
        raise RuntimeError("Not implemented yet, only for plotting")
        x = self.bstrapdata("mpi")**2
        F = self.m(x, ratio0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK")/self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff / self.var)

class fKbyfpi_linear_chiral_x(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fK"] = self.make_array("fK") / np.sqrt(2)
        self.data["fpi"] = self.make_array("fpi") / np.sqrt(2)
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
        self.update_paramdict("gamma_1", 0.0, 0.0001)

        self.contlim_args = ["F_K0","alpha","F","B","Lambda3","Lambda4","Lambda12","km","kf"]
        self.finbeta_args = ["F_K0","alpha","F","B","Lambda3","Lambda4","Lambda12","km","kf","gamma_1", "gamma_2", "gammaK_1"]

    def m(self, x, F_K0, alpha, F, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1=0, gamma_2=0, gamma_s1=0, gamma_s2=0, gammaK_1=0, gammaK_s1=0):

        Msqr = x * (16 * (np.pi**2) * (F**2))
        # x = Msqr/(16*(np.pi**2)*(F**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        arg12 = (Lambda12**2)/Msqr

        l1 = -0.4
        l2 = 4.3
        # colangelo
        # l1 = -0.4 \pm 0.6
        # l2 = 4.3 \pm 0.1

        Lambda1sqr = (pv.phys_pion**2)*np.exp(l1)
        Lambda2sqr = (pv.phys_pion**2)*np.exp(l2)

        lnLambda12sqr = (7.0*np.log(Lambda1sqr) + 8.0*np.log(Lambda2sqr))/15.0
        lambda12sqr = np.exp(lnLambda12sqr)

        arg12 = lambda12sqr/Msqr

        lm = 1.0/51.0 * (60.0*np.log(arg12) - 9.0*np.log(arg3)+49.0)
        lf = 1.0/30.0 * (30.0*np.log(arg12) + 6.0*np.log(arg3)-6.0*np.log(arg4)+23.0)

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = (Mss - phys_Mss)

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Mpisqr = delta_M * 2*B * (1.0 - 0.5*x*np.log(arg3) + 17.0/8.0*(x**2)*(lm)**2 + km*x**2)

        Fpi = delta_F * F * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        delta_FK = (1+gammaK_1*asqr + gammaK_s1*delta_Mss)
        FK = delta_FK * F_K0  * (1.0 + alpha * x) /  np.sqrt(2)
        # FK = delta_FK * F_K0 * (1.0 + alpha * Mpisqr)

        return FK/Fpi

    def sqr_diff(self, ratio0, alpha, gamma_1=0, gamma_s1=0):
        raise RuntimeError("Not implemented yet, only for plotting")
        x = self.bstrapdata("mpi")**2
        F = self.m(x, ratio0, alpha, gamma_1, gamma_s1)
        sqr_diff = (self.bstrapdata("fK")/self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff / self.var)
