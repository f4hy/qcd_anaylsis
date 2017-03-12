from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging
import inspect


class fpi_x_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)
        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.label = "NLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))
        self.update_paramdict("Lambda4", 1100.0, 50.0, limits=(0, 4000))
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma_2", -2.0, 0.1)

        self.contlim_args = ["F", "B", "Lambda4"]
        self.finbeta_args = ["F", "B", "Lambda4", "gamma_s2", "gamma_2"]

    def m(self, x, F, B, Lambda4, gamma_2=0, gamma_s2=0):
        mpisqr = x
        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(16*(np.pi**2)*(F**2))
        arg4 = (Lambda4**2)/Msqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - mpisqr)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)
        Fpi = delta_F * F * (1.0 + x*np.log(arg4))

        return Fpi

    def eval_fit(self, x, F, B, Lambda4):
        """ Override the normal evaluation, this needs to not call m in this case """
        Msqr = x * (16.*(np.pi**2)*(F**2))
        arg4 = (Lambda4**2)/Msqr
        return F * (1.0 + x*np.log(arg4))

    def sqr_diff(self, F, B, Lambda4, gamma_2, gamma_s2):
        x = self.bstrapdata("mpi")**2
        Fp = self.m(x, F, B, Lambda4, gamma_2, gamma_s2)
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff2 / self.var2)

class fpi_mpi_x_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.evalmodes = ["fpi", "mpi", "fpi_f"]

        self.label = "NLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))

        self.update_paramdict("Lambda3", 500.0, 50.0)
        self.update_paramdict("Lambda4",  1100.0, 50.0, limits=(0, None))
        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s1", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)

        self.contlim_args = ["F", "B", "Lambda3", "Lambda4" ]
        self.finbeta_args = ["F", "B", "Lambda3", "Lambda4", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        return Model.degrees_of_freedom(self,data_multiply=2.0)

    def m(self, F, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):

        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(16*(np.pi**2)*(F**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Mpisqr = delta_M * 2*B*(1.0-0.5*x*np.log(arg3))
        Fpi = delta_F * F * (1.0 + x*np.log(arg4))

        return Mpisqr, Fpi

    def eval_fit(self, x, F, B, Lambda3, Lambda4):
        """ Override the normal evaluation, this needs to not call m in this case """
        Msqr = x * (16 * (np.pi**2) * (F**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr
        if self.evalmode == "fpi":
            return F * (1.0 + x*np.log(arg4))
        if self.evalmode == "fpi_f":
            return (1.0 + x*np.log(arg4))
        else:
            return 2 * B * (1.0-0.5*x*np.log(arg3))

    def sqr_diff(self, F, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):

        M, Fp = self.m(F, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)



class fpi_mpi_x_NNLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.evalmodes = ["fpi", "mpi", "fpi_f"]
        self.label = "NNLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))

        self.update_paramdict("Lambda3", 500.0, 50.0)
        self.update_paramdict("Lambda4",  1000.0, 50.0, limits=(0, None))
        self.update_paramdict("Lambda12", 20.0, 0.1, limits=(0, None), fix=True)
        self.update_paramdict("km", -0.15, 0.01)
        self.update_paramdict("kf", 1.98, 0.01)
        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s1", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)

        self.contlim_args = ["F", "B", "Lambda3", "Lambda4", "Lambda12", "km", "kf", ]
        self.finbeta_args = ["F", "B", "Lambda3", "Lambda4", "Lambda12", "km", "kf", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        return Model.degrees_of_freedom(self,data_multiply=2.0)

    def m(self, F, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(16*(np.pi**2)*(F**2))
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

        return Mpisqr, Fpi

    def eval_fit(self, x, F, B, Lambda3, Lambda4, Lambda12, km, kf):
        """ Override the normal evaluation, this needs to not call m in this case """
        Msqr = x * (16 * (np.pi**2) * (F**2))
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

        if self.evalmode == "fpi":
            return F * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)
        elif self.evalmode == "fpi_f":
            return (F * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)) / F
        else:
            return 2 * B * (1.0 - 0.5 * x * np.log(arg3) + 17.0 / 8.0 * (x**2) * (lm)**2 + km * x**2)

    def sqr_diff(self, F, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        M, Fp = self.m(F, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)



class fpi_mpi_xi_inverse_NNLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.evalmodes = ["fpi", "mpi", "fpi_f"]

        self.label = "NNLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))

        self.update_paramdict("Lambda3", 500.0, 50.0, limits=(0, None))
        self.update_paramdict("Lambda4",  1100.0, 50.0, limits=(0, None))
        self.update_paramdict("Lambda12", 20.0, 0.1, limits=(0, None), fix=True)
        self.update_paramdict("cm", 2.0, 1.1)
        self.update_paramdict("cf", -12.0, 1.1)
        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.000001)
        self.update_paramdict("gamma_s2", 0.0, 0.000001)

        self.contlim_args = ["F", "B", "Lambda3", "Lambda4", "Lambda12", "cm", "cf", ]
        self.finbeta_args = ["F", "B", "Lambda3", "Lambda4", "Lambda12", "cm", "cf", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        return Model.degrees_of_freedom(self, data_multiply=2.0)

    def m(self, xi, F, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):

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

        return Mpisqr, Fpi

    def eval_fit(self, xi, F, B, Lambda3, Lambda4, Lambda12, cm, cf):
        """ Override the normal evaluation, this needs to not call m in this case """
        self.data["mpi"] = np.array([[pv.phys_pion]])
        self.data["mK"] = np.array([[pv.phys_kaon]])
        Mpisqr, Fpi = self.m(xi, F, B, Lambda3, Lambda4, Lambda12, cm, cf, 0, 0 ,0 ,0)
        if self.evalmode == "fpi":
            return Fpi
        if self.evalmode == "fpi_f":
            return Fpi/F
        else:
            return Mpisqr

    def sqr_diff(self, F, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        xi = self.bstrapdata("xi")
        M, Fp = self.m(xi, F, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)


class fpi_mpi_xi_NNLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.evalmodes = ["fpi", "mpi", "fpi_f"]
        self.label = "NNLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))

        self.update_paramdict("c3", 1453.0, 100.0, limits=(-100, 8000))
        self.update_paramdict("c4", -100.0, 100.0)
        self.update_paramdict("alpha", -400.0, 1.0)
        self.update_paramdict("beta", 200.0, 1.0)
        self.update_paramdict("ellphys", -32.0, 4.3, fix=True)

        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s1", 1.0e-7, 1.0e-8)
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)

        self.contlim_args = ["F", "B", "c3", "c4", "alpha", "beta", "ellphys"]
        self.finbeta_args = ["F", "B", "c3", "c4", "alpha", "beta", "ellphys", "gamma_1", "gamma_2"]


    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        return Model.degrees_of_freedom(self, data_multiply=2.0)

    def m(self, xi, F, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)


        Fpi = delta_F * (F * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi)) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2)
        Mpisqr = delta_M * (2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                                 (c4/F - 1.0/3.0 * (ellphys+16))*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2)

        return Mpisqr, Fpi

    def eval_fit(self, xi, F, B, c3, c4, alpha, beta, ellphys):
        """ Override the normal evaluation, this needs to not call m in this case """

        if self.evalmode == "fpi":
            return F * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi)) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2
        if self.evalmode == "fpi_f":
            return (F * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi)) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2) / F
        else:
            return 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                        (c4/F - 1.0/3.0 * (ellphys+16))*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2

    def sqr_diff(self, F, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2):

        xi = self.bstrapdata("xi")
        M, Fp = self.m(xi, F, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)

class fpi_xi_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN


        self.label = "NLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("c4", 0.0, 1.0)

        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)

        self.contlim_args = ["F", "c4"]
        self.finbeta_args = ["F", "c4", "gamma_2"]


    def m(self, xi, F, c4, gamma_2=0, gamma_s2=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Fpi = delta_F * (F * (1 - xi*np.log(xi)) + c4 * xi)

        return Fpi


    def sqr_diff(self, F, c4, gamma_2, gamma_s2):

        xi = self.bstrapdata("xi")
        Fp = self.m(xi, F, c4, gamma_2, gamma_s2)
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff2 / self.var2)

class fpi_xi_inverse_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.evalmodes = ["fpi", "mpi", "fpi_f"]

        self.label = "NLO"

        self.update_paramdict("F", 90.0, 2.0, limits=(0, None))
        self.update_paramdict("Lambda4", 1100.0, 50.0)

        self.update_paramdict("gamma_2", -2.0, 0.1)
        self.update_paramdict("gamma_s2", 1.0e-7, 1.0e-8)

        self.contlim_args = ["F", "Lambda4"]
        self.finbeta_args = ["F", "Lambda4", "gamma_2"]

    def eval_fit(self, xi, F, Lambda4):
        """ Override the normal evaluation, this needs to not call m in this case """
        self.data["mpi"] = np.array([[pv.phys_pion]])
        self.data["mK"] = np.array([[pv.phys_kaon]])
        if self.evalmode == "fpi_f":
            return self.m( xi, F, Lambda4) / F
        else:
            return self.m( xi, F, Lambda4)


    def m(self, xi, F, Lambda4, gamma_2=0, gamma_s2=0):

        mpisqr = self.bstrapdata("mpi")**2
        arg4 = (Lambda4**2)/mpisqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Fpi = delta_F * (F / (1 - xi*np.log(arg4)))

        return Fpi


    def sqr_diff(self, F, Lambda4, gamma_2, gamma_s2):

        xi = self.bstrapdata("xi")
        Fp = self.m(xi, F, Lambda4, gamma_2, gamma_s2)
        sqr_diff2 = (self.bstrapdata("fpi") - Fp)**2
        return np.sum(sqr_diff2 / self.var2)


class mpi_xi_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.label = "NLO"

        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))
        self.update_paramdict("c3", 1453.0, 1.0)

        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_s1", 1.0e-7, 1.0e-8)

        self.contlim_args = ["B", "c3"]
        self.finbeta_args = ["B", "c3", "gamma_1"]


    def m(self, xi, B, c3, gamma_1=0, gamma_s1=0):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)
        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)

        Mpisqr = delta_M * (2*B*(1.0+0.5*xi*np.log(xi)) + c3*xi)

        return Mpisqr

    def sqr_diff(self, B, c3, gamma_1, gamma_s1):

        xi = self.bstrapdata("xi")
        M = self.m(xi, B, c3, gamma_1, gamma_s1)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        return np.sum(sqr_diff1 / self.var1)

class mpi_xi_inverse_NLO(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fpi"] = self.make_array("fpi")/np.sqrt(2)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["xi"] = self.make_array("xi")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.label = "NLO"

        self.update_paramdict("B", 2900.0, 10.0, limits=(0, None))
        self.update_paramdict("Lambda3", 500.0, 50.0)

        self.update_paramdict("gamma_1", 2.0, 0.1)
        self.update_paramdict("gamma_s1", 1.0e-7, 1.0e-8)

        self.contlim_args = ["B", "Lambda3"]
        self.finbeta_args = ["B", "Lambda3", "gamma_1"]


    def m(self, xi, B, Lambda3, gamma_1=0, gamma_s1=0):

        mpisqr = self.bstrapdata("mpi")**2
        arg3 = (Lambda3**2)/mpisqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)
        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)

        Mpisqr = delta_M * (2*B/(1.0+0.5*xi*np.log(arg3)))

        return Mpisqr

    def eval_fit(self, xi, B, Lambda3):
        """ Override the normal evaluation, this needs to not call m in this case """
        self.data["mpi"] = np.array([[pv.phys_pion]])
        self.data["mK"] = np.array([[pv.phys_kaon]])
        return self.m( xi, B, Lambda3)


    def sqr_diff(self, B, Lambda3, gamma_1, gamma_s1):

        xi = self.bstrapdata("xi")
        M = self.m(xi, B, Lambda3, gamma_1, gamma_s1)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        return np.sum(sqr_diff1 / self.var1)
