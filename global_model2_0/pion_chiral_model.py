from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging
import inspect


class fpi_x_NLO(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["fpi"] = self.make_array("fpi")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        self.label = "Linear continuum fit"

        self.update_paramdict("F_0", 110.0, 1.0, limits=(0, None))
        self.update_paramdict("B", 3000.0, 3.0, limits=(2000, None))
        self.update_paramdict("Lambda4", 1169.7, 1.0, limits=(0, 4000))
        self.update_paramdict("gamma_s2", 0.0, 0.0001)
        self.update_paramdict("gamma_2", 0.0, 0.0001, fixzero=True)

        self.contlim_args = ["F_0", "B", "Lambda4"]
        self.finbeta_args = ["F_0", "B", "Lambda4", "gamma_s2", "gamma_2"]

    def m(self, x, F_0, B, Lambda4, gamma_2=0, gamma_s2=0):
        mpisqr = x
        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg4 = (Lambda4**2)/Msqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - mpisqr)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)
        Fpi = delta_F * F_0 * (1.0 + x*np.log(arg4))

        return Fpi

    def plot_fit(self, x, F_0, B, Lambda4):
        """ Override the normal plotter, this needs to not call m in this case """
        Msqr = x * (8*(np.pi**2)*(F_0**2))
        arg4 = (Lambda4**2)/Msqr
        return F_0 * (1.0 + x*np.log(arg4))

    def sqr_diff(self, F_0, B, Lambda4, gamma_2, gamma_s2):
        x = self.bstrapdata("mpi")**2
        F = self.m(x, F_0, B, Lambda4, gamma_2, gamma_s2)
        sqr_diff2 = (self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff2 / self.var2)

class fpi_mpi_x_NLO(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["fpi"] = self.make_array("fpi")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        try:
            if "fpi" in options.ydata:
                self.plotmode = "fpi"
            else:
                self.plotmode = "mpisqr"
        except AttributeError as e:
            self.plotmode = ""

        self.label = "Linear continuum fit"

        self.update_paramdict("F_0", 117.8, 0.1, limits=(0, None))
        self.update_paramdict("B", 2736.37, 1.0, limits=(2500, None))

        self.update_paramdict("Lambda3", 579.59, 50.0)
        self.update_paramdict("Lambda4",  967.35, 140.55, limits=(0, None))
        self.update_paramdict("gamma_1", -0.1, 0.1)
        self.update_paramdict("gamma_2", -0.1, 0.1)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_s2", 0.0, 0.0001)

        self.contlim_args = ["F_0", "B", "Lambda3", "Lambda4" ]
        self.finbeta_args = ["F_0", "B", "Lambda3", "Lambda4", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        logging.info("data")
        datapoints = [d.shape[0] for d in self.data.values()]
        ndata = datapoints[0] * 2 # Twice the dof because combined fit to two data sets

        fixed_parms = [p for p in self.params if "fix" in p and self.params[p]]
        Nparams = inspect.getargspec(self.m).args[1:]
        Nfree_params = len(Nparams) - len(fixed_parms)
        dof = float(ndata - Nfree_params)
        logging.info("DOF {}, data {}, free {}".format(dof, ndata, Nfree_params))
        return dof

    def m(self, F_0, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):

        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Mpisqr = delta_M * 2*B*(1.0-0.5*x*np.log(arg3))
        Fpi = delta_F * F_0 * (1.0 + x*np.log(arg4))

        return Mpisqr, Fpi

    def plot_fit(self, x, F_0, B, Lambda3, Lambda4):
        """ Override the normal plotter, this needs to not call m in this case """
        Msqr = x * (8 * (np.pi**2) * (F_0**2))
        arg3 = (Lambda3**2)/Msqr
        arg4 = (Lambda4**2)/Msqr

        if self.plotmode == "fpi":
            return F_0 * (1.0 + x*np.log(arg4))
        else:
            return 2 * B * (1.0-0.5*x*np.log(arg3))

    def sqr_diff(self, F_0, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2):

        M, F = self.m(F_0, B, Lambda3, Lambda4, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)



class fpi_mpi_x_NNLO(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["fpi"] = self.make_array("fpi")
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        try:
            self.var1 = (((self.data["mpi"]**2).std(1)/self.consts["renorm_qmass"])**2
                         + (self.consts["residual_error"]*self.data["mpi"].mean(1) / self.consts["qmass"])**2)
            self.var2 = (self.data["fpi"]).var(1)

        except IndexError:
            self.var1 = np.NaN
            self.var2 = np.NaN

        try:
            if "fpi" in options.ydata:
                self.plotmode = "fpi"
            else:
                self.plotmode = "mpisqr"
        except AttributeError as e:
            self.plotmode = ""

        self.label = "Linear continuum fit"

        self.update_paramdict("F_0", 117.8, 0.1, limits=(0, None))
        self.update_paramdict("B", 2736.37, 1.0, limits=(2500, None))

        self.update_paramdict("Lambda3", 579.59, 50.0)
        self.update_paramdict("Lambda4",  967.35, 140.55, limits=(0, None))
        self.update_paramdict("Lambda12", 20.0, 0.1, limits=(0, None), fix=True)
        self.update_paramdict("km", -0.15, 0.01)
        self.update_paramdict("kf", 1.98, 0.01)
        self.update_paramdict("gamma_1", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_2", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_s2", 0.0, 0.0001)

        self.contlim_args = ["F_0", "B", "Lambda3", "Lambda4", "Lambda12", "km", "kf", ]
        self.finbeta_args = ["F_0", "B", "Lambda3", "Lambda4", "Lambda12", "km", "kf", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        logging.info("data")
        datapoints = [d.shape[0] for d in self.data.values()]
        ndata = datapoints[0] * 2 # Twice the dof because combined fit to two data sets

        fixed_parms = [p for p in self.params if "fix" in p and self.params[p]]
        Nparams = inspect.getargspec(self.m).args[1:]
        Nfree_params = len(Nparams) - len(fixed_parms)
        dof = float(ndata - Nfree_params)
        logging.info("DOF {}, data {}, free {}".format(dof, ndata, Nfree_params))
        return dof

    def m(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        qm = self.consts["renorm_qmass"]
        Msqr = B*(qm+qm)
        x = Msqr/(8*(np.pi**2)*(F_0**2))
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
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)

        Mpisqr = delta_M * 2*B * (1.0 - 0.5*x*np.log(arg3) + 17.0/8.0*(x**2)*(lm)**2 + km*x**2)

        Fpi = delta_F * F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)

        return Mpisqr, Fpi

    def plot_fit(self, x, F_0, B, Lambda3, Lambda4, Lambda12, km, kf):
        """ Override the normal plotter, this needs to not call m in this case """
        Msqr = x * (8 * (np.pi**2) * (F_0**2))
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

        if self.plotmode == "fpi":
            return F_0 * (1.0 + x*np.log(arg4)-5.0/4.0*(x**2)*(lf)**2 + kf*x**2)
        else:
            return 2 * B * (1.0 - 0.5 * x * np.log(arg3) + 17.0 / 8.0 * (x**2) * (lm)**2 + km * x**2)

    def sqr_diff(self, F_0, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        M, F = self.m(F_0, B, Lambda3, Lambda4, Lambda12, km, kf, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)



class fpi_mpi_xi_inverse_NNLO(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["fpi"] = self.make_array("fpi")
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

        try:
            if "fpi" in options.ydata:
                self.plotmode = "fpi"
            else:
                self.plotmode = "mpisqr"
        except AttributeError as e:
            self.plotmode = ""

        self.label = "Linear continuum fit"

        self.update_paramdict("F_0", 117.8, 0.1, limits=(0, None))
        self.update_paramdict("B", 2736.37, 1.0, limits=(0, None))

        self.update_paramdict("Lambda3", 579.59, 50.0, limits=(0, None))
        self.update_paramdict("Lambda4",  967.35, 140.55, limits=(0, None))
        self.update_paramdict("Lambda12", 20.0, 0.1, limits=(0, None), fix=True)
        self.update_paramdict("cm", 1.0, 0.1)
        self.update_paramdict("cf", -12.0, 0.1)
        self.update_paramdict("gamma_1", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_2", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_s1", 0.0, 0.0001, fixzero=True)
        self.update_paramdict("gamma_s2", 0.0, 0.0001, fixzero=True)

        self.contlim_args = ["F_0", "B", "Lambda3", "Lambda4", "Lambda12", "cm", "cf", ]
        self.finbeta_args = ["F_0", "B", "Lambda3", "Lambda4", "Lambda12", "cm", "cf", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        logging.info("data")
        datapoints = [d.shape[0] for d in self.data.values()]
        ndata = datapoints[0] * 2 # Twice the dof because combined fit to two data sets

        fixed_parms = [p for p in self.params if "fix" in p and self.params[p]]
        Nparams = inspect.getargspec(self.m).args[1:]
        Nfree_params = len(Nparams) - len(fixed_parms)
        dof = float(ndata - Nfree_params)
        logging.info("DOF {}, data {}, free {}".format(dof, ndata, Nfree_params))
        return dof

    def m(self, xi, F_0, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        mpisqr = self.bstrapdata("mpi")**2
        print mpisqr
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

        print arg3
        print arg4
        Mpisqr = delta_M * 2*B / (1.0 + 0.5*xi*np.log(arg3) -5.0/8.0*(xi*lnOmegaM)**2 + cm*(xi**2))

        Fpi = delta_F * F_0 / (1.0 - xi*np.log(arg4) - 1.0/4.0*(xi*lnOmegaF)**2 + cf*(xi**2))

        return Mpisqr, Fpi

    def plot_fit(self, xi, F_0, B, Lambda3, Lambda4, Lambda12, cm, cf):
        """ Override the normal plotter, this needs to not call m in this case """
        Mpisqr, Fpi = self.m(xi, F_0, B, Lambda3, Lambda4, Lambda12, cm, cf, 0, 0 ,0 ,0)
        print Fpi
        print Mpisqr
        exit(-1)
        return Fpi
        if self.plotmode == "fpi":
            return Fpi
        else:
            return Mpisqr

    def sqr_diff(self, F_0, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2):

        xi = self.bstrapdata("xi")
        M, F = self.m(xi, F_0, B, Lambda3, Lambda4, Lambda12, cm, cf, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)


class fpi_mpi_xi_NNLO(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["fpi"] = self.make_array("fpi")
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

        try:
            if "fpi" in options.ydata:
                self.plotmode = "fpi"
            else:
                self.plotmode = "mpisqr"
        except AttributeError as e:
            self.plotmode = ""

        self.label = "Linear continuum fit"

        self.update_paramdict("F_0", 117.8, 0.1, limits=(0, None))
        self.update_paramdict("B", 2736.37, 1.0, limits=(2500, None))

        self.update_paramdict("c3", 3000.0, 1.0, limits=(-1000, 4000))
        self.update_paramdict("c4", -1000.0, 1.0)
        self.update_paramdict("alpha", -4000.0, 1.0)
        self.update_paramdict("beta", 2000.0, 1.0)
        self.update_paramdict("ellphys", -32.0, 4.3, fix=True)

        self.update_paramdict("gamma_1", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_2", -0.1, 0.1, fixzero=True)
        self.update_paramdict("gamma_s1", 0.0, 0.0001)
        self.update_paramdict("gamma_s2", 0.0, 0.0001)

        self.contlim_args = ["F_0", "B", "c3", "c4", "alpha", "beta", "ellphys"]
        self.finbeta_args = ["F_0", "B", "c3", "c4", "alpha", "beta", "ellphys", "gamma_1", "gamma_2"]

    def degrees_of_freedom(self):
        """
        Return the degrees of freedom of the model. Overload this method if different if different
        """
        logging.info("data")
        datapoints = [d.shape[0] for d in self.data.values()]
        ndata = datapoints[0] * 2  # Twice the dof because combined fit to two data sets

        fixed_parms = [p for p in self.params if "fix" in p and self.params[p]]
        Nparams = inspect.getargspec(self.m).args[1:]
        Nfree_params = len(Nparams) - len(fixed_parms)
        dof = float(ndata - Nfree_params)
        logging.info("DOF {}, data {}, free {}".format(dof, ndata, Nfree_params))
        return dof

    def m(self, xi, F_0, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2):

        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi")**2)
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)
        delta_Mss = Mss - phys_Mss

        asqr = (self.consts["lat"]**2)

        delta_M = (1+gamma_1*asqr+gamma_s1*delta_Mss)
        delta_F = (1+gamma_2*asqr+gamma_s2*delta_Mss)


        Fpi = delta_F * F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi)) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2
        Mpisqr = delta_M * 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                                (c4/F_0 - 1.0/3.0 * (ellphys+16))*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2

        return Mpisqr, Fpi

    def plot_fit(self, xi, F_0, B, c3, c4, alpha, beta, ellphys):
        """ Override the normal plotter, this needs to not call m in this case """

        if self.plotmode == "fpi":
            return F_0 * (1 - xi*np.log(xi) + 5.0/4.0*(xi*np.log(xi))**2 + 1/6.0*(ellphys+53.0/2.0)*xi*xi*np.log(xi)) + c4*xi*(1-5*xi*np.log(xi)) + beta*xi**2
        else:
            return 2*B*(1.0+0.5*xi*np.log(xi) +7.0/8.0*(xi*np.log(xi))**2+
                        (c4/F_0 - 1.0/3.0 * (ellphys+16))*np.log(xi)*xi**2) + c3*xi*(1-5*xi*np.log(xi)) + alpha*xi**2

    def sqr_diff(self, F_0, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2):

        xi = self.bstrapdata("xi")
        M, F = self.m(xi, F_0, B, c3, c4, alpha, beta, ellphys, gamma_1, gamma_2, gamma_s1, gamma_s2)
        mdata = self.bstrapdata("mpi")**2 / self.consts["renorm_qmass"]
        sqr_diff1 = (mdata - M)**2
        sqr_diff2 = (self.bstrapdata("fpi") - F)**2
        return np.sum(sqr_diff1 / self.var1) + np.sum(sqr_diff2 / self.var2)
