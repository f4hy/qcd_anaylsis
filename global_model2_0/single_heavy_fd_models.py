from global_fit_model2_0 import Model
import physical_values as pv
import numpy as np
import logging


class linear_FD_in_mpi(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["fD"] = self.make_array("fD")

        try:
            self.var = self.data["fD"].var(1)
        except IndexError:
            self.var = np.NaN

        self.update_paramdict("a", 0.1, 0.2)
        self.update_paramdict("b", 0.1, 0.2)
        self.contlim_args = ["a", "b"]

    def m(self, x, a, b):
        return a + b * (x)

    def sqr_diff(self, a, b):
        x = self.bstrapdata("mpi")
        M = self.m(x, a, b)
        data = self.bstrapdata("fD")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class poly_fDs_mpi(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["fDs"] = self.make_array("fDs")

        try:
            self.var = self.data["fDs"].var(1)
        except IndexError:
            self.var = np.NaN

        self.update_paramdict("a", 0.1, 0.2)
        self.update_paramdict("b", 0.1, 0.2)
        self.update_paramdict("c", 0.1, 0.2)
        self.contlim_args = ["a", "b", "c"]

    def m(self, x, a, b, c):
        return a + b * (x) + c * (x**2)

    def sqr_diff(self, a, b, c):

        x = self.bstrapdata("mpi")
        M = self.m(x, a, b, c)
        data = self.bstrapdata("fDs")
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class poly_fDssqrtmDs_a(Model):

    def __init__(self, ensemble_datas, options, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["fDs"] = self.make_array("fDs")
        self.data["mDs"] = self.make_array("Ds_mass")

        try:
            self.var = (self.data["fDs"] * np.sqrt(self.data["mDs"])).var(1)
        except IndexError:
            self.var = np.NaN

        # logging.info(self.bstrapdata("fDs")*np.sqrt(self.bstrapdata("mDs")))
        self.update_paramdict("a", 10600, 100.0)
        self.update_paramdict("b", -8000, 7000)
        self.update_paramdict("c", -30000.0, 30000.0)
        self.update_paramdict("gamma_p", 0.001, 0.02)
        self.contlim_args = ["a", "b", "c"]

    def m(self, x, a, b, c, gamma_p=0.0):
        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)
        return (1.0 + gamma_p * delta_mpisqr) * (a + b * (x) + c * (x**2))

    def sqr_diff(self, a, b, c, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x, a, b, c, gamma_p)
        data = self.bstrapdata("fDs") * np.sqrt(self.bstrapdata("mDs"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class poly_fhssqrtmhs_a(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")

        try:
            self.var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        except IndexError:
            self.var = np.NaN

        logging.info(self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs")))
        self.update_paramdict("a", 10600, 100.0)
        self.update_paramdict("b", 8000, 7000)
        self.update_paramdict("c", -30000.0, -30000.0)
        self.update_paramdict("gamma_p", 1.38340202243e-07, 0.02)
        self.contlim_args = ["a", "b", "c"]

    def m(self, x, a, b, c, gamma_p=0.0):
        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)
        return (1.0 + gamma_p * delta_mpisqr) * (a + b * (x) + c * (x**2))

    def sqr_diff(self, a, b, c, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x, a, b, c, gamma_p)
        data = self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class poly_fhssqrtmhs_a_m0(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0", **kargs)


class poly_fhssqrtmhs_a_m1(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1", **kargs)


class poly_fhssqrtmhs_a_m2(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2", **kargs)


class poly_fhssqrtmhs_a_m3(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3", **kargs)


class poly_fhssqrtmhs_a_m4(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4", **kargs)


class poly_fhssqrtmhs_a_m5(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5", **kargs)


class poly_fhssqrtmhs_hqet_a_m0(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0", hqet=True, **kargs)


class poly_fhssqrtmhs_hqet_a_m1(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1", hqet=True, **kargs)


class poly_fhssqrtmhs_hqet_a_m2(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2", hqet=True, **kargs)


class poly_fhssqrtmhs_hqet_a_m3(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3", hqet=True, **kargs)


class poly_fhssqrtmhs_hqet_a_m4(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4", hqet=True, **kargs)


class poly_fhssqrtmhs_hqet_a_m5(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5", hqet=True, **kargs)


class linear_fhssqrtmhs_a(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False, **kargs):

        Model.__init__(self, ensemble_datas, options, **kargs)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")

        try:
            self.var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        except IndexError:
            self.var = np.NaN

        logging.info(self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs")))
        self.update_paramdict("a", 10600, 100.0)
        self.update_paramdict("b", 8000, 7000)
        self.update_paramdict("gamma_p", 1.38340202243e-07, 0.02)
        self.contlim_args = ["a", "b"]

    def m(self, x, a, b, gamma_p=0.0):
        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)
        return (1.0 + gamma_p * delta_mpisqr) * (a + b * (x))

    def sqr_diff(self, a, b, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x, a, b, gamma_p)
        data = self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs"))
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / self.var)


class linear_fhssqrtmhs_hqet_a_m0(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0", hqet=True, **kargs)


class linear_fhssqrtmhs_hqet_a_m1(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1", hqet=True, **kargs)


class linear_fhssqrtmhs_hqet_a_m2(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2", hqet=True, **kargs)


class linear_fhssqrtmhs_hqet_a_m3(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3", hqet=True, **kargs)


class linear_fhssqrtmhs_hqet_a_m4(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4", hqet=True, **kargs)


class linear_fhssqrtmhs_hqet_a_m5(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options, **kargs):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5", hqet=True, **kargs)
