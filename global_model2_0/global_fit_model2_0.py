#!/usr/bin/env python2
import logging
import numpy as np

from alpha_s import get_alpha
from all_ensemble_data import MissingData
import physical_values as pv
import inspect


class Model(object):

    def __init__(self, ensemble_datas, options, each_heavy=False):
        """ Initialize the model object with constants and placeholders

        The 'each_heavy' parameter treats each heavy mass within an ensemble as a seperate ensemble

        """

        self.each_heavy = each_heavy
        self.eds = ensemble_datas
        self.bstrap_data = {}
        try:
            self.hqm_cutoff = options.hqm_cutoff
        except AttributeError:
            self.hqm_cutoff = 100000000.
        if self.each_heavy:
            for ed in self.eds:
                ed.selected_heavies = []
                for h, m in ed.ep.heavies.iteritems():
                    if m < options.hqm_cutoff:
                        ed.selected_heavies.append(h)
        if len(self.eds) > 1:
            # one data per ensemble
            datas = self.eds
            if self.each_heavy:
                # We need N data for each ensemble
                datas = [ed for ed in self.eds for _ in ed.selected_heavies]
            self.consts = {"a": np.array([ed.ep.a_gev for ed in datas]),
                           "lat": np.array([ed.ep.latspacing for ed in datas]),
                           "alphas": np.array([get_alpha(ed.ep.scale) for ed in datas])}

        else:
            # Model object is created without data, (maybe for plotting)
            # set some defaults
            self.consts = {"a": 0.0, "lat": 0.0, "alphas": 1.0}

        self.data = {}
        self.options = options

        self.bootstrap = None

        self.params = {}

        self.label = ""

        logging.info("Data read")

    def make_array(self, fun_name, **args):
        logging.info("making array for {} with {}".format(fun_name, args))

        def choose_fun(ed):
            d = dict(inspect.getmembers(ed, inspect.ismethod))

            return d[fun_name]

        ls = []
        added = []
        must_be_removed = []
        for ed in self.eds:
            logging.debug("searching {}".format(ed))
            try:
                data = choose_fun(ed)(**args)
                keys = data.keys()
                if self.each_heavy:
                    if isinstance(keys[0], basestring) and any("m0" in k for k in keys): # noqa
                        for m in ed.selected_heavies:
                            for k in (j for j in data.keys() if m in j):
                                ls.append(data[k])
                    else:
                        for m in ed.selected_heavies:
                            ls.append(data)
                else:
                    ls.append(data)
                added.append(ed.ep)
            except MissingData:
                logging.warn("ensemble {} is missing {}".format(ed.ep, fun_name))
                enum = self.eds.index(ed)
                must_be_removed.append(enum)
                if self.each_heavy:
                    raise RuntimeError("missing data not properly handled if ploting each heavy")
        for i in must_be_removed:
            N = len(self.eds)
            for n, d in self.data.iteritems():
                if d.shape[0] == N:
                    logging.warn("required to remove some data")
                    self.data[n] = np.delete(d, i, 0)
            for n, d in self.consts.iteritems():
                if len(d) == N:
                    logging.warn("required to remove some consts {} {}".format(N, len(d)))
                    self.consts[n] = np.delete(d, i)
            del self.eds[i]

        logging.info("made array for {} of size {} vs {}".format(fun_name, len(ls), len(self.eds)))
        return np.array(ls)

    def bstrapdata(self, data_string):
        if len(self.eds) < 1:
            logging.debug("running in reference mode, just give zero")
            return 0.0
        d = self.data[data_string]
        if self.bootstrap is None or self.bootstrap == "mean":
            return d.mean(1)
        else:
            return d[:, self.bootstrap]

    def set_bootstrap(self, b):
        self.bootstrap = b

    def update_paramdict(self, parameter, guess, err, limits=None, fix=False, fixzero=False):

        paramdict = {parameter: guess}
        paramdict["error_" + parameter] = err
        paramdict["fix_" + parameter] = fix
        # if parameter in self.options.zero:
        #     logging.info("option passed to set {} to zero".format(parameter))
        #     logging.info("zero {self.options.zero}")
        # if fixzero or parameter in self.options.zero:
        if fixzero:
            paramdict[parameter] = 0.0
            paramdict["fix_" + parameter] = True
        if limits:
            paramdict["limit_" + parameter] = limits
        self.params.update(paramdict)


class linear_FD_in_mpi(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")
        self.data["fD"] = self.make_array("fD")

        self.update_paramdict("a", 0.1, 0.2)
        self.update_paramdict("b", 0.1, 0.2)
        self.contlim_args = ["a", "b"]

    def m(self, x, a, b):
        return a + b * (x)

    def sqr_diff(self, a, b):
        x = self.bstrapdata("mpi")
        M = self.m(x, a, b)
        data = self.bstrapdata("fD")
        var = self.data["fD"].var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class poly_fDs_mpi(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["fDs"] = self.make_array("fDs")

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
        var = self.data["fDs"].var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class poly_fDssqrtmDs_a(Model):

    def __init__(self, ensemble_datas, options):

        Model.__init__(self, ensemble_datas, options)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["fDs"] = self.make_array("fDs")
        self.data["mDs"] = self.make_array("Ds_mass")

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
        var = (self.data["fDs"] * np.sqrt(self.data["mDs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class poly_fhssqrtmhs_a(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")

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
        var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class poly_fhssqrtmhs_a_m0(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0")


class poly_fhssqrtmhs_a_m1(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1")


class poly_fhssqrtmhs_a_m2(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2")


class poly_fhssqrtmhs_a_m3(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3")


class poly_fhssqrtmhs_a_m4(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4")


class poly_fhssqrtmhs_a_m5(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5")


class poly_fhssqrtmhs_hqet_a_m0(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0", hqet=True)


class poly_fhssqrtmhs_hqet_a_m1(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1", hqet=True)


class poly_fhssqrtmhs_hqet_a_m2(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2", hqet=True)


class poly_fhssqrtmhs_hqet_a_m3(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3", hqet=True)


class poly_fhssqrtmhs_hqet_a_m4(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4", hqet=True)


class poly_fhssqrtmhs_hqet_a_m5(poly_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        poly_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5", hqet=True)


class linear_fhssqrtmhs_a(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")

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
        var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class linear_fhssqrtmhs_hqet_a_m0(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m0", hqet=True)


class linear_fhssqrtmhs_hqet_a_m1(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m1", hqet=True)


class linear_fhssqrtmhs_hqet_a_m2(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m2", hqet=True)


class linear_fhssqrtmhs_hqet_a_m3(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m3", hqet=True)


class linear_fhssqrtmhs_hqet_a_m4(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m4", hqet=True)


class linear_fhssqrtmhs_hqet_a_m5(linear_fhssqrtmhs_a):
    def __init__(self, ensemble_datas, options):
        linear_fhssqrtmhs_a.__init__(self, ensemble_datas, options, heavy="m5", hqet=True)


class linear_fD_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhl"] = self.make_array("fD", heavy=heavy, div=hqet)
        self.data["mhl"] = self.make_array("D_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

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
        var = (self.data["fhl"]).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class linear_fDs_mpi(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

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
        var = (self.data["fhs"]).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class fdsqrtm_chiral_dmss(Model):

    def __init__(self, ensemble_datas, options, hqet=False):

        Model.__init__(self, ensemble_datas, options, each_heavy=True)
        self.data["mhl"] = self.make_array("get_mass", flavor="heavy-ud", div=hqet)
        self.data["fhl"] = self.make_array("fhl", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

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
        delta_mpisqr = self.bstrapdata("mpi") - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi"))
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)

        delta_Mss = Mss - phys_Mss
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
        var = (self.data["fhl"] * np.sqrt(self.data["mhl"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


fdsqrtm_chiral_dmss_HQET = fdsqrtm_chiral_dmss
fdsqrtm_HQET_matched = fdsqrtm_chiral_dmss


class fdsqrtm_HQET_matched_alphas(Model):

    def __init__(self, ensemble_datas, options, hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhs"] = self.make_array("fhl", div=hqet)
        self.data["mhl"] = self.make_array("get_mass", flavor="heavy-ud", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", 0.0, 0.01)
        self.update_paramdict("eta", 0.0, 0.01)
        self.update_paramdict("mu", 0.0, 0.01)
        self.update_paramdict("b", 0.0, 0.01)
        self.update_paramdict("delta_S", 0.0, 0.01)
        self.update_paramdict("Fsqrtm_inf", 0.0, 0.01)

        self.contlim_args = ["Fsqrtm_inf", "C1", "C2"]
        self.finbeta_args = ["Fsqrtm_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fsqrtm_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        delta_mpisqr = self.bstrapdata("mpi") - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi"))
        phys_Mss = (2.0 * (pv.phys_kaon**2)) - (pv.phys_pion**2)

        delta_mpisqr = (self.bstrapdata("mpi")**2) - (pv.phys_pion**2)

        delta_Mss = Mss - phys_Mss
        asqr = (self.consts["a"]**2)
        asqr = (self.consts["lat"]**2)
        alphas = self.consts["alphas"]
        deltas = (1.0 + delta_S * delta_Mss + b * delta_mpisqr +
                  mu * asqr + eta * asqr / x + gamma * alphas * (asqr) / (x**2))
        # deltas = 1
        poly = Fsqrtm_inf * (1.0 + C1 * x + C2 * x**2)
        M = deltas * poly
        return M

    def sqr_diff(self, FDsphys, b, gamma_1, gamma_s1):

        x = self.bstrapdata("mpi")**2
        M = self.m(x, FDsphys, b, gamma_1, gamma_s1)
        data = self.bstrapdata("fhl")
        var = (self.data["fhl"]).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


class fdssqrtms_chiral_dmss(Model):

    def __init__(self, ensemble_datas, options, hqet=False):

        Model.__init__(self, ensemble_datas, options, each_heavy=True)
        self.data["mhs"] = self.make_array("get_mass", flavor="heavy-s", div=hqet)
        self.data["fhs"] = self.make_array("fhs", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

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
        delta_mpisqr = self.bstrapdata("mpi") - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi"))
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
        var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)


fdssqrtms_HQET_matched = fdssqrtms_chiral_dmss
fdssqrtms_HQET_matched_alphas = fdssqrtms_chiral_dmss


class fdssqrtms_HQET_matched_alphas(Model):

    def __init__(self, ensemble_datas, options, hqet=True):

        hqet = True
        Model.__init__(self, ensemble_datas, options, each_heavy=True)
        self.data["fhs"] = self.make_array("fhs", div=hqet, matched=True)
        self.data["mhs"] = self.make_array("get_mass", flavor="heavy-s", div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")
        self.data["mK"] = self.make_array("kaon_mass")

        self.label = "Continuum fit"

        self.update_paramdict("C1", -100.0, 20.0)
        self.update_paramdict("C2", 100000.0, 10000.0)
        self.update_paramdict("gamma", 0.0, 0.01)
        self.update_paramdict("eta", 0.0, 0.01)
        self.update_paramdict("mu", 0.0, 0.01)
        self.update_paramdict("b", 0.0, 0.01)
        self.update_paramdict("delta_S", 0.0, 0.01)
        self.update_paramdict("Fssqrtms_inf", 180000.0, 0.01)

        self.contlim_args = ["Fssqrtms_inf", "C1", "C2"]
        self.finbeta_args = ["Fssqrtms_inf", "C1", "C2", "mu", "eta", "gamma"]

    def m(self, x, Fssqrtms_inf, C1, C2, mu=0, eta=0, gamma=0, b=0, delta_S=0):
        x = x
        delta_mpisqr = self.bstrapdata("mpi") - (pv.phys_pion**2)
        Mss = (2.0 * self.bstrapdata("mK")**2 - self.bstrapdata("mpi"))
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

        x = 1.0 / self.bstrapdata("mhs")
        M = self.m(x, Fssqrtms_inf, C1, C2, mu, eta, gamma, b, delta_S)
        data = self.bstrapdata("fhs") * np.sqrt(self.bstrapdata("mhs"))
        var = (self.data["fhs"] * np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff / var)
