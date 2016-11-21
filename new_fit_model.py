#!/usr/bin/env python2
import logging
import numpy as np

from residualmasses import residual_mass, residual_mass_errors


# from ensemble_data import ensemble_data, MissingData

from msbar_convert import get_matm

from alpha_s import get_alpha

from all_ensemble_data import ensemble_data, MissingData, NoStrangeInterp

import physical_values as pv

import inspect

class Model(object):

    def __init__(self, ensemble_datas, options):

        self.eds = ensemble_datas
        self.bstrap_data = {}

        self.consts = {"a": np.array([ed.ep.a_gev for ed in self.eds])}
        self.data = {}

        self.options = options

        # #self.a = np.array([ed.ep.latspacing for ed in self.eds])
        # self.a = np.array([1.0/ed.ep.scale for ed in self.eds])
        # self.a = np.array([ed.ep.a_gev for ed in self.eds])

        dps = self.data.keys()

        self.bootstrap = None

        self.params = {}

        # def safe_array(d):
        #     try:
        #         return np.array(d)
        #     except MissingData:
        #         return np.array(float("NAN"))

        # def make_array(funname, **params):
        #     try:
        #         return np.array([getattr(data[dp], funname)(**params)
        #                          for dp in dps])
        #     except MissingData:
        #         logging.warning("Missing {} data".format(funname))
        #         return None

        # logging.info("buidling data")


        logging.info("Data read")

    def make_array(self, fun_name, **args):
        logging.info("making array for {} with {}".format(fun_name, args))
        def choose_fun(ed):
            d = dict(inspect.getmembers(ed,inspect.ismethod))
            return d[fun_name]

        ls = []
        added = []
        must_be_removed = []
        for ed in self.eds:
            try:
                data = choose_fun(ed)(**args)
                ls.append(data)
                added.append(ed.ep)
            except MissingData:
                logging.warn("ensemble {} is missing {}".format(ed.ep, fun_name))
                enum = self.eds.index(ed)
                must_be_removed.append(enum)
        for i in must_be_removed:
            N = len(self.eds)
            for n,d in self.data.iteritems():
                if d.shape[0] == N:
                    logging.warn("required to remove some data")
                    self.data[n] = np.delete(d, i, 0)
            for n,d in self.consts.iteritems():
                if len(d) == N:
                    logging.warn("required to remove some consts {} {}".format(N, len(d)))
                    self.consts[n] = np.delete(d, i)
            del self.eds[i]

        logging.info("made array for {} of size {} vs {}".format(fun_name, len(ls), len(self.eds)))
        return np.array( ls )


    # def bstrapdata(self, d):
    #     if self.bootstrap is None or self.bootstrap == "mean":
    #         return d.mean(1)
    #     else:
    #         return d[:, self.bootstrap]

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
        paramdict["error_"+parameter] = err
        paramdict["fix_"+parameter] = fix
        # if parameter in self.options.zero:
        #     logging.info("option passed to set {} to zero".format(parameter))
        #     logging.info("zero {self.options.zero}")
        # if fixzero or parameter in self.options.zero:
        if fixzero:
            paramdict[parameter] = 0.0
            paramdict["fix_"+parameter] = True
        if limits:
            paramdict["limit_"+parameter] = limits
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
        return a+b*(x)

    def sqr_diff(self, a, b):
        x = self.bstrapdata("mpi")
        M = self.m(x,a,b)
        data = self.bstrapdata("fD")
        var = self.data["fD"].var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

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
        return a+b*(x) + c*(x**2)

    def sqr_diff(self, a, b, c):

        x = self.bstrapdata("mpi")
        M = self.m(x,a,b,c)
        data = self.bstrapdata("fDs")
        var = self.data["fDs"].var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

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
        delta_mpisqr = (self.bstrapdata("mpi")**2)-(pv.phys_pion**2)
        return (1.0+gamma_p*delta_mpisqr)*(a + b*(x) + c*(x**2))

    def sqr_diff(self, a, b, c, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x,a,b,c, gamma_p)
        data = self.bstrapdata("fDs")*np.sqrt(self.bstrapdata("mDs"))
        var = (self.data["fDs"]*np.sqrt(self.data["mDs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)


class poly_fhssqrtmhs_a(Model):

    def __init__(self, ensemble_datas, options, heavy="m0", hqet=False):

        Model.__init__(self, ensemble_datas, options)
        self.data["fhs"] = self.make_array("fDs", heavy=heavy, div=hqet)
        self.data["mhs"] = self.make_array("Ds_mass", heavy=heavy, div=hqet)
        self.data["mpi"] = self.make_array("pion_mass")

        logging.info(self.bstrapdata("fhs")*np.sqrt(self.bstrapdata("mhs")))
        self.update_paramdict("a", 10600, 100.0)
        self.update_paramdict("b", 8000, 7000)
        self.update_paramdict("c", -30000.0, -30000.0)
        self.update_paramdict("gamma_p", 1.38340202243e-07, 0.02)
        self.contlim_args = ["a", "b", "c"]

    def m(self, x, a, b, c, gamma_p=0.0):
        delta_mpisqr = (self.bstrapdata("mpi")**2)-(pv.phys_pion**2)
        return (1.0+gamma_p*delta_mpisqr)*(a + b*(x) + c*(x**2))

    def sqr_diff(self, a, b, c, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x,a,b,c, gamma_p)
        data = self.bstrapdata("fhs")*np.sqrt(self.bstrapdata("mhs"))
        var = (self.data["fhs"]*np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

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

        logging.info(self.bstrapdata("fhs")*np.sqrt(self.bstrapdata("mhs")))
        self.update_paramdict("a", 10600, 100.0)
        self.update_paramdict("b", 8000, 7000)
        self.update_paramdict("gamma_p", 1.38340202243e-07, 0.02)
        self.contlim_args = ["a", "b"]

    def m(self, x, a, b, gamma_p=0.0):
        delta_mpisqr = (self.bstrapdata("mpi")**2)-(pv.phys_pion**2)
        return (1.0+gamma_p*delta_mpisqr)*(a + b*(x) )

    def sqr_diff(self, a, b, gamma_p):

        x = self.consts["a"]**2
        M = self.m(x,a,b, gamma_p)
        data = self.bstrapdata("fhs")*np.sqrt(self.bstrapdata("mhs"))
        var = (self.data["fhs"]*np.sqrt(self.data["mhs"])).var(1)
        sqr_diff = (data - M)**2
        return np.sum(sqr_diff/var)

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
