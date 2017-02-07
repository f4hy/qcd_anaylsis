#!/usr/bin/env python2
import logging
import numpy as np

from alpha_s import get_alpha
from all_ensemble_data import MissingData
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

        # Model object is created without data, (maybe for plotting)
        # set some defaults
        self.consts = {"a": 0.0, "lat": 0.0, "alphas": 1.0, "m1": 0.0, "m2": 0.0}
        if len(self.eds) > 1:
            # one data per ensemble
            datas = self.eds
            if self.each_heavy:
                # We need N data for each ensemble
                datas = [ed for ed in self.eds for m in ed.selected_heavies]
                self.consts['m1'] = np.array([ed.ep.m12s[m][0]*ed.ep.scale for ed in self.eds
                                              for m in ed.selected_heavies])
                self.consts['m2'] = np.array([ed.ep.m12s[m][1]*ed.ep.scale for ed in self.eds
                                              for m in ed.selected_heavies])
            self.consts["a"] = np.array([ed.ep.a_gev for ed in datas])
            self.consts["lat"] = np.array([ed.ep.latspacing for ed in datas])
            self.consts["alphas"] = np.array([get_alpha(ed.ep.scale) for ed in datas])

        self.data = {}
        self.options = options

        self.bootstrap = None

        self.params = {}

        self.label = ""

        logging.info("Data read")

    def make_array(self, fun_name, **args):
        logging.debug("making array for {} with {}".format(fun_name, args))

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

        logging.debug("made array for {} of size {} vs {}".format(fun_name, len(ls), len(self.eds)))
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
