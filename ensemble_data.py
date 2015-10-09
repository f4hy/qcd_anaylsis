#!/usr/bin/env python2
import logging
import pandas as pd
import re
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass

class ensemble_data(object):



    def __init__(self, ensamble_info, fit_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot", decay_file_wildcard="decay_constants/*light_fixed_0_2-2_2/decay_*_decayconstant_*.boot", xi_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/xi/xi.out"):

        self.dp = ensamble_info

        self.scale = scale[self.dp.beta]

        self.fit_file_wildcard = fit_file_wildcard
        self.decay_file_wildcard = decay_file_wildcard
        self.xi_file_wildcard = xi_file_wildcard


        self.fit_file = self.narrow_wildcard(fit_file_wildcard)
        self.decay_file = self.narrow_wildcard(decay_file_wildcard)

        self.mass_data = {}
        self.decay_data = {}
        self.fit_mass_data = None
        self.fit_amp_data = None
        self.xi_data = None




    def narrow_wildcard(self, fit_file_wildcard, flavor=None):
        import glob
        dp = self.dp

        if flavor is None:
            flavor_str = dp.flavor_string
        else:
            flavor_str = flavor

        smearing = dp.smearing
        if flavor == "xi":
            smearing = None


        fitdatafiles = glob.glob(fit_file_wildcard.strip("'\""))
        for i in [dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, flavor_str, smearing]:
            if i is not None:
                fitdatafiles = [f for f in fitdatafiles if str(i) in f ]
        if len(fitdatafiles) != 1:
            logging.critical("Unique fit file not found!")
            logging.error("found: {}".format(fitdatafiles))
            raise SystemExit("Unique fit file not found!")


        return fitdatafiles[0]

    def get_mass(self, flavor):

        if flavor in self.mass_data.keys():
            return self.mass_data[flavor]

        mass_file = self.narrow_wildcard(self.fit_file_wildcard, flavor=flavor)

        with open(mass_file) as fitfile:
            df = pd.read_csv(fitfile,comment='#', names=["config", "mass", "amp1", "amp2"])
            self.mass_data[flavor] = df.mass
        return self.mass_data[flavor]


    def get_decay(self, flavor):
        if flavor in self.decay_data.keys():
            return self.decay_data[flavor]

        decay_file = self.narrow_wildcard(self.decay_file_wildcard, flavor=flavor)

        with open(self.decay_file) as decayfile:
            df = pd.read_csv(decayfile,comment='#', names=["decay"])
            self.decay_data[flavor] = df.decay
        return self.decay_data[flavor]


    def pion_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("ud-ud")
        return self.get_mass("ud-ud")

    def kaon_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("ud-s")
        return self.get_mass("ud-s")

    def xi(self, scaled=False):
        if self.xi_data is None:
            xi_file = self.narrow_wildcard(self.xi_file_wildcard, flavor="xi")
            with open(xi_file) as xifile:
                df = pd.read_csv(xifile,comment='#', names=["xi"])
                self.xi_data = df.xi
        if scaled:
            return self.scale*self.xi_data
        return self.xi_data

    def fpi(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("ud-ud")
        return self.get_decay("ud-ud")


    def fit_mass(self):

        if self.fit_mass_data is None:
            with open(self.fit_file) as fitfile:
                df = pd.read_csv(fitfile,comment='#', names=["config", "mass", "amp1", "amp2"])
                self.fit_mass_data = df.mass
        return self.fit_mass_data

    def fit_amp(self):
        if self.fit_amp_data is None:
            with open(self.fit_file) as fitfile:
                df = pd.read_csv(fitfile,comment='#', names=["config", "mass", "amp1", "amp2"])
                self.fit_amp_data = df.amp
        return self.fit_amp_data

    def decay_const(self):
        if self.decay_data is None:
            with open(self.decay_file) as decayfile:
                df = pd.read_csv(decayfile,comment='#', names=["decay"])
                self.decay_data = df.decay
        return self.decay_data


def test():


    fit_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot"
    decay_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot"

    filename = "SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030/simul_fixed_fit_uncorrelated_ud-ud/simul_fit_uncorrelated_ll_ud-ud_0_2-2_2_PP.boot"

    dp = data_params(filename)

    ed = ensemble_data(dp)

    print ed.fit_mass()
    print ed.decay_const()

    print ed.get_mass("ud-ud")
    print ed.get_mass("ud-s")
    print ed.kaon_mass()

if __name__ == "__main__":

    test()
