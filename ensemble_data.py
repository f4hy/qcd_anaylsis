#!/usr/bin/env python2
import logging
import pandas as pd
import re
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass

class MissingData(RuntimeError):
    pass


class ensemble_data(object):



    def __init__(self, ensamble_info, fit_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot", decay_file_wildcard="decay_constants/*_fixed_0_1-1_1/decay_*_decayconstant_*.boot"):

        self.dp = ensamble_info

        self.scale = scale[self.dp.beta]

        self.fit_file_wildcard = fit_file_wildcard
        self.decay_file_wildcard = decay_file_wildcard

        self.mass_data = {}
        self.decay_data = {}
        self.fit_mass_data = None
        self.fit_amp_data = None
        self.xi_data = None




    def narrow_wildcard(self, fit_file_wildcard, flavor=None, operator="PP"):
        import glob
        dp = self.dp

        if flavor is None:
            flavor_str = dp.flavor_string
        else:
            flavor_str = flavor

        smearing = dp.smearing
        if flavor == "xi":
            smearing = None

        heavyness = dp.heavyness
        if heavyness == "ll":
            heavyness = None
        if heavyness is not None:
            heavyness = "_"+heavyness

        if flavor == "heavy-heavy":
            smearing = "0"


        fitdatafiles = glob.glob(fit_file_wildcard.strip("'\""))
        fitdatafiles = [f for f in fitdatafiles if "axial" not in f]

        logging.debug(fitdatafiles)
        search_params = [operator, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, flavor_str, smearing]
        if dp.heavyness != "ll":
            if flavor is None:
                search_params = [operator, heavyness, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, flavor_str, smearing]
            elif "heavy" in flavor:
                search_params = [operator, heavyness, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, flavor_str, smearing]
        logging.debug("{}".format(search_params))
        for i in search_params:
            logging.debug(fitdatafiles)
            logging.debug(i)

            if i is not None:
                fitdatafiles = [f for f in fitdatafiles if str(i) in f ]
            logging.debug(fitdatafiles)
            logging.debug("")
        if len(fitdatafiles) != 1:
            logging.critical("Unique fit file not found!")
            logging.error("looking for : {} {}".format(fit_file_wildcard, dp))
            logging.error("found: {}".format(fitdatafiles))
            raise MissingData("Unique fit file not found!")

        logging.info("narrowed to file {}".format(fitdatafiles[0]))
        return fitdatafiles[0]

    def get_mass(self, flavor, wild=None, op="PP"):
        if wild is None:
            wild = self.fit_file_wildcard

        if (flavor, wild, op) in self.mass_data.keys():
            return self.mass_data[(flavor, wild, op)]

        mass_file = self.narrow_wildcard(wild, flavor=flavor, operator=op)

        with open(mass_file) as fitfile:
            df = pd.read_csv(fitfile,comment='#', names=["config", "mass", "amp1", "amp2"])
            self.mass_data[(flavor, wild, op)] = df.mass
        return self.mass_data[(flavor, wild, op)]


    def get_decay(self, flavor):
        if flavor in self.decay_data.keys():
            return self.decay_data[flavor]


        decay_file = self.narrow_wildcard(self.decay_file_wildcard, flavor=flavor)
        with open(decay_file) as decayfile:
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

    def eta_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("s-s")
        return self.get_mass("s-s")


    def D_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-ud")
        return self.get_mass("heavy-ud")

    def HH_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild="SymDW_sHtTanh_b2.0_smr3_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_PP.boot")
        return self.get_mass("heavy-heavy", wild="SymDW_sHtTanh_b2.0_smr3_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_PP.boot")

    def HHv_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild="SymDW_sHtTanh_b2.0_smr3_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_vectorave.boot", op="vectorave")
        return self.get_mass("heavy-heavy", wild="SymDW_sHtTanh_b2.0_smr3_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_vectorave.boot", op="vectorave")



    def Ds_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-s")
        return self.get_mass("heavy-s")

    def xi(self, scaled=False):
        if self.xi_data is None:
            mpi = self.pion_mass(scaled=scaled)
            fpi = self.fpi(scaled=scaled)
            xi = ((mpi**2) / (8.0 * (np.pi**2)*(fpi**2)))
            self.xi_data = xi

        return self.xi_data

    def fpi(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("ud-ud")
        return self.get_decay("ud-ud")

    def fK(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("ud-s")
        return self.get_decay("ud-s")


    def fD(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-ud")
        return self.get_decay("heavy-ud")

    def fDs(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-s")
        return self.get_decay("heavy-s")

    def fDsbyfD(self, scaled=False):
        if scaled:
            return (self.scale*self.get_decay("heavy-s"))/(self.scale*self.get_decay("heavy-ud"))
        return self.get_decay("heavy-s")/self.get_decay("heavy-ud")


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

    filename = "SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030/simul_fixed_fit_uncorrelated_ud-ud/simul_fit_uncorrelated_ll_ud-ud_0_1-1_1_PP.boot"

    dp = data_params(filename)

    ed = ensemble_data(dp)

    print ed.fit_mass()
    print ed.decay_const()

    print ed.get_mass("ud-ud")
    print ed.get_mass("ud-s")
    print ed.kaon_mass()

if __name__ == "__main__":

    test()
