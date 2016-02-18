#!/usr/bin/env python2
import logging
import pandas as pd
import re
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
import glob

class MissingData(RuntimeError):
    pass


class NoStrangeInterp(MissingData):
    pass


class ensemble_data(object):

    def __init__(self, ensamble_info,
                 fit_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/simul_?????_fit_uncorrelated_*/*.boot",
                 decay_file_wildcard="decay_constants/*_fixed_0_1-1_1/*decay_*_decayconstant_*.boot",
                 interpstrange=False):

        self.dp = ensamble_info

        self.scale = scale[self.dp.beta]

        self.fit_file_wildcard = fit_file_wildcard
        self.decay_file_wildcard = decay_file_wildcard

        self.interpstrange = interpstrange

    def narrow_wildcard(self, fit_file_wildcard, flavor=None, operator="PP", axial=False):
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

        prefix = ""
        if self.interpstrange:
            prefix = "interpolated_strange/"
            if dp.ud_mass < 0.0036 or dp.beta == "4.47":
                logging.warn("No strange interpolated data for {}".format(dp))
                raise NoStrangeInterp("No strange interpolated data for {}".format(dp))

        fitdatafiles = glob.glob(prefix+fit_file_wildcard.strip("'\""))

        logging.debug(fitdatafiles)
        search_params = [operator, flavor_str, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, smearing]

        if dp.heavyness != "ll":
            if flavor is None:
                search_params = [operator, flavor_str, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, smearing, heavyness]
            elif "heavy" in flavor:
                search_params = [operator, flavor_str, dp.ud_mass, dp.s_mass, dp.latsize, dp.beta, smearing, heavyness]

        if self.interpstrange:
            search_params[3] = "interpolated"

        logging.debug("{}".format(search_params))
        for i in search_params:
            logging.debug(fitdatafiles)
            logging.debug(i)

            if i is not None:
                fitdatafiles = [f for f in fitdatafiles if str(i) in f]
            logging.debug(fitdatafiles)
            logging.debug("")
        if len(fitdatafiles) != 1:
            logging.critical("Unique fit file not found!")
            logging.error("looking for : {} {}".format(prefix+fit_file_wildcard, dp, "_".join(map(str, search_params))))
            logging.error("found: {}".format(fitdatafiles))
            raise MissingData("Unique fit file not found!")

        logging.info("narrowed to file {}".format(fitdatafiles[0]))
        return fitdatafiles[0]

    def get_mass(self, flavor, wild=None, op="PP"):
        if wild is None:
            wild = self.fit_file_wildcard

        mass_file = self.narrow_wildcard(wild, flavor=flavor, operator=op)

        with open(mass_file) as fitfile:
            df = pd.read_csv(fitfile, comment='#', names=["config", "mass", "amp1", "amp2"])
            return df.mass

    def get_amps(self, flavor, wild=None, op="PP"):
        if wild is None:
            wild = self.fit_file_wildcard

        amp_file = self.narrow_wildcard(wild, flavor=flavor, operator=op)

        with open(amp_file) as fitfile:
            df = pd.read_csv(fitfile, comment='#', names=["config", "mass", "amp1", "amp2"])
            return df.amp1, df.amp2

    def get_decay(self, flavor, wild=None, op="PP"):
        if wild is None:
            wild = self.decay_file_wildcard

        decay_file = self.narrow_wildcard(wild, flavor=flavor, operator=op)

        with open(decay_file) as decayfile:
            df = pd.read_csv(decayfile, comment='#', names=["decay"])
            return df.decay

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

    def D_mass_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_uncorrelated_*/*.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-ud", wild=divwild)
        return self.get_mass("heavy-ud", wild=divwild)

    def D_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_uncorrelated_*/*.boot"
        return self.get_amps("heavy-ud", wild=divwild)

    def DA_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
        return self.get_amps("heavy-ud", wild=divwild, op="A4")

    def DsA_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
        return self.get_amps("heavy-s", wild=divwild, op="A4")


    def HH_mass(self, scaled=False):
        hhwild = "SymDW_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_PP.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild=hhwild)
        return self.get_mass("heavy-heavy", wild=hhwild)

    def HHv_mass(self, scaled=False):
        hhwild = "SymDW_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_vectorave.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild=hhwild, op="vectorave")
        return self.get_mass("heavy-heavy", wild=hhwild, op="vectorave")

    def Ds_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-s")
        return self.get_mass("heavy-s")

    def Ds_mass_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_uncorrelated_*/*.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-s", wild=divwild)
        return self.get_mass("heavy-s", wild=divwild)

    def xi(self, scaled=False):
        mpi = self.pion_mass(scaled=scaled)
        fpi = self.fpi(scaled=scaled)
        xi = ((mpi**2) / (8.0 * (np.pi**2)*(fpi**2)))
        return self.xi

    def fpi(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("ud-ud")
        return self.get_decay("ud-ud")

    def fK(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("ud-s")
        return self.get_decay("ud-s")

    def fD_div(self, scaled=False):
        divwild = "decay_constants_div/*_fixed_0_1-1_1/*decay_*_decayconstant_*.boot"
        if scaled:
            return self.scale*self.get_decay("heavy-ud", op="PP", wild=divwild)
        return self.get_decay("heavy-ud", op="PP", wild=divwild)

    def fDs_div(self, scaled=False):
        divwild = "decay_constants_div/*_fixed_0_1-1_1/*decay_*_decayconstant_*.boot"
        if scaled:
            return self.scale*self.get_decay("heavy-s", op="PP", wild=divwild)
        return self.get_decay("heavy-s", op="PP", wild=divwild)

    def fD(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-ud", op="PP")
        return self.get_decay("heavy-ud", op="PP")

    def fD_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-ud", op="A4")
        return self.get_decay("heavy-ud", op="A4")

    def fD_axial_div(self, scaled=False):
        divwild = "decay_constants_div/*_fixed_0_1-1_1/*decay_*_decayconstant_*.boot"
        if scaled:
            return self.scale*self.get_decay("heavy-ud", op="A4", wild=divwild)
        return self.get_decay("heavy-ud", op="A4", wild=divwild)

    def fDs_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-s", op="A4")
        return self.get_decay("heavy-s", op="A4")

    def fDs_axial_div(self, scaled=False):
        divwild = "decay_constants_div/*_fixed_0_1-1_1/*decay_*_decayconstant_*.boot"
        if scaled:
            return self.scale*self.get_decay("heavy-s", op="A4", wild=divwild)
        return self.get_decay("heavy-s", op="A4", wild=divwild)

    def fDs(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-s", op="PP")
        return self.get_decay("heavy-s", op="PP")

    def fDs_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_decay("heavy-s", op="A4")
        return self.get_decay("heavy-s", op="A4")

    def fHH(self, scaled=False):
        hhwild = "decay_constants/*_fixed_single/*_decayconstant_heavy-heavy.boot"
        if scaled:
            return self.scale*self.get_decay("heavy-heavy", wild=hhwild)
        return self.get_decay("heavy-heavy", wild=hhwild)

    def fDsbyfD(self, scaled=False):
        if scaled:
            return (self.scale*self.get_decay("heavy-s"))/(self.scale*self.get_decay("heavy-ud"))
        return self.get_decay("heavy-s")/self.get_decay("heavy-ud")






def test():

    fit_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot"
    decay_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_uncorrelated_*/*.boot"

    filename = "SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030/simul_fixed_fit_uncorrelated_ud-ud/simul_fit_uncorrelated_ll_ud-ud_0_1-1_1_PP.boot"

    dp = data_params(filename)

    ed = ensemble_data(dp)

    print ed.get_mass("ud-ud")
    print ed.get_mass("ud-s")
    print ed.kaon_mass()

if __name__ == "__main__":

    test()
