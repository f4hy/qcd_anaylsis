#!/usr/bin/env python2
import logging
import pandas as pd
import re
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from residualmasses import residual_mass
import glob
from ensamble_info import Zs, Zv


class MissingData(RuntimeError):
    pass


class NoStrangeInterp(MissingData):
    pass


class ensemble_data(object):

    def __init__(self, ensamble_info,
                 fit_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/simul_?????_fit_uncorrelated_*/*.boot",
                 interpstrange=False):

        self.dp = ensamble_info

        self.scale = scale[self.dp.beta]

        self.fit_file_wildcard = fit_file_wildcard

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

        logging.debug("narrowed to file {}".format(fitdatafiles[0]))
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

    def D_mass_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-ud", op="A4")
        return self.get_mass("heavy-ud", op="A4")

    def Ds_mass_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-s", op="A4")
        return self.get_mass("heavy-s", op="A4")

    def D_mass_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_uncorrelated_*/*.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-ud", wild=divwild)
        return self.get_mass("heavy-ud", wild=divwild)

    def D_mass_axial_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_uncorrelated_*/*.boot"
        if scaled:
            return self.scale*self.get_mass("heavy-ud", wild=divwild, op="A4")
        return self.get_mass("heavy-ud", wild=divwild, op="A4")


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
        return xi

    def fpi(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-ud")
        massdata = self.get_mass("ud-ud")
        ampfactor = self.dp.volume
        q1 = self.dp.ud_mass + residual_mass(self.dp)
        q2 = self.dp.ud_mass + residual_mass(self.dp)
        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = (q1 + q2)*np.sqrt(2*(ampdata) / massdata**3)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fpiA(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-ud", op="A4")
        massdata = self.get_mass("ud-ud", op="A4")
        ampfactor = self.dp.volume

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = np.sqrt(2*(ampdata) / massdata)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fK(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-s")
        massdata = self.get_mass("ud-s")
        ampfactor = self.dp.volume
        q1 = self.dp.ud_mass + residual_mass(self.dp)
        q2 = self.dp.s_mass + residual_mass(self.dp)
        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = (q1 + q2)*np.sqrt(2*(ampdata) / massdata**3)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fKA(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-s", op="A4")
        massdata = self.get_mass("ud-s", op="A4")
        ampfactor = self.dp.volume
        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = Zv[self.dp.beta]*np.sqrt(2*(ampdata) / massdata)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fD(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
            amp1data, amp2data = self.get_amps("heavy-ud", wild=divwild)
            massdata = self.get_mass("heavy-ud", wild=divwild)
        else:
            amp1data, amp2data = self.get_amps("heavy-ud")
            massdata = self.get_mass("heavy-ud")

        ampfactor = self.dp.volume
        q1 = self.dp.heavyq_mass + residual_mass(self.dp)
        q2 = self.dp.ud_mass + residual_mass(self.dp)

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor *= heavyfactor

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = (q1 + q2)*np.sqrt(2*(ampdata) / massdata**3)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fDA(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
            amp1data, amp2data = self.get_amps("heavy-ud", op="A4", wild=divwild)
            massdata = self.get_mass("heavy-ud", op="A4", wild=divwild)
        else:
            amp1data, amp2data = self.get_amps("heavy-ud", op="A4")
            massdata = self.get_mass("heavy-ud", op="A4")

        ampfactor = self.dp.volume

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor *= heavyfactor

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = Zv[self.dp.beta]*np.sqrt(2*(ampdata) / massdata)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fDs(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
            amp1data, amp2data = self.get_amps("heavy-s", wild=divwild)
            massdata = self.get_mass("heavy-s", wild=divwild)
        else:
            amp1data, amp2data = self.get_amps("heavy-s")
            massdata = self.get_mass("heavy-s")

        ampfactor = self.dp.volume
        q1 = self.dp.heavyq_mass + residual_mass(self.dp)
        q2 = self.dp.s_mass + residual_mass(self.dp)

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor *= heavyfactor

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = (q1 + q2)*np.sqrt(2*(ampdata) / massdata**3)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fDsA(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
            amp1data, amp2data = self.get_amps("heavy-s", op="A4", wild=divwild)
            massdata = self.get_mass("heavy-s", op="A4", wild=divwild)
        else:
            amp1data, amp2data = self.get_amps("heavy-s", op="A4")
            massdata = self.get_mass("heavy-s", op="A4")

        ampfactor = self.dp.volume

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor *= heavyfactor

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = Zv[self.dp.beta]*np.sqrt(2*(ampdata) / massdata)
        if scale:
            data = scale[self.dp.beta] * data
        return data

    def fHH(self, scaled=False):
        hhwild = "SymDW_*/fit_uncorrelated_heavy-heavy/fit_uncorrelated_*_heavy-heavy_0_0_PP.boot"
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_uncorrelated_*/*.boot"
            amp1data, amp2data = self.get_amps("heavy-heavy", wild=divwild)
            massdata = self.get_mass("heavy-heavy", wild=divwild)
        else:
            amp1data, amp2data = self.get_amps("heavy-heavy")
            massdata = self.get_mass("heavy-heavy")

        ampfactor = self.dp.volume
        q1 = self.dp.heavyq_mass + residual_mass(self.dp)
        q2 = self.dp.heavyq_mass + residual_mass(self.dp)

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor *= heavyfactor

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = (q1 + q2)*np.sqrt(2*(ampdata) / massdata**3)
        if scale:
            data = scale[self.dp.beta] * data
        return data



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
