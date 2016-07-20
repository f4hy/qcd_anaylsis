#!/usr/bin/env python2
import logging
import pandas as pd
import re
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from residualmasses import residual_mass
import glob
from ensamble_info import Zs, Zv
from alpha_s import get_Cmu_mbar

#FITTYPE="singlecorrelated"
FITTYPE="uncorrelated"
#FITTYPE="fullcorrelated"

class MissingData(RuntimeError):
    pass


class NoStrangeInterp(MissingData):
    pass


class ensemble_data(object):

    def __init__(self, ensamble_info,
                 fit_file_wildcard="SymDW_sHtTanh_b2.0_smr3_*/simul_?????_fit_{0}_*/*.boot".format(FITTYPE),
                 interpstrange=False):

        self.dp = ensamble_info

        self.bootstraps = self.dp.bootstraps

        self.scale = scale[self.dp.beta]

        self.fit_file_wildcard = fit_file_wildcard

        self.interpstrange = interpstrange

    def narrow_wildcard(self, fit_file_wildcard, flavor=None, operator="PP", axial=False, nextheavy=False):
        dp = self.dp

        if flavor is None:
            flavor_str = dp.flavor_string
        else:
            flavor_str = flavor

        smearing = dp.smearing
        if flavor == "xi":
            smearing = None

        heavyness = dp.heavyness
        if nextheavy:

            logging.debug("base heavyness {}".format(heavyness))
            heavyness = heavyness[0] + str(int(heavyness[1]) + 1)
            logging.debug("new heavyness {}".format(heavyness))


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

    def get_mass(self, flavor, wild=None, op="PP", nextheavy=False):
        if wild is None:
            wild = self.fit_file_wildcard

        mass_file = self.narrow_wildcard(wild, flavor=flavor, operator=op, nextheavy=nextheavy)

        with open(mass_file) as fitfile:
            df = pd.read_csv(fitfile, comment='#', names=["config", "mass", "amp1", "amp2"])
            assert(self.bootstraps == len(df))
            return df.mass

    def get_amps(self, flavor, wild=None, op="PP", nextheavy=False):
        if wild is None:
            wild = self.fit_file_wildcard

        amp_file = self.narrow_wildcard(wild, flavor=flavor, operator=op, nextheavy=nextheavy)

        with open(amp_file) as fitfile:
            df = pd.read_csv(fitfile, comment='#', names=["config", "mass", "amp1", "amp2"])
            assert(self.bootstraps == len(df))
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

    def D_mass_ratio(self, scaled=False, div=False, corrected=False):
        divwild = None
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)

        mass1  = self.get_mass("heavy-ud")
        try:
            mass2 = self.get_mass("heavy-ud", nextheavy=True)
        except MissingData:
            return [np.nan]*len(mass1)
        if corrected:
            mass1 = mass1 + (self.dp.heavy_m2 - self.dp.heavy_m1)
            mass2 = mass2 + (self.dp.heavy_m2_next - self.dp.heavy_m1_next)
        return mass2 / mass1

    def Ds_mass_ratio(self, scaled=False, div=False, corrected=False):
        divwild = None
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)

        mass1  = self.get_mass("heavy-s", wild=divwild)
        try:
            mass2 = self.get_mass("heavy-s", wild=divwild, nextheavy=True)
        except MissingData:
            return [np.nan]*len(mass1)
        if corrected:
            mass1 = mass1 + (self.dp.heavy_m2 - self.dp.heavy_m1)
            mass2 = mass2 + (self.dp.heavy_m2_next - self.dp.heavy_m1_next)
        return mass2 / mass1


    def D_mass_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-ud", op="A4")
        return self.get_mass("heavy-ud", op="A4")

    def Ds_mass_axial(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-s", op="A4")
        return self.get_mass("heavy-s", op="A4")

    # def D_mass_div_ratio(self, scaled=False):
    #     divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
    #     try:
    #         mass1 = self.get_mass("heavy-ud", wild=divwild)
    #         mass2 = self.get_mass("heavy-ud", wild=divwild, nextheavy=True)
    #     except MissingData:
    #         return [np.nan]*len(mass1)
    #     return mass2 / mass1

    # def Ds_mass_div_ratio(self, scaled=False):
    #     divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
    #     try:
    #         mass1 = self.get_mass("heavy-s", wild=divwild)
    #         mass2 = self.get_mass("heavy-s", wild=divwild, nextheavy=True)
    #     except MissingData:
    #         return [np.nan]*len(mass1)
    #     return mass2 / mass1


    def D_mass_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
        if scaled:
            return self.scale*self.get_mass("heavy-ud", wild=divwild)
        return self.get_mass("heavy-ud", wild=divwild)


    def D_mass_axial_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
        if scaled:
            return self.scale*self.get_mass("heavy-ud", wild=divwild, op="A4")
        return self.get_mass("heavy-ud", wild=divwild, op="A4")


    def D_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
        return self.get_amps("heavy-ud", wild=divwild)

    def DA_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
        return self.get_amps("heavy-ud", wild=divwild, op="A4")

    def DsA_amps_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
        return self.get_amps("heavy-s", wild=divwild, op="A4")

    def HH_mass(self, scaled=False):
        hhwild = "SymDW_*/fit_{0}_heavy-heavy/fit_{0}_*_heavy-heavy_0_0_PP.boot".format(FITTYPE)
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild=hhwild)
        return self.get_mass("heavy-heavy", wild=hhwild)

    def HHv_mass(self, scaled=False):
        hhwild = "SymDW_*/fit_{0}_heavy-heavy/fit_{0}_*_heavy-heavy_0_0_vectorave.boot".format(FITTYPE)
        if scaled:
            return self.scale*self.get_mass("heavy-heavy", wild=hhwild, op="vectorave")
        return self.get_mass("heavy-heavy", wild=hhwild, op="vectorave")

    def Ds_mass(self, scaled=False):
        if scaled:
            return self.scale*self.get_mass("heavy-s")
        return self.get_mass("heavy-s")

    def Ds_mass_div(self, scaled=False):
        divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_div_fit_{0}_*/*.boot".format(FITTYPE)
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fpiA(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-ud", op="A4")
        massdata = self.get_mass("ud-ud", op="A4")
        ampfactor = self.dp.volume

        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = np.sqrt(2*(ampdata) / massdata)
        if scaled:
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fKA(self, scaled=False):
        amp1data, amp2data = self.get_amps("ud-s", op="A4")
        massdata = self.get_mass("ud-s", op="A4")
        ampfactor = self.dp.volume
        ampdata = (amp1data**2 / amp2data) / ampfactor
        data = Zv[self.dp.beta]*np.sqrt(2*(ampdata) / massdata)
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fD(self, scaled=False, renorm=False, div=False, matched=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
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

        if scaled:
            data = scale[self.dp.beta] * data

        if matched:
            mq1 = self.scale * self.dp.heavyq_mass / Zs[self.dp.beta]
            print mq1
            C1 = get_Cmu_mbar(mq1)
            data = data / C1

        return data

    def fD_ratio(self, scaled=False, renorm=False, div=False, matched=False):
        try:
            if div:
                divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
                amp1data1, amp2data1 = self.get_amps("heavy-ud", wild=divwild)
                amp1data2, amp2data2 = self.get_amps("heavy-ud", wild=divwild, nextheavy=True)
                massdata1 = self.get_mass("heavy-ud", wild=divwild)
                massdata2 = self.get_mass("heavy-ud", wild=divwild, nextheavy=True)
            else:
                amp1data1, amp2data1 = self.get_amps("heavy-ud")
                amp1data2, amp2data2 = self.get_amps("heavy-ud", nextheavy=True)
                massdata1 = self.get_mass("heavy-ud")
                massdata2 = self.get_mass("heavy-ud", nextheavy=True)
        except MissingData:
            return [np.nan] * len(amp1data1)
        ampfactor1 = self.dp.volume
        ampfactor2 = self.dp.volume


        qh1 = self.dp.heavyq_mass + residual_mass(self.dp)
        qh2 = self.dp.heavyq_mass_next + residual_mass(self.dp)
        ql = self.dp.ud_mass + residual_mass(self.dp)

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor1 *= heavyfactor

            m = self.dp.heavyq_mass_next + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor2 *= heavyfactor

        ampdata1 = (amp1data1**2 / amp2data1) / ampfactor1
        ampdata2 = (amp1data2**2 / amp2data2) / ampfactor2
        data1 = (qh1 + ql)*np.sqrt(2*(ampdata1) / massdata1**3)
        data2 = (qh2 + ql)*np.sqrt(2*(ampdata2) / massdata2**3)

        if matched:
            mq1 = self.scale * self.dp.heavyq_mass / Zs[self.dp.beta]
            mq2 = self.scale * self.dp.heavyq_mass_next / Zs[self.dp.beta]
            C1 = get_Cmu_mbar(mq1)
            C2 = get_Cmu_mbar(mq2)
            data1 = data1 / C1
            data2 = data2 / C2

        data = data2 / data1



        return data


    def fDs_ratio(self, scaled=False, renorm=False, div=False, matched=False):
        try:
            if div:
                divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
                amp1data1, amp2data1 = self.get_amps("heavy-s", wild=divwild)
                amp1data2, amp2data2 = self.get_amps("heavy-s", wild=divwild, nextheavy=True)
                massdata1 = self.get_mass("heavy-s", wild=divwild)
                massdata2 = self.get_mass("heavy-s", wild=divwild, nextheavy=True)
            else:
                amp1data1, amp2data1 = self.get_amps("heavy-s")
                amp1data2, amp2data2 = self.get_amps("heavy-s", nextheavy=True)
                massdata1 = self.get_mass("heavy-s")
                massdata2 = self.get_mass("heavy-s", nextheavy=True)
        except MissingData:
            return [np.nan] * len(amp1data1)
        ampfactor1 = self.dp.volume
        ampfactor2 = self.dp.volume


        qh1 = self.dp.heavyq_mass + residual_mass(self.dp)
        qh2 = self.dp.heavyq_mass_next + residual_mass(self.dp)
        ql = self.dp.s_mass + residual_mass(self.dp)

        if renorm:
            m = self.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor1 *= heavyfactor

            m = self.dp.heavyq_mass_next + residual_mass(self.dp)
            Q = ((1 + m**2)/(1 - m**2))**2
            W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
            T = 1 - W0
            heavyfactor = 2.0/((1 - m**2)*(1 + np.sqrt(Q/(1 + 4*W0))))
            ampfactor2 *= heavyfactor

        ampdata1 = (amp1data1**2 / amp2data1) / ampfactor1
        ampdata2 = (amp1data2**2 / amp2data2) / ampfactor2
        data1 = (qh1 + ql)*np.sqrt(2*(ampdata1) / massdata1**3)
        data2 = (qh2 + ql)*np.sqrt(2*(ampdata2) / massdata2**3)

        if matched:
            mq1 = self.scale * self.dp.heavyq_mass / Zs[self.dp.beta]
            mq2 = self.scale * self.dp.heavyq_mass_next / Zs[self.dp.beta]
            C1 = get_Cmu_mbar(mq1)
            C2 = get_Cmu_mbar(mq2)
            data1 = data1 / C1
            data2 = data2 / C2

        data = data2 / data1

        return data


    def fDA(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fDs(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fDsA(self, scaled=False, renorm=False, div=False):
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data

    def fHH(self, scaled=False):
        hhwild = "SymDW_*/fit_{0}_heavy-heavy/fit_{0}_*_heavy-heavy_0_0_PP.boot".format(FITTYPE)
        if div:
            divwild = "SymDW_sHtTanh_b2.0_smr3_*/simul_?????_div_fit_{0}_*/*.boot".format(FITTYPE)
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
        if scaled:
            data = scale[self.dp.beta] * data
        return data



def test():

    fit_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_{0}_*/*.boot".format(FITTYPE)
    decay_file_wild = "SymDW_sHtTanh_b2.0_smr3_*/simul_fixed_fit_{0}_*/*.boot".format(FITTYPE)

    filename = "SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030/simul_fixed_fit_{0}_ud-ud/simul_fit_{0}_ll_ud-ud_0_1-1_1_PP.boot".format(FITTYPE)

    dp = data_params(filename)

    ed = ensemble_data(dp)

    print ed.get_mass("ud-ud")
    print ed.get_mass("ud-s")
    print ed.kaon_mass()

if __name__ == "__main__":

    test()
