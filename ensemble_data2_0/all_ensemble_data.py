#!/usr/bin/env python2
import logging
import numpy as np
from data_params import ensemble_params, bootstrap_data

from residualmasses import residual_mass
from alpha_s import get_Cmu_mbar

from pickle_ensemble_data import read_pickle

from collections import OrderedDict


# FITTYPE="singlecorrelated"
FITTYPE = "uncorrelated"
# FITTYPE="fullcorrelated"


ensemble_names = {}
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.0035_ms0.0400"] = "KC0"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030"] = "KC1"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.040"] = "KC2"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.012_ms0.030"] = "KC3"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.012_ms0.040"] = "KC4"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.019_ms0.030"] = "KC5"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.019_ms0.040"] = "KC6"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x12_b4.17_M1.00_mud0.0035_ms0.040"] = "KC7"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0042_ms0.0180"] = "KM0"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0042_ms0.0250"] = "KM1"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0080_ms0.0180"] = "KM2"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0080_ms0.0250"] = "KM3"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0120_ms0.0180"] = "KM4"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0120_ms0.0250"] = "KM5"
ensemble_names["SymDW_sHtTanh_b2.0_smr3_64x128x08_b4.47_M1.00_mud0.0030_ms0.0150"] = "Kf0"


class MissingData(RuntimeError):
    pass


class NoStrangeInterp(MissingData):
    pass


class ensemble_data(object):

    def __init__(self, ensemble,
                 smearing="0_1-1_1",
                 interpstrange=False, scale_values=True, def_op="PP", fittype="uncorrelated"):

        logging.debug("creating ensembledata for {}".format(ensemble))
        self.ep = ensemble_params(ensemble)
        logging.debug("created ensembledata with params {}".format(self.ep))

        self.default_smearing = smearing
        self.default_operator = def_op

        # self.scale = self.ep.scale
        # if scale_values is False:
        #     self.scale = 1.0
        if scale_values is False:
            self.ep.scale = 1.0

        self.interpstrange = interpstrange

        self.data = read_pickle(ensemble, fittype=fittype)

    def select_data(self, flavor, operator=None, heavy=None, smearing=None, axial=False, div=False, **args):

        if operator is None:
            if axial:
                operator = "A4P"
            else:
                operator = self.default_operator

        if smearing is None:
            if flavor == "heavy-heavy":
                smearing = "0_0"
            else:
                smearing = self.default_smearing
        selected = [d for d in self.data if smearing in d and flavor in d and operator in d]
        logging.debug("selected {}".format(selected))

        selected = [d for d in selected if div == ("div" in d)]
        logging.debug("selected {}".format(selected))

        if heavy is not None:
            selected = [d for d in selected if heavy in d]
            logging.debug("selected {}".format(selected))

        for s in selected:
            logging.debug("selected file {}".format(self.data[s].filename))
            logging.debug("selected data {}:{}".format(s, self.data[s]))

        if len(selected) == 1:
            return self.data[selected[0]]
        if len(selected) == 0:
            logging.error("No data found for selection!")
            raise MissingData()
        else:
            logging.debug("selected more than one data value")
            return OrderedDict([(k, self.data[k]) for k in sorted(selected)])

    def get_mass(self, flavor, **args):
        data = self.select_data(flavor, **args)
        if isinstance(data, bootstrap_data):
            return self.ep.scale * data.mass
        else:
            mass_data = OrderedDict([(k, self.ep.scale * v.mass) for k, v in data.iteritems()])
            return mass_data

    def get_amps(self, flavor, **args):
        data = self.select_data(flavor, **args)
        if isinstance(data, bootstrap_data):
            return data.amp1, data.amp2
        else:
            amp_data = OrderedDict([(k, (v.amp1, v.amp2)) for k, v in data.iteritems()])
            return amp_data

    def pion_mass(self):
        return self.get_mass("ud-ud")

    def kaon_mass(self):
        return self.get_mass("ud-s")

    def eta_mass(self):
        return self.get_mass("s-s")

    def D_mass(self, **args):
        if "heavy" not in args:
            args["heavy"] = "m0"
        return self.get_mass("heavy-ud", **args)

    def Ds_mass(self, **args):
        if "heavy" not in args:
            args["heavy"] = "m0"
        return self.get_mass("heavy-s", **args)

    def hl_mass_ratio(self, corrected=False, **args):

        ratios = OrderedDict()
        hl_data = self.select_data("heavy-ud", **args)
        for i in range(len(hl_data) - 1):
            num = "m{}".format(i + 1)
            dem = "m{}".format(i)
            num_k = next(k for k in hl_data if num in k)
            dem_k = next(k for k in hl_data if dem in k)
            num_m = hl_data[num_k].mass
            dem_m = hl_data[dem_k].mass
            if corrected:
                num_m += hl_data[num_k].dp.heavy_m2 - hl_data[num_k].dp.heavy_m1
                dem_m += hl_data[dem_k].dp.heavy_m2 - hl_data[dem_k].dp.heavy_m1
            ratios[dem] = (num_m / dem_m) / 1.15
        return ratios

    def hs_mass_ratio(self, corrected=False, **args):

        ratios = OrderedDict()
        hs_data = self.select_data("heavy-s", **args)
        for i in range(len(hs_data) - 1):
            num = "m{}".format(i + 1)
            dem = "m{}".format(i)
            num_k = next(k for k in hs_data if num in k)
            dem_k = next(k for k in hs_data if dem in k)
            num_m = hs_data[num_k].mass
            dem_m = hs_data[dem_k].mass
            if corrected:
                num_m += hs_data[num_k].dp.heavy_m2 - hs_data[num_k].dp.heavy_m1
                dem_m += hs_data[dem_k].dp.heavy_m2 - hs_data[dem_k].dp.heavy_m1
            ratios[dem] = (num_m / dem_m) / 1.15
        return ratios

    def xi(self, scaled=False):
        mpi = self.pion_mass()
        fpi = self.fpi()/np.sqrt(2)
        xi = ((mpi**2) / (16.0 * (np.pi**2) * (fpi**2)))
        return xi

    def fpi(self, **args):
        if args.get("axial", False):
            logging.error("fpi called with axial, using fpiA")
            return self.fpiA(**args)
        d = self.select_data("ud-ud", **args)
        ampfactor = self.ep.volume
        q1 = self.ep.ud_mass + self.ep.residual_mass
        q2 = self.ep.ud_mass + self.ep.residual_mass
        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = (q1 + q2) * np.sqrt(2 * (ampdata) / d.mass**3)
        return self.ep.scale * data

    def fpiA(self, **args):
        d = self.select_data("ud-ud", axial=True, **args)
        ampfactor = self.ep.volume
        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = self.ep.Zv * np.sqrt(2 * (ampdata) / d.mass)
        return self.ep.scale * data

    def fK(self, **args):
        if args.get("axial", False):
            logging.error("fK called with axial, using fKA")
            return self.fpiA(**args)
        d = self.select_data("ud-s", **args)
        ampfactor = self.ep.volume
        q1 = self.ep.ud_mass + self.ep.residual_mass
        q2 = self.ep.s_mass + self.ep.residual_mass
        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = (q1 + q2) * np.sqrt(2 * (ampdata) / d.mass**3)
        return self.ep.scale * data

    def fKA(self, **args):
        d = self.select_data("ud-s", axial=True, **args)
        ampfactor = self.ep.volume
        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = self.ep.Zv * np.sqrt(2 * (ampdata) / d.mass)
        return self.ep.scale * data

    def fD(self, **args):
        if args.get("axial", False):
            logging.error("fD called with axial, using fDA")
            return self.fDA(**args)
        if args.get("div", False):
            args["renorm"] = True

        if "heavy" not in args:
            args["heavy"] = "m0"

        d = self.select_data("heavy-ud", **args)
        ampfactor = self.ep.volume
        q1 = d.dp.heavyq_mass + self.ep.residual_mass
        q2 = self.ep.ud_mass + self.ep.residual_mass

        if args.get("renorm", False):
            m = d.dp.heavyq_mass + self.ep.residual_mass
            Q = ((1 + m**2) / (1 - m**2))**2
            W0 = (1 + Q) / 2 - np.sqrt(3 * Q + Q**2) / 2
            T = 1 - W0          # noqa
            heavyfactor = 2.0 / ((1 - m**2) * (1 + np.sqrt(Q / (1 + 4 * W0))))
            ampfactor *= heavyfactor

        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = (q1 + q2) * np.sqrt(2 * (ampdata) / d.mass**3)

        data = self.ep.scale * data
        if args.get("matched", False):
            mq1 = self.ep.scale * d.dp.heavyq_mass / self.ep.Zs
            C1 = get_Cmu_mbar(mq1)
            data = data / C1

        return data

    def fDA(self, **args):
        if args.get("div", False):
            args["renorm"] = True

        if "heavy" not in args:
            args["heavy"] = "m0"

        d = self.select_data("heavy-ud", axial=True, **args)
        ampfactor = self.ep.volume

        if args.get("renorm", False):
            m = d.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2) / (1 - m**2))**2
            W0 = (1 + Q) / 2 - np.sqrt(3 * Q + Q**2) / 2
            T = 1 - W0  # noqa
            heavyfactor = 2.0 / ((1 - m**2) * (1 + np.sqrt(Q / (1 + 4 * W0))))
            ampfactor *= heavyfactor

        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = self.ep.Zv * np.sqrt(2 * (ampdata) / d.mass)

        data = self.ep.scale * data
        if args.get("matched", False):
            mq1 = self.ep.scale * d.dp.heavyq_mass / self.ep.Zs
            C1 = get_Cmu_mbar(mq1)
            data = data / C1

        return data

    def fhl(self, **args):
        N = len(self.select_data("heavy-ud"))
        data = OrderedDict()
        for i in range(N):
            m = "m{}".format(i)
            args["heavy"] = m
            data[m] = self.fD(**args)
        return data

    def fDs(self, **args):
        logging.debug("fds called with {}".format(args))
        if args.get("axial", False):
            logging.error("fDs called with axial, using fDsA")
            return self.fDsA(**args)
        if args.get("div", False):
            logging.debug("div is active, setting renorm")
            args["renorm"] = True

        if "heavy" not in args:
            args["heavy"] = "m0"

        d = self.select_data("heavy-s", **args)
        ampfactor = self.ep.volume
        q1 = d.dp.heavyq_mass + self.ep.residual_mass
        q2 = self.ep.s_mass + self.ep.residual_mass

        if args.get("renorm", False):
            m = d.dp.heavyq_mass + self.ep.residual_mass
            Q = ((1 + m**2) / (1 - m**2))**2
            W0 = (1 + Q) / 2 - np.sqrt(3 * Q + Q**2) / 2
            T = 1 - W0 # noqa
            heavyfactor = 2.0 / ((1 - m**2) * (1 + np.sqrt(Q / (1 + 4 * W0))))
            ampfactor *= heavyfactor

        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = (q1 + q2) * np.sqrt(2 * (ampdata) / d.mass**3)

        data = self.ep.scale * data
        if args.get("matched", False):
            mq1 = self.ep.scale * d.dp.heavyq_mass / self.ep.Zs
            C1 = get_Cmu_mbar(mq1)
            data = data / C1

        return data

    def fDsA(self, **args):
        if args.get("div", False):
            args["renorm"] = True

        if "heavy" not in args:
            args["heavy"] = "m0"

        d = self.select_data("heavy-s", axial=True, **args)
        ampfactor = self.ep.volume

        if args.get("renorm", False):
            m = d.dp.heavyq_mass + residual_mass(self.dp)
            Q = ((1 + m**2) / (1 - m**2))**2
            W0 = (1 + Q) / 2 - np.sqrt(3 * Q + Q**2) / 2
            T = 1 - W0 # noqa
            heavyfactor = 2.0 / ((1 - m**2) * (1 + np.sqrt(Q / (1 + 4 * W0))))
            ampfactor *= heavyfactor

        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = self.ep.Zv * np.sqrt(2 * (ampdata) / d.mass)

        data = self.ep.scale * data
        if args.get("matched", False):
            mq1 = self.ep.scale * d.dp.heavyq_mass / self.ep.Zs
            C1 = get_Cmu_mbar(mq1)
            data = data / C1

        return data

    def fhs(self, **args):
        N = len(self.select_data("heavy-s"))
        data = OrderedDict()
        for i in range(N):
            m = "m{}".format(i)
            args["heavy"] = m
            data[m] = self.fDs(**args)
        return data

    def fHH(self, **args):
        if args.get("axial", False):
            logging.error("fDs called with axial, using fDsA")
            return self.fDsA(**args)
        if args.get("div", False):
            args["renorm"] = True

        if "heavy" not in args:
            args["heavy"] = "m0"

        d = self.select_data("heavy-heavy", **args)
        ampfactor = self.ep.volume
        q1 = d.dp.heavyq_mass + self.ep.residual_mass
        q2 = d.dp.heavyq_mass + self.ep.residual_mass

        if args.get("renorm", False):
            m = d.dp.heavyq_mass + self.ep.residual_mass
            Q = ((1 + m**2) / (1 - m**2))**2
            W0 = (1 + Q) / 2 - np.sqrt(3 * Q + Q**2) / 2
            T = 1 - W0 # noqa
            heavyfactor = 2.0 / ((1 - m**2) * (1 + np.sqrt(Q / (1 + 4 * W0))))
            ampfactor *= heavyfactor

        ampdata = (d.amp1**2 / d.amp2) / ampfactor
        data = (q1 + q2) * np.sqrt(2 * (ampdata) / d.mass**3)

        data = self.ep.scale * data

        return data

    def ZA_est(self, **args):
        if "flavor" not in args:
            args["flavor"] = "ud-ud"

        d = self.select_data(axial=False, **args)
        dA = self.select_data(axial=True, **args)
        ampfactor = self.ep.volume
        Pampdata = np.sqrt((d.amp1**2 / d.amp2) / ampfactor)
        Aampdata = np.sqrt((dA.amp1**2 / dA.amp2) / ampfactor)

        qu = self.ep.ud_mass + self.ep.residual_mass
        qs = self.ep.s_mass + self.ep.residual_mass
        qh = d.dp.heavyq_mass + self.ep.residual_mass
        qmasses = {"ud-ud": qu+qu, "ud-s": qs+qu, "s-s":qs+qs,
                   "h-h":qh+qh, "h-ud":qh+qu, "h-s":qh+qs}
        data = qmasses[args["flavor"]] * Pampdata / (d.mass * Aampdata)

        return data



def test():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # ed = ensemble_data("SymDW_sHtTanh_b2.0_smr3_32x64x12_b4.17_M1.00_mud0.007_ms0.030")
    ed = ensemble_data("SymDW_sHtTanh_b2.0_smr3_48x96x08_b4.35_M1.00_mud0.0080_ms0.0180")

    logging.info(np.mean(ed.fpi()))
    logging.info(np.mean(ed.fD()))


if __name__ == "__main__":

    test()
