import logging
import pandas as pd
import re
import numpy as np
from residualmasses import residual_mass, residual_mass_errors

from physical_values import hbar_c

flavor_map = {"ud-ud": "\pi", "ud-s": "K", "s-s": "\eta", "heavy-ud": "Hl", "heavy-s": "Hs", "heavy-heavy": "HH", "KPratio": "KPratio", "2k-pi": "2m_K-m_\pi", "Omega": "\Omega", 't0': 't_0', 'w0': "w_0", "fds_fd_ratio": "fds/fd" , "fk_fpi_ratio": "fk/fpi"}

scale = {"4.17": 2453.1, "4.35": 3609.7, "4.47": 4496.1}

# Zv(=Za)<MSbar>
# beta4.17: Zv = 0.9517(58)(10)(33)
# beta4.35: Zv = 0.9562(42)(8)(20)
# beta4.47: Zv = 0.9624(33)(7)(20)
Zv = {"4.17": 0.9517, "4.35": 0.9562, "4.47": 0.9624}

# Zs(=Zp):<MSbar, 2GeV>
# beta4.17: Zs = 1.024(15)(84)(6)
# beta4.35: Zs = 0.922(11)(45)(5)
# beta4.47: Zs = 0.880(7)(38)(4)
Zs = {"4.17": 1.024, "4.35": 0.922, "4.47": 0.880}


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


def determine_flavor(f):
    return next(flavor for flavor in flavor_map if flavor in f)

def get_heavyq_mass(beta, heavytype):
    if heavytype is None:
        return 0.0
    if beta == "4.17":
        #0.44037 0.55046 0.68808 0.86001
        heavymap = {"m0": 0.44037, "m1": 0.55046, "m2": 0.68808, "m3": 0.86001, "m4": float("NAN"), "m5": float("NAN")}
    if beta == "4.35":
        #0.27287 0.34109 0.42636 0.53295 0.66619 0.83273
        heavymap = {"m0": 0.27287, "m1": 0.34109, "m2": 0.42636, "m3": 0.53295, "m4": 0.66619, "m5": 0.83273}
    if beta == "4.47":
        #0.210476 0.263095 0.328869 0.4110859 0.5138574 0.6423218
        heavymap = {"m0": 0.210476, "m1": 0.263095, "m2": 0.328869, "m3": 0.4110859, "m4": 0.5138574, "m5": 0.6423218}
    return heavymap.get(heavytype, 0.0)

def get_heavy_m1_m2(m):
    Q = ((1 + m**2)/(1 - m**2))**2
    W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
    T = 1 - W0
    m1 = np.log(T + np.sqrt(T**2 - 1))
    m2 =    np.sqrt(W0**2 - 2*W0)*(Q + 1 - 2*W0)/((Q + 1) + (Q - 1) * (2*W0**2 + W0))
    return m1, m2


class ensemble_params(object):

    def __init__(self, filename):
        filename = filename.replace(".binned_to_20","")

        self.filename = filename

        self.ename = None
        for i in ensemble_names:
            if i in filename:
                self.ename = ensemble_names[i]


        self.ud_mass = float(re.search("mud([0-9]\.[0-9]*)_", filename).group(1))
        strange_mass = re.search("ms([a-z0-9.]+)", filename).group(1)
        strange_mass = strange_mass.replace(".jack", "")
        strange_mass = strange_mass.replace(".binned", "")


        try:
            self.s_mass = float(strange_mass)
        except ValueError:
            logging.warning("Found strange mass to be {}".format(strange_mass))
            self.s_mass = strange_mass
        self.beta = re.search("_b([0-9]\.[0-9][0-9])_", filename).group(1)
        if self.beta == "0.00":
            self.beta = "4.47"
        self.latspacing = hbar_c/scale[self.beta]

        self.scale = scale[self.beta]

        self.a_gev = 1000.0/(self.scale)

        if self.ename is None:
            self.ename = "Ib{}u{}s{}".format(self.beta, self.ud_mass, self.s_mass)


        try:
            self.latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", filename).group(1)
            self.volume = int(self.latsize.split("x")[0])**3
        except AttributeError:
            self.latsize = None

        if self.latsize is None:
            try:
                self.latsize = re.search("L([0-9]+)", filename).group(1)
                self.volume = int(self.latsize[1:])**3
            except AttributeError:
                pass

        self.residual_mass = residual_mass(self)
        self.residual_mass_error = residual_mass_errors(self)

        self.Zv = Zv.get(self.beta, 0.0)
        self.Zs = Zs.get(self.beta, 0.0)

        self.heavies = {}
        if self.beta == "4.17":
            self.heavies = {"m0": 0.44037, "m1": 0.55046, "m2": 0.68808, "m3": 0.86001}

        if self.beta == "4.35":
            self.heavies = {"m0": 0.27287, "m1": 0.34109, "m2": 0.42636, "m3": 0.53295, "m4": 0.66619, "m5": 0.83273}

        if self.beta == "4.47":
            self.heavies = {"m0": 0.210476, "m1": 0.263095, "m2": 0.328869, "m3": 0.4110859, "m4": 0.5138574, "m5": 0.642}


        self.m12s = { k: get_heavy_m1_m2(m) for k,m  in self.heavies.iteritems()}

    def __repr__(self):
        rstr = "{}_{}_{}_{}_{}".format(self.ename, self.beta, self.latsize, self.ud_mass, self.s_mass)
        return rstr


class data_params(ensemble_params):

    def __init__(self, filename):
        ensemble_params.__init__(self,filename)

        self.bootstraps = sum(1 for line in open(self.filename) if not line.startswith("#"))

        self.smearing = None
        try:
            self.smearing = re.search("(\d_\d-\d_\d)", filename).group(1)
        except:
            try:
                self.smearing = re.search("_(\d_\d)_", filename).group(1)
            except:
                logging.warn("No smearing found for file")
                pass

        self.flavor_string = determine_flavor(filename)
        self.flavor = flavor_map[self.flavor_string]

        try:
            self.heavyness = re.search("_([a-z][a-z0-9])_", filename).group(1)
        except AttributeError:
            self.heavyness = None
            self.heavymass_next = None

        if self.heavyness != "ll" and self.heavyness is not None:
            #self.heavymass = re.search("_heavy(0.[0-9]*)_", filename).group(1)
            self.heavymass = re.search("_([ms][012345])_", filename).group(1)
            self.heavymass_next = self.heavymass[0] + str(int(self.heavymass[1])+1)
        else:
            self.heavymass = None
            self.heavymass_next = None

        self.heavyq_mass = get_heavyq_mass(self.beta, self.heavymass)
        self.heavyq_mass_next = get_heavyq_mass(self.beta, self.heavymass_next)

        if self.heavyq_mass is not None:
            self.heavy_m1, self.heavy_m2 = get_heavy_m1_m2(self.heavyq_mass)
            self.heavy_m1_next, self.heavy_m2_next = get_heavy_m1_m2(self.heavyq_mass_next)

        self.ratio = "ratio" in filename

        self.axial = "axial" in filename
        self.div = "_div_" in filename

        # self.operator = None
        # for i in ["PP", "A4P", "PA4", "A4A4", "vectorave", "decayconst", "interpolated"]:
        #     if i in filename:
        #         self.operator = i
        operator_choices = ("PP", "A4P", "PA4", "A4A4", "vectorave", "decayconst", "fsqrtm_linearhqet_continuum", "fsqrtm_hqet_continuum", "fsqrtm_continuum")
        self.operator = next((i for i  in operator_choices if i in filename), None)


    def __repr__(self):
        rstr = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.ename, self.beta, self.latsize, self.ud_mass, self.s_mass, self.flavor_string, self.heavyness, self.smearing, self.operator)
        if self.heavyness != "ll":
            rstr += "_{}".format(self.heavymass)
        if self.axial:
            rstr += "_axial"
        if self.div:
            rstr += "_div"
        return rstr

class bootstrap_data(object):

    def __init__(self, d, values=None):

        if values is None:
            filename = d
            self.filename = filename
            self.dp = data_params(filename)
            self.values = self.read_data(filename)
        else:
            self.filename = d.filename
            self.dp = d
            self.values = values


    def __getattr__(self, attr):
        if attr == "mass":
            return self.values.mass
        if attr == "amp1":
            return self.values.amp1
        if attr == "amp2":
            return self.values.amp2
        else:
            raise AttributeError("%r object has no attribute %r" %
                                  (self.__class__, attr))

    def read_data(self, filename):
        with open(filename) as fitfile:
            df = pd.read_csv(fitfile, comment='#', names=["config", "mass", "amp1", "amp2"])
        return df

    def __repr__(self):
        rstring = "{}: straps:{}, mean:{}, std:{}".format(repr(self.dp), self.dp.bootstraps, self.values.mass.mean(), self.values.mass.std())
        return rstring


def read_fit_mass(data_properties, flavor, fitdata):
    if fitdata is None:
        raise RuntimeError("--fitdata required when plotting with pionmass")
    import glob
    fitdatafiles = glob.glob(fitdata.strip("'\""))
    fitdatafiles = [f for f in fitdatafiles if flavor in f ]
    for i in [data_properties.ud_mass, data_properties.s_mass, data_properties.latsize, data_properties.beta]:
        if i is not None:
            fitdatafiles = [f for f in fitdatafiles if str(i) in f ]
    if len(fitdatafiles) != 1:
        logging.critical("Unique fit file not found!")
        logging.error("found: {}".format(fitdatafiles))
        raise SystemExit("Unique fit file not found!")

    with open(fitdatafiles[0]) as fitfile:
        if flavor is "xi":
            df = pd.read_csv(fitfile,comment='#', names=["xi"])
            return df.xi
        df = pd.read_csv(fitfile,comment='#', names=["config", "mass", "amp1", "amp2"])
        return df.mass


def allEqual(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def all_same_beta(files):
    logging.info("Determining beta")
    beta_filesnames = [re.search("_b(4\.[0-9]*)_", f).group(1) for f in files]
    return allEqual(beta_filesnames)

def all_same_heavy(files):
    logging.info("Determining heavy")
    beta_filesnames = [re.search("_([ms][012])_|_(ll)_", f).group(1) for f in files]
    return allEqual(beta_filesnames)


def all_same_flavor(files):
    logging.info("Determining flavor")
    flavors_filesnames = [determine_flavor(f) for f in files]
    return allEqual(flavors_filesnames)
