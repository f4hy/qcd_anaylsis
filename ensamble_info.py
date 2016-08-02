import logging
import pandas as pd
import re
import numpy as np

flavor_map = {"ud-ud": "\pi", "ud-s": "K", "s-s": "\eta", "heavy-ud": "Hl", "heavy-s": "Hs", "heavy-heavy": "HH", "KPratio": "KPratio", "2k-pi": "2m_K-m_\pi", "Omega": "\Omega", 't0': 't_0', 'w0': "w_0", "fds_fd_ratio": "fds/fd" , "fk_fpi_ratio": "fk/fpi"}
scale = {"4.17": 2492, "4.35": 3660, "4.47": 4600}
scale = {"4.17": 2473, "4.35": 3618, "4.47": 4600}
scale = {"4.17": 2453.1, "4.35": 3609.7, "4.47": 4496.1}

phys_pionplus = 139.570

phys_pion = 134.8
phys_kaon = 494.2

phys_eta = 547.862 # +/- 0.018

unphys_etas = 685.8 # unphysical s\bar{s} meson

phys_Fpi = 130.41
phys_FK = 156.1 # MeV

phys_MB = 5279.4

phys_MBs = 5366.79

phys_Jpsi = 3096.916 # \pm 0.011 MeV PDG
phys_Upsilon = 9460.30 # \pm 0.26 MeV PDG

phys_etac = 2983.6 # \pm 0.6 MeV PDG
phys_etab = 9390.9 # \pm 2.8 MeV

phys_D = (1864.84 + 2*1869.61)/3.0 #
phys_Ds = 1968.3

phys_FD = 209.2

phys_FDs = 248.6

phys_FB = 190.5 # \pm 4.2 FLAG

phys_FBs = 227.7 #\pm 4.5 FLAG

phys_FBsbyFB = 1.202 #\pm 0.022 FLAG


# pdg m_u=2.3 m_d = 4.8 , so (m_d+m_d)/2 = 3.55
phys_mq = 3.55

phys_mhq = 1275                 # at mc(mc)

phys_mc_2gev = 1080.0            # from ratio methods at 2gv

phys_mb_2gev = 4100.0           # from ratio methods at 2gv

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

# # Zs(=Zp):<RGI>  = Zs:<MSbar, 2GeV>*0.75
# Zs = {"4.17": 1.024*0.75, "4.35": 0.922*0.75, "4.47": 0.880*0.75}


hbar_c = 197.3269788

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
    flavors = flavor_map.keys()
    for flavor in flavors:
        if flavor in f:
            return flavor
    raise RuntimeError("Flavor not found")

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
    try:
        return heavymap[heavytype]
    except:
        return 0.0

def get_heavy_m1_m2(m):
    Q = ((1 + m**2)/(1 - m**2))**2
    W0 = (1 + Q)/2 - np.sqrt(3*Q + Q**2)/2
    T = 1 - W0
    m1 = np.log(T + np.sqrt(T**2 - 1))
    m2 =    np.sqrt(W0**2 - 2*W0)*(Q + 1 - 2*W0)/((Q + 1) + (Q - 1) * (2*W0**2 + W0))
    return m1, m2


class data_params(object):

    def __init__(self, filename):
        filename = filename.replace(".binned_to_20","")

        self.filename = filename

        self.bootstraps = sum(1 for line in open(self.filename) if not line.startswith("#"))

        for i in ensemble_names.keys():
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
        self.beta = re.search("_b(4\.[0-9]*)_", filename).group(1)

        self.latspacing = hbar_c/scale[self.beta]

        try:
            self.smearing = re.search("(\d_\d-\d_\d)", filename).group(1)
        except:
            self.smearing = None

        self.flavor_string = determine_flavor(filename)
        self.flavor = flavor_map[self.flavor_string]

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

        self.heavyq_mass = get_heavyq_mass(self.beta, self.heavymass)
        self.heavyq_mass_next = get_heavyq_mass(self.beta, self.heavymass_next)

        if self.heavyq_mass is not None:
            self.heavy_m1, self.heavy_m2 = get_heavy_m1_m2(self.heavyq_mass)
            self.heavy_m1_next, self.heavy_m2_next = get_heavy_m1_m2(self.heavyq_mass_next)

        self.ratio = "ratio" in filename

        self.operator = None
        for i in ["PP", "A4P", "PA4", "vectorave", "decayconst"]:
            if i in filename:
                self.operator = i


    def __repr__(self):
        if self.heavyness == "ll":
            return "{}_{}_{}_{}_{}_{}_{}".format(self.ename, self.beta, self.latsize, self.ud_mass, self.s_mass, self.flavor, self.heavyness)
        else:
            return "{}_{}_{}_{}_{}_{}_{}_{}".format(self.ename, self.beta, self.latsize, self.ud_mass, self.s_mass, self.flavor, self.heavyness, self.heavymass)


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
