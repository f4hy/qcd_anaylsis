import logging
import pandas as pd
import re

flavor_map = {"ud-ud": "\pi", "ud-s": "K", "s-s": "\eta", "heavy-ud": "Hl", "heavy-s": "Hs", "heavy-heavy": "HH", "KPratio": "KPratio", "2k-pi": "2m_K-m_\pi", "Omega": "\Omega"}
scale = {"4.17": 2492, "4.35": 3660, "4.47": 4600}


def determine_flavor(f):
    print f
    flavors = ["ud-ud", "ud-s", "s-s", "heavy-ud", "heavy-s", "heavy-heavy", "KPratio", "2k-pi", "Omega"]
    for flavor in flavors:
        if flavor in f:
            return flavor
    raise RuntimeError("Flavor not found")


class data_params(object):

    def __init__(self, filename):
        self.filename = filename
        self.ud_mass = float(re.search("mud([0-9]\.[0-9]*)_", filename).group(1))
        strange_mass = re.search("ms([a-z0-9.]+)", filename).group(1)
        try:
            self.s_mass = float(strange_mass)
        except ValueError:
            logging.warning("Found strange mass to be {}".format(strange_mass))
            self.s_mass = strange_mass
        self.beta = re.search("_b(4\.[0-9]*)_", filename).group(1)
        try:
            self.smearing = re.search("fixed_(.*)/", filename).group(1)
        except:
            self.smearing = "none"
        self.flavor = flavor_map[determine_flavor(filename)]
        self.heavyness = re.search("_([a-z][a-z0-9])_", filename).group(1)
        self.latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", filename).group(1)

        if self.heavyness != "ll":
            #self.heavymass = re.search("_heavy(0.[0-9]*)_", filename).group(1)
            self.heavymass = re.search("_([ms][012])_", filename).group(1)
        else:
            self.heavymass = None


def read_fit_mass(data_properties, flavor, fitdata):
    if fitdata is None:
        raise RuntimeError("--fitdata required when plotting with pionmass")
    import glob
    fitdatafiles = glob.glob(fitdata.strip("'\""))
    fitdatafiles = [f for f in fitdatafiles if flavor in f ]
    for i in [data_properties.ud_mass, data_properties.s_mass, data_properties.latsize, data_properties.beta]:
        fitdatafiles = [f for f in fitdatafiles if str(i) in f ]
    if len(fitdatafiles) != 1:
        logging.critical("Unique fit file not found!")
        logging.error("found: {}".format(fitdatafiles))
        raise SystemExit("Unique fit file not found!")

    with open(fitdatafiles[0]) as fitfile:
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
