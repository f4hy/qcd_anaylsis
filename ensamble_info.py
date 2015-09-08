import logging
import pandas as pd
import re

flavor_map = {"ud-ud": "\pi", "ud-s": "K", "s-s": "\eta", "heavy-ud": "Hl", "heavy-s": "Hs", "heavy-heavy": "HH", "KPratio": "KPratio", "2k-pi": "2m_K-m_\pi", "Omega": "\Omega", 't0': 't_0', 'w0': "w_0", "fds_fd_ratio": "fds/fd" , "fk_fpi_ratio": "fk/fpi"}
scale = {"4.17": 2492, "4.35": 3660, "4.47": 4600}
scale = {"4.17": 2473, "4.35": 3618, "4.47": 4600}
scale = {"4.17": 2453.1, "4.35": 3609.7, "4.47": 4496.1}

phys_pion = 134.8
phys_kaon = 494.2

# pdg m_u=2.3 m_d = 4.8 , so (m_d+m_d)/2 = 3.55
phys_mq = 3.55

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



def determine_flavor(f):
    flavors = flavor_map.keys()
    for flavor in flavors:
        if flavor in f:
            return flavor
    raise RuntimeError("Flavor not found")


class data_params(object):

    def __init__(self, filename):
        filename = filename.replace(".binned_to_20","")

        self.filename = filename

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
        try:
            self.smearing = re.search("fixed_(.*)/", filename).group(1)
        except:
            self.smearing = "none"
        self.flavor = flavor_map[determine_flavor(filename)]

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

        if self.heavyness != "ll" and self.heavyness is not None:
            #self.heavymass = re.search("_heavy(0.[0-9]*)_", filename).group(1)
            self.heavymass = re.search("_([ms][012])_", filename).group(1)
        else:
            self.heavymass = None

        self.ratio = "ratio" in filename


    def __repr__(self):
        if self.heavyness == "ll":
            return "{}_{}_{}_{}_{}_{}".format(self.beta, self.latsize, self.ud_mass, self.s_mass, self.flavor, self.heavyness)
        else:
            return "{}_{}_{}_{}_{}_{}_{}".format(self.beta, self.latsize, self.ud_mass, self.s_mass, self.flavor, self.heavyness, self.heavymass)


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
