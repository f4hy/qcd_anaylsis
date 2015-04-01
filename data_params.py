

class data_params(object):
    def __init__(self, filename):
        self.filename = filename
        self.ud_mass = float(re.search("mud([0-9]\.[0-9]*)_", filename).group(1))
        self.s_mass = float(re.search("ms([0-9]\.[0-9]*)", filename).group(1))
        self.beta = re.search("_b(4\.[0-9]*)_", filename).group(1)

        self.smearing = re.search("([0-2]_[0-2])", filename).group(1)
        self.flavor = flavor_map[determine_flavor(filename)]
        self.heavyness = re.search("_([a-z][a-z0-9])_", filename).group(1)
        self.latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", filename).group(1)

        if self.heavyness != "ll":
            self.heavymass = re.search("_(m[012])_", filename).group(1)
        else:
            self.heavymass = None
