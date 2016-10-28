from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon

import numpy as np


def fDssqrtmDs_m0(ed, options):
    fdata = ed.fDs(heavy="m0")
    mdata = ed.get_mass("heavy-s", heavy="m0")

    data = fdata*np.sqrt(mdata)

    label = "$f_{hs}\, \sqrt{m_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})


def fDssqrtmDs_m1(ed, options):
    fdata = ed.fDs(heavy="m1")
    mdata = ed.get_mass("heavy-s", heavy="m1")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m1}_{hs}\, \sqrt{m^{m1}_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})


def fDssqrtmDs_m2(ed, options):
    fdata = ed.fDs(heavy="m2")
    mdata = ed.get_mass("heavy-s", heavy="m2")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m2}_{hs}\, \sqrt{m^{m2}_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

def fDssqrtmDs_m3(ed, options):
    fdata = ed.fDs(heavy="m3")
    mdata = ed.get_mass("heavy-s", heavy="m3")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m3}_{hs}\, \sqrt{m^{m3}_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

def fDssqrtmDs_m4(ed, options):
    fdata = ed.fDs(heavy="m4")
    mdata = ed.get_mass("heavy-s", heavy="m4")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m4}_{hs}\, \sqrt{m^{m4}_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

def fDssqrtmDs_m5(ed, options):
    fdata = ed.fDs(heavy="m5")
    mdata = ed.get_mass("heavy-s", heavy="m5")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m5}_{hs}\, \sqrt{m^{m5}_{hs}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return plot_data(data.mean(), data.std(), label,
                     {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

def fDssqrtmDs_ratio(ed, options):
    fdata = ed.fhs()
    mdata = ed.get_mass("heavy-s")
    print fdata.keys()
    print mdata.keys()

    data = {}
    for m in fdata.keys():
        mkey = [k for k in mdata.keys() if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = {}
    for i in range(1, len(fdata.keys())):
        m = "m{}".format(i)
        mm = "m{}".format(i-1)
        ratiodata[m] = data[m] / data[mm]

    label = "$\\frac{f_{hs}\, \sqrt{m_{hs}}}{f_{h^{-1}s}\, \sqrt{m_{h^{-1}}}}$"
    # if options.scale:
    #     label += " [MeV^(3/2)]"
    phys =  {"HQL": 1.0}
    return {m: (d.mean(), d.std(), label, phys) for m,d in ratiodata.iteritems()}
