from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon

import numpy as np

def fDssqrtmDs(ed, options, heavy="m0"):
    fdata = ed.fDs(heavy=heavy)
    mdata = ed.get_mass("heavy-s", heavy=heavy)

    data = fdata*np.sqrt(mdata)

    label = "$f^{"+heavy+"}_{hs}\, \sqrt{m^{"+heavy+"}_{hs}}$"

    if ed.scale != 1.0:

        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})


def fDssqrtmDs_m0(ed, options):
    return fDssqrtmDs(ed, options, heavy="m0")
def fDssqrtmDs_m1(ed, options):
    return fDssqrtmDs(ed, options, heavy="m1")
def fDssqrtmDs_m2(ed, options):
    return fDssqrtmDs(ed, options, heavy="m2")
def fDssqrtmDs_m3(ed, options):
    return fDssqrtmDs(ed, options, heavy="m3")
def fDssqrtmDs_m4(ed, options):
    return fDssqrtmDs(ed, options, heavy="m4")
def fDssqrtmDs_m5(ed, options):
    return fDssqrtmDs(ed, options, heavy="m5")

def fhssqrtmhs(ed, options):
    fdata = ed.fhs()
    mdata = ed.get_mass("heavy-s")

    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$f_{hs}\, \sqrt{m_{hs}}$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}

def fDssqrtmDs_hqet(ed, options, heavy="m0"):
    fdata = ed.fDs(heavy=heavy, div=True)
    mdata = ed.get_mass("heavy-s", heavy=heavy, div=True)

    data = fdata*np.sqrt(mdata)

    label = "$\hat{f}^{"+heavy+"}_{hs}\, \sqrt{\hat{m}^{"+heavy+"}_{hs}}$"

    if ed.scale != 1.0:

        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})


def fDssqrtmDs_hqet_m0(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m0")
def fDssqrtmDs_hqet_m1(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m1")
def fDssqrtmDs_hqet_m2(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m2")
def fDssqrtmDs_hqet_m3(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m3")
def fDssqrtmDs_hqet_m4(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m4")
def fDssqrtmDs_hqet_m5(ed, options):
    return fDssqrtmDs_hqet(ed, options, heavy="m5")

def fhssqrtmhs_hqet(ed, options):
    fdata = ed.fhs(div=True)
    mdata = ed.get_mass("heavy-s", div=True)

    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$\hat{f}_{hs}\, \sqrt{\hat{m}_{hs}}$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}



def fDssqrtmDs_ratio(ed, options):
    fdata = ed.fhs()
    mdata = ed.get_mass("heavy-s")

    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = {}
    for i in range(1, len(fdata)):
        m = "m{}".format(i)
        mm = "m{}".format(i-1)
        ratiodata[m] = data[m] / data[mm]

    label = "$\\frac{f_{hs}\, \sqrt{m_{hs}}}{f_{h^{-1}s}\, \sqrt{m_{h^{-1}}}}$"
    # if ed.scale != 1.0:

    #     label += " [MeV^(3/2)]"
    phys =  {"HQL": 1.0}
    return {m: (d.mean(), d.std(), label, phys) for m,d in ratiodata.iteritems()}
