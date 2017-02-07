from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


import numpy as np

from collections import OrderedDict


def fDsqrtmD(ed, options, heavy="m0"):
    fdata = ed.fD(heavy=heavy)
    mdata = ed.get_mass("heavy-ud", heavy=heavy)

    data = fdata*np.sqrt(mdata)

    label = "$f^{"+heavy+"}_{hl}\, \sqrt{m^{"+heavy+"}_{hl}}$"

    if ed.scale != 1.0:

        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})

def fDsqrtmD_m0(ed, options):
    return fDsqrtmD(ed, options, heavy="m0")
def fDsqrtmD_m1(ed, options):
    return fDsqrtmD(ed, options, heavy="m1")
def fDsqrtmD_m2(ed, options):
    return fDsqrtmD(ed, options, heavy="m2")
def fDsqrtmD_m3(ed, options):
    return fDsqrtmD(ed, options, heavy="m3")
def fDsqrtmD_m4(ed, options):
    return fDsqrtmD(ed, options, heavy="m4")
def fDsqrtmD_m5(ed, options):
    return fDsqrtmD(ed, options, heavy="m5")

def fhlsqrtmhl(ed, options):
    fdata = ed.fhl()
    mdata = ed.get_mass("heavy-ud")

    data = OrderedDict()
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$f_{hl}\, \sqrt{m_{hl}}$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}



def fDsqrtmD_ratio(ed, options):
    fdata = ed.fhl()
    mdata = ed.get_mass("heavy-ud")

    data = OrderedDict()
    for m in fdata:
        mkey = next(k for k in mdata if m in k)
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = OrderedDict()
    for i in range(1, len(fdata)):
        m = "m{}".format(i)
        mm = "m{}".format(i-1)
        ratiodata[m] = data[m] / data[mm]

    label = "$\\frac{f_{hl}\, \sqrt{m_{hl}}}{f_{h^{-1}l}\, \sqrt{m_{h^{-1}l}}}$"
    # if options.scale:
    #     label += " [MeV^(3/2)]"
    phys =  {"HQL": 1.0}
    return {m: (d.mean(), d.std(), label, phys) for m,d in ratiodata.iteritems()}

def fDsqrtmD_ratio_div(ed, options):
    fdata = ed.fhl(div=True)
    mdata = ed.get_mass("heavy-ud",div=True)

    data = OrderedDict()
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = OrderedDict()
    for i in range(1, len(fdata)):
        m = "m{}".format(i)
        mm = "m{}".format(i-1)
        ratiodata[m] = data[m] / data[mm]

    label = "$\\frac{f^{div}_{hl}\, \sqrt{m^{div}_{hl}}}{f^{div}_{h^{-1}l}\, \sqrt{m^{div}_{h^{-1}l}}}$"
    # if options.scale:
    #     label += " [MeV^(3/2)]"
    phys =  {"HQL": 1.0}
    return {m: (d.mean(), d.std(), label, phys) for m,d in ratiodata.iteritems()}

def fhlsqrtmhl_hqet(ed, options):
    fdata = ed.fhl(div=True, matched=True)
    mdata = ed.get_mass("heavy-ud", div=True)

    data = OrderedDict()
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$\hat{f}_{hl}\, \sqrt{\hat{m}_{hl}} / C(\mu)$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}


def fhssqrtmhs_hqet(ed, options):
    fdata = ed.fhs(div=True, matched=True)
    mdata = ed.get_mass("heavy-s", div=True)
    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$\hat{f}_{hs}\, \sqrt{\hat{m}_{hs}} / C(\mu)$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}
