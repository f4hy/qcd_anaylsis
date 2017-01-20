from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


import numpy as np

def fDsqrtmD_m0(ed, options):
    fdata = ed.fD(heavy="m0")
    mdata = ed.get_mass("heavy-ud", heavy="m0")

    data = fdata*np.sqrt(mdata)

    label = "$f_{hl}\, \sqrt{m_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})


def fDsqrtmD_m1(ed, options):
    fdata = ed.fD(heavy="m1")
    mdata = ed.get_mass("heavy-ud", heavy="m1")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m1}_{hl}\, \sqrt{m^{m1}_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})


def fDsqrtmD_m2(ed, options):
    fdata = ed.fD(heavy="m2")
    mdata = ed.get_mass("heavy-ud", heavy="m2")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m2}_{hl}\, \sqrt{m^{m2}_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})

def fDsqrtmD_m3(ed, options):
    fdata = ed.fD(heavy="m3")
    mdata = ed.get_mass("heavy-ud", heavy="m3")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m3}_{hl}\, \sqrt{m^{m3}_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})

def fDsqrtmD_m4(ed, options):
    fdata = ed.fD(heavy="m4")
    mdata = ed.get_mass("heavy-ud", heavy="m4")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m4}_{hl}\, \sqrt{m^{m4}_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return (data.mean(), data.std(), label,
            {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})

def fDsqrtmD_m5(ed, options):
    fdata = ed.fD(heavy="m5")
    mdata = ed.get_mass("heavy-ud", heavy="m5")

    data = fdata*np.sqrt(mdata)

    label = "$f^{m5}_{hl}\, \sqrt{m^{m5}_{hl}}$"

    if options.scale:
        label += " [MeV^(3/2)]"
    return plot_data(data.mean(), data.std(), label,
                     {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)})

def fDsqrtmD_ratio(ed, options):
    fdata = ed.fhl()
    mdata = ed.get_mass("heavy-ud")

    data = {}
    for m in fdata:
        mkey = next(k for k in mdata if m in k)
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = {}
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

    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    ratiodata = {}
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

    data = {}
    for m in fdata:
        mkey = [k for k in mdata if m in k][0]
        data[m] = fdata[m]*np.sqrt(mdata[mkey])

    label = "$\hat{f}_{hl}\, \sqrt{\hat{m}_{hl}} / C(\mu)$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}
    return {m: (d.mean(), d.std(), label, phys ) for m,d in data.iteritems()}
