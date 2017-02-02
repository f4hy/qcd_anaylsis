from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon

import numpy as np


def interpolated_fsqrtm(ed, options, heavy="m0"):
    dfs = ed.select_data("heavy-s", operator="fsqrtm_continuum", smearing="0_0")
    data = {}
    for i in range(len(dfs)):
        m = "m{}".format(i)
        data[m] = next(v.values[v.values.columns[0]] for k, v in dfs.iteritems() if m in k)

    label = "$f_{hs}\, \sqrt{m_{hs}}$"

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs * np.sqrt(phys_Ds), "Bottom": phys_FBs * np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys) for m, d in data.iteritems()}


def interpolated_fsqrtm_hqet(ed, options, heavy="m0"):
    dfs = ed.select_data("heavy-s", operator="fsqrtm_hqet_continuum", smearing="0_0")
    data = {}
    for i in range(len(dfs)):
        m = "m{}".format(i)
        data[m] = next(v.values[v.values.columns[0]] for k, v in dfs.iteritems() if m in k)

    label = '$f^{{HQET}}_{hs}\, \sqrt{m^{{HQET}}_{hs}}$'

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs * np.sqrt(phys_Ds), "Bottom": phys_FBs * np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys) for m, d in data.iteritems()}


def interpolated_fsqrtm_linearhqet(ed, options, heavy="m0"):
    dfs = ed.select_data("heavy-s", operator="fsqrtm_linearhqet_continuum", smearing="0_0")
    data = {}
    for i in range(len(dfs)):
        m = "m{}".format(i)
        data[m] = next(v.values[v.values.columns[0]] for k, v in dfs.iteritems() if m in k)

    label = '$\mathrm{linear } f^{{HQET}}_{hs}\, \sqrt{m^{{HQET}}_{hs}}$'

    if ed.scale != 1.0:
        label += " [MeV^(3/2)]"

    phys = {"Charm": phys_FDs * np.sqrt(phys_Ds), "Bottom": phys_FBs * np.sqrt(phys_MBs)}
    return {m: (d.mean(), d.std(), label, phys) for m, d in data.iteritems()}
