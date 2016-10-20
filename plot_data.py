import logging
import argparse
import os
import numpy as np
from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon
from data_params import Zs, Zv
import matplotlib.pyplot as plt
from residualmasses import residual_mass, residual_mass_errors

from msbar_convert import get_matm
from alpha_s import get_Cmu_mbar


class plot_data(object):

    def __init__(self, value, error, label=None, physical=None):
        self.value = value
        self.error = error
        self.label = label
        if label is None:
            self.label = ""
        self.physical = physical
        if physical is None:
            self.physical = {}


def package_heavies(datas):
    pass

def get_data(ed, data_type, options):

    def dataindex():
        num = 0
        while num < 100:
            yield num
            num += 1

    if data_type == "mpi":
        data = ed.pion_mass()

        label = "$m_\pi$"
        if ed.scale != 1.0:
            label += " [MeV]"
        print data
        return plot_data(data.mean(), data.std(),
                         label, {"PDG": phys_pion})

    if data_type == "asqr":
        data = ed.ep.latspacing**2

        label = "$a$"
        return plot_data(data, 0,
                         label)


    if data_type == "mpisqr":
        pdata = ed.pion_mass()
        data = (pdata**2)
        label = "$m_\pi^2 $"
        if ed.scale != 1.0:
            label += " [MeV^2]"
        return plot_data(data.mean(), data.std(),
                         label, {"PDG": phys_pion**2})


    if data_type == "2mksqr_mpisqr":
        kdata = ed.kaon_mass()
        pdata = ed.pion_mass()
        data = 2.0*(kdata**2) - (pdata**2)
        label = "$2m_K^2 - m_\pi^2 $"
        if ed.scale != 1.0:
            label += " [MeV^2]"
        return plot_data(data.mean(), data.std(),
                         label, {"PDG": 2.0*phys_kaon**2 - phys_pion**2})


    if data_type == "mud":
        data = ed.ep.ud_mass + ed.ep.residual_mass
        err = ed.ep.residual_mass_error
        label = "$m_{ud}$"
        data = ed.scale*data
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data, err, label, {"PDG": phys_mq})

    if data_type == "ms":
        data = ed.ep.s_mass + ed.ep.residual_mass
        err = ed.ep.residual_mass_error
        label = "$m_{s}$"
        data = ed.scale*data
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data, err, label)

    if data_type == "ms_renorm":
        data = (ed.ep.s_mass + ed.ep.residual_mass)/ed.ep.Zs
        err = (ed.ep.residual_mass_error)/ed.ep.Zs
        label = "$m_{s}/Z_{s}$"
        data = ed.scale*data
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data, err, label)

    if data_type == "mh_renorm":
        data = (ed.ep.heavyq_mass + ed.ep.residual_mass)/ed.ep.Zs
        err = (ed.ep.residual_mass_error)/ed.ep.Zs
        label = "$m_{h}/Z_{s}$"
        data = ed.scale*data
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data, err, label)



    if data_type == "mD":
        data = ed.D_mass()
        label = "$m_{hl}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, {"Charm": phys_D, "Bottom": phys_MB})

    if data_type == "fD":
        data = ed.fD()

        label = "$f_D$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, {"Charm": phys_FD, "Bottom": phys_FB})

    if data_type == "fm1":
        data = ed.fD(heavy="m1")

        label = "$f_{hl}^{m1}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )

    if data_type == "fm2":
        data = ed.fD(heavy="m2")

        label = "$f_{hl}^{m2}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fm3":
        data = ed.fD(heavy="m3")

        label = "$f_{hl}^{m3}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fm4":
        data = ed.fD(heavy="m4")

        label = "$f_{hl}^{m4}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fm5":
        data = ed.fD(heavy="m5")

        label = "$f_{hl}^{m5}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )

    if data_type == "fhl":
        data = ed.fhl()

        label = "$f_{hl}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        phys = {"Charm": phys_FD, "Bottom": phys_FB}
        pds = {m: plot_data(d.mean(), d.std(), label, phys) for m, d in data.iteritems()}
        return pds

    if data_type == "fDs":
        data = ed.fDs()

        label = "$f_{Ds}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, {"Charm": phys_FD, "Bottom": phys_FB})

    if data_type == "fDs_m1":
        data = ed.fDs(heavy="m1")

        label = "$f_{hs}^{m1}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )

    if data_type == "fDs_m2":
        data = ed.fDs(heavy="m2")

        label = "$f_{hs}^{m2}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fDs_m3":
        data = ed.fDs(heavy="m3")

        label = "$f_{hs}^{m3}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fDs_m4":
        data = ed.fDs(heavy="m4")

        label = "$f_{hs}^{m4}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )
    if data_type == "fDs_m5":
        data = ed.fDs(heavy="m5")

        label = "$f_{hs}^{m5}$"
        if ed.scale != 1.0:
            label += " [MeV]"
        return plot_data(data.mean(), data.std(),
                         label, )

    if data_type == "fDssqrtmDs_m0":
        fdata = ed.fDs(heavy="m0")
        mdata = ed.get_mass("heavy-s", heavy="m0")

        data = fdata*np.sqrt(mdata)

        label = "$f_{hs}\, \sqrt{m_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

    if data_type == "fDssqrtmDs_m1":
        fdata = ed.fDs(heavy="m1")
        mdata = ed.get_mass("heavy-s", heavy="m1")

        data = fdata*np.sqrt(mdata)

        label = "$f^{m1}_{hs}\, \sqrt{m^{m1}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

    if data_type == "fDssqrtmDs_m2":
        fdata = ed.fDs(heavy="m2")
        mdata = ed.get_mass("heavy-s", heavy="m2")

        data = fdata*np.sqrt(mdata)

        label = "$f^{m2}_{hs}\, \sqrt{m^{m2}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})
    if data_type == "fDssqrtmDs_m3":
        fdata = ed.fDs(heavy="m3")
        mdata = ed.get_mass("heavy-s", heavy="m3")

        data = fdata*np.sqrt(mdata)

        label = "$f^{m3}_{hs}\, \sqrt{m^{m3}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})
    if data_type == "fDssqrtmDs_m4":
        fdata = ed.fDs(heavy="m4")
        mdata = ed.get_mass("heavy-s", heavy="m4")

        data = fdata*np.sqrt(mdata)

        label = "$f^{m4}_{hs}\, \sqrt{m^{m4}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})
    if data_type == "fDssqrtmDs_m5":
        fdata = ed.fDs(heavy="m5")
        mdata = ed.get_mass("heavy-s", heavy="m5")

        data = fdata*np.sqrt(mdata)

        label = "$f^{m5}_{hs}\, \sqrt{m^{m5}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return plot_data(data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)})

    raise RuntimeError("{} not supported as a data type yet".format(data_type))
