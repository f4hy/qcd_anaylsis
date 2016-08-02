import logging
import argparse
import os
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor
from ensamble_info import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from ensamble_info import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs
from ensamble_info import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon
from ensamble_info import Zs, Zv
from ensamble_info import unphys_etas
import matplotlib.pyplot as plt
from residualmasses import residual_mass, residual_mass_errors

from msbar_convert import get_matm
from alpha_s import get_Cmu_mbar


def get_data(ed, data_type, options):

    def dataindex():
        num = 0
        while num < 100:
            yield num
            num += 1




    if data_type == "index":
        data = dataindex()
        err = 0
        label = "index"
        return data, err, label, {"": 0}

    if data_type == "fDsqrtmD":
        fdata = ed.fD(scaled=options.scale)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$f_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fDsqrtmD_ratio":
        fdata_ratio = ed.fD_ratio(scaled=options.scale)
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale)

        data = fdata_ratio*np.sqrt(mdata_ratio)

        label = "$\\frac{f_{h^{+1}l}\, \sqrt{m_{h^{+1}l}}}{f_{hl}\, \sqrt{m_{hl}}}$"
        # if options.scale:
        #     label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"HQL": 1.0}

    if data_type == "fDssqrtmDs_ratio":
        fdata_ratio = ed.fDs_ratio(scaled=options.scale)
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale)

        data = fdata_ratio*np.sqrt(mdata_ratio)
        #data =  fdata_ratio*np.sqrt(mdata_ratio)

        label = "$\\frac{f_{h^{+1}s}\, \sqrt{m_{h^{+1}s}}}{f_{hs}\, \sqrt{m_{hs}}}$"
        # if options.scale:
        #     label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"HQL": 1.0}


    if data_type == "fD_divsqrtmD":
        fdata = ed.fD(scaled=options.scale, div=True)
        mdata = ed.D_mass_div(scaled=options.scale)

        data = fdata*np.sqrt(mdata)


        label = "${f}_{hl}\, \sqrt{{m}_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}


    if data_type == "fDs_divsqrtmDs":
        fdata = ed.fDs(scaled=options.scale, div=True)
        mdata = ed.Ds_mass_div(scaled=options.scale)

        data = fdata*np.sqrt(mdata)


        label = "${f}_{hs}\, \sqrt{{m}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}


    if data_type == "fDsqrtmD_renorm":
        fdata = ed.fD(scaled=options.scale, renorm=True)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$\widetilde{f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}


    if data_type == "Bs_B_raw":
        fdata = ed.fD(scaled=options.scale)
        mdata = ed.D_mass(scaled=options.scale)
        fsdata = ed.fDs(scaled=options.scale)
        msdata = ed.Ds_mass(scaled=options.scale)

        datahl = fdata*np.sqrt(mdata)
        datahs = fsdata*np.sqrt(msdata)

        data = datahs / datahl

        label = "$\widetilde{f}_{hs}\, \sqrt{m_{hs}}$ / $\widetilde{f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds)/(phys_FD*np.sqrt(phys_D)),
                                                "Bottom": phys_FBs*np.sqrt(phys_MBs)/(phys_FB*np.sqrt(phys_MB))}

    if data_type == "Bs_B_matched":
        fdata = ed.fD(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata = ed.D_mass(scaled=options.scale)
        fsdata = ed.fDs(scaled=options.scale, renorm=True, div=True, matched=True)
        msdata = ed.Ds_mass(scaled=options.scale)

        datahl = fdata*np.sqrt(mdata)
        datahs = fsdata*np.sqrt(msdata)

        data = datahs / datahl

        label = "$\widetilde{f}_{hs}\, \sqrt{m_{hs}}$ / $\widetilde{f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds)/(phys_FD*np.sqrt(phys_D)),
                                                "Bottom": phys_FBs*np.sqrt(phys_MBs)/(phys_FB*np.sqrt(phys_MB))}


    if data_type == "fBs_fB_raw":
        fdata = ed.fD(scaled=options.scale)
        fsdata = ed.fDs(scaled=options.scale)

        datahl = fdata
        datahs = fsdata

        data = datahs / datahl

        label = "$\widetilde{f}_{hs}$ / $\widetilde{f}_{hl}$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm": phys_FDs/(phys_FD),
                                                "Bottom": phys_FBs/(phys_FB)}

    if data_type == "fBs_fB_matched":
        fdata = ed.fD(scaled=options.scale, renorm=True, div=True, matched=True)
        fsdata = ed.fDs(scaled=options.scale, renorm=True, div=True, matched=True)

        datahl = fdata
        datahs = fsdata

        data = datahs / datahl

        label = "$\widetilde{f}_{hs}$ / $\widetilde{f}_{hl}$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm": phys_FDs/(phys_FD),
                                                "Bottom": phys_FBs/(phys_FB)}


    if data_type == "fDsqrtmD_raw":
        fdata = ed.fD(scaled=options.scale)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$\widetilde{f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}


    if data_type == "fDAsqrtmD_raw":
        fdata = ed.fDA(scaled=options.scale)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$\widetilde{f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fDsAsqrtmDs_raw":
        fdata = ed.fDsA(scaled=options.scale)
        mdata = ed.Ds_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$\widetilde{f}_{hs}\, \sqrt{m_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}




    if data_type == "fDAsqrtmD_renorm":
        fdata = ed.fDA(scaled=options.scale, renorm=True)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$f_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fDsAsqrtmDs_renorm":
        fdata = ed.fDsA(scaled=options.scale, renorm=True)
        mdata = ed.Ds_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$f_{hs}\, \sqrt{m_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}


    if data_type == "fD_divsqrtmD_renorm":
        fdata = ed.fD(scaled=options.scale, renorm=True, div=True)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fD_divsqrtmD":
        fdata = ed.fD(scaled=options.scale, renorm=False, div=True)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}


    if data_type == "fD_divsqrtmD_renorm_matched":
        fdata = ed.fD(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata = ed.D_mass_div(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{m_{hl}} / C(\mu)$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D)/get_Cmu_mbar(1080.0) , "Bottom": phys_FB*np.sqrt(phys_MB)/get_Cmu_mbar(4100.0)}

    if data_type == "fDs_divsqrtmDs_renorm_matched":
        fdata = ed.fDs(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata = ed.Ds_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hs}\, \sqrt{m_{hs}} / C(\mu)$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds)/get_Cmu_mbar(1080.0) , "Bottom": phys_FBs*np.sqrt(phys_MBs)/get_Cmu_mbar(4100.0)}

    if data_type == "fDA_divsqrtmD_renorm_matched":
        fdata = ed.fDA(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata = ed.DA_mass_div(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{m_{hl}} / C(\mu)$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D)/get_Cmu_mbar(1080.0) , "Bottom": phys_FB*np.sqrt(phys_MB)/get_Cmu_mbar(4100.0)}

    if data_type == "fDsA_divsqrtmDs_renorm_matched":
        fdata = ed.fDsA(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata = ed.DsA_mass_div(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hs}\, \sqrt{m_{hs}} / C(\mu)$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds)/get_Cmu_mbar(1080.0) , "Bottom": phys_FBs*np.sqrt(phys_MBs)/get_Cmu_mbar(4100.0)}



    if data_type == "fD_divsqrtmD_renorm_matched_ratio":
        fdata_ratio = ed.fD_ratio(scaled=options.scale, renorm=True, div=True, matched=True)
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale)

        data = fdata_ratio*np.sqrt(mdata_ratio)

        label = "$\\frac{{f}_{h^{+1}l}\, \sqrt{m_{h^{+1}l}} }{ {f}_{hl}\, \sqrt{m_{hl}} } \\frac{C(\\bar{m}_q)}{C(\\bar{m}_q^{+1})})}$"
        return data.mean(), data.std(), label, {"HQL": 1.0}


    if data_type == "fD_divsqrtmD_renorm_ratio":
        fdata_ratio = ed.fD_ratio(scaled=options.scale, renorm=True, div=True)
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale)

        data = fdata_ratio*np.sqrt(mdata_ratio)

        label = "$\\frac{{f}_{h^{+1}l}\, \sqrt{m_{h^{+1}l}} }{ {f}_{hl}\, \sqrt{m_{hl}} }$"
        return data.mean(), data.std(), label, {"HQL": 1.0}


    if data_type == "fDs_div_renorm_ratio":
        fdata_ratio = ed.fDs_ratio(scaled=options.scale, renorm=True, div=True)
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale)


        data = np.sqrt(1.25)* fdata_ratio

        label = "$ \sqrt{1.25} \\frac{{f}_{h^{+1}s} }{ {f}_{hs} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}


    if data_type == "fDs_divsqrtmDs_renorm_ratio":
        fdata_ratio = ed.fDs_ratio(scaled=options.scale, renorm=True, div=True)
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale)

        data = fdata_ratio*np.sqrt(mdata_ratio)

        label = "$ \\frac{{f}_{h^{+1}s}\, \sqrt{m_{h^{+1}s}} }{ {f}_{hs}\, \sqrt{m_{hs}} }$"
        return data.mean(), data.std(), label, {"HQL": 1.0}


    if data_type == "mD_ratio":
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale, corrected=False)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio / 1.25

        label = "$  \\frac{1}{1.25}\\frac{m_{h^{+1}\ell} }{ m_{h\ell} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}


    if data_type == "mD_corrected_ratio":
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale, corrected=True)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio / 1.25

        label = "$  \\frac{1}{1.25}\\frac{\widetilde{m}_{h^{+1}\ell} }{ \widetilde{m}_{h\ell} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}


    if data_type == "mD_pole_ratio":
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale, corrected=False)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio

        m1 = ed.dp.heavyq_mass*scale[ed.dp.beta]
        m2 = ed.dp.heavyq_mass_next*scale[ed.dp.beta]

        rho_mu = get_matm(m1, m1)
        rho_mu_next = get_matm(m2, m2)

        data = data / (rho_mu_next / rho_mu)

        label = "$  \\frac{m_{q}^{m(m)}}{m_{q^{+1}}^{m(m)}} \\frac{{m}_{h^{+1}\ell} }{ {m}_{h\ell} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}



    if data_type == "mD_corrected_pole_ratio":
        mdata_ratio = ed.D_mass_ratio(scaled=options.scale, corrected=True)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio

        m1 = ed.dp.heavyq_mass*scale[ed.dp.beta]
        m2 = ed.dp.heavyq_mass_next*scale[ed.dp.beta]

        rho_mu = get_matm(m1, m1)
        rho_mu_next = get_matm(m2, m2)

        data = data / (rho_mu_next / rho_mu)

        label = "$  \\frac{m_{q}^{m(m)}}{m_{q^{+1}}^{m(m)}} \\frac{\widetilde{m}_{h^{+1}\ell} }{ \widetilde{m}_{h\ell} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}



    if data_type == "mDs_ratio":
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale, corrected=False)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio / 1.25

        label = "$  \\frac{1}{1.25}\\frac{m_{h^{+1}s} }{ m_{hs} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}


    if data_type == "mDs_corrected_ratio":
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale, corrected=True)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio / 1.25

        label = "$  \\frac{1}{1.25}\\frac{\widetilde{m}_{h^{+1}s} }{ \widetilde{m}_{hs} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}


    if data_type == "mDs_pole_ratio":
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale, corrected=False)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio

        m1 = ed.dp.heavyq_mass*scale[ed.dp.beta]
        m2 = ed.dp.heavyq_mass_next*scale[ed.dp.beta]

        rho_mu = get_matm(m1, m1)
        rho_mu_next = get_matm(m2, m2)

        data = data / (rho_mu_next / rho_mu)

        label = "$  \\frac{m_{q}^{m(m)}}{m_{q^{+1}}^{m(m)}} \\frac{{m}_{h^{+1}s} }{ {m}_{hs} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}



    if data_type == "mDs_corrected_pole_ratio":
        mdata_ratio = ed.Ds_mass_ratio(scaled=options.scale, corrected=True)
        if np.all(np.isnan(mdata_ratio)):
            mdata_ratio = np.mean(mdata_ratio)
        data = mdata_ratio

        m1 = ed.dp.heavyq_mass*scale[ed.dp.beta]
        m2 = ed.dp.heavyq_mass_next*scale[ed.dp.beta]

        rho_mu = get_matm(m1, m1)
        rho_mu_next = get_matm(m2, m2)

        data = data / (rho_mu_next / rho_mu)

        label = "$  \\frac{m_{q}^{m(m)}}{m_{q^{+1}}^{m(m)}} \\frac{\widetilde{m}_{h^{+1}s} }{ \widetilde{m}_{hs} }$"
        return np.mean(data), np.std(data), label, {"HQL": 1.0}




    if data_type == "fDA_divsqrtmD_renorm":
        fdata = ed.fDA(scaled=options.scale, renorm=True, div=True)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fDsA_divsqrtmDs_renorm":
        fdata = ed.fDsA(scaled=options.scale, renorm=True, div=True)
        mdata = ed.Ds_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hs}\, \sqrt{m_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}


    if data_type == "fDAsqrtmD":
        fdata = ed.fDA(scaled=options.scale)
        mdata = ed.D_mass(scaled=options.scale)

        data = fdata*np.sqrt(mdata)

        label = "$f_{hl}\, \sqrt{m_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}


    if data_type == "fDA_divsqrtmD":
        fdata = ed.fDA(scaled=options.scale, div=True)
        mdata = ed.D_mass_div(scaled=options.scale)

        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hl}\, \sqrt{{m}_{hl}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FD*np.sqrt(phys_D), "Bottom": phys_FB*np.sqrt(phys_MB)}

    if data_type == "fDsA_divsqrtmDs":
        fdata = ed.fDsA(scaled=options.scale, div=True)
        mdata = ed.Ds_mass_div(scaled=options.scale)

        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2

        data = fdata*np.sqrt(mdata)

        label = "${f}_{hs}\, \sqrt{{m}_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}


    if data_type == "fDssqrtmDs":
        fdata = ed.fDs(scaled=options.scale)
        mdata = ed.Ds_mass(scaled=options.scale)


        data = fdata*np.sqrt(mdata)

        label = "$f_{hs}\, \sqrt{m_{hs}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs*np.sqrt(phys_Ds), "Bottom": phys_FBs*np.sqrt(phys_MBs)}


    if data_type == "fD":
        data = ed.fD(scaled=options.scale)

        label = "$f_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}

    if data_type == "fDA":
        data = ed.fDA(scaled=options.scale)

        label = "$f_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}


    if data_type == "fD_new":
        data = ed.fD(scaled=options.scale)

        label = "$f_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}

    if data_type == "fD_new_div":
        data = ed.fD(scaled=options.scale, div=True)

        label = "${f}_D$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}



    if data_type == "fD_new_renorm":
        data = ed.fD(scaled=options.scale, renorm=True)

        label = "$f_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}

    if data_type == "fD_new_renorm_div":
        data = ed.fD(scaled=options.scale, renorm=True, div=True)

        label = "${f}_D$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}



    if data_type == "fD_div":
        data = ed.fD(scaled=options.scale, div=True)

        label = "${f}_D$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}


    if data_type == "fD_axial":
        data = ed.fDA(scaled=options.scale)
        label = "$f_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}

    if data_type == "fD_axial_new":
        data = ed.fDA(scaled=options.scale)
        label = "$f_D^A$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}


    if data_type == "fD_axial_div":
        data = ed.fDA(scaled=options.scale, div=True)

        label = "${f}_D^A$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FD, "Bottom": phys_FB}


    if data_type == "fD_axialratio":
        dataA = ed.fDA(scaled=options.scale)
        dataP = ed.fD(scaled=options.scale)

        data = (dataP/dataA)

        label = "$f_{hl}/f_{hl}^A$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {}

    if data_type == "fDs":
        data = ed.fDs(scaled=options.scale)

        label = "$f_{hs}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs, "Bottom": phys_FBs}

    if data_type == "fDs_axial":
        data = ed.fDsA(scaled=options.scale)

        label = "$f_{D_s}^A$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs, "Bottom": phys_FBs}

    if data_type == "fDsA":
        data = ed.fDsA(scaled=options.scale)

        label = "$f_{D_s}^A$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_FDs, "Bottom": phys_FBs}


    if data_type == "fDs_axialratio":
        dataA = ed.fDsA(scaled=options.scale)
        dataP = ed.fDs(scaled=options.scale)
        data = (dataP/dataA)

        label = "$f_{D_s}/f_{D_s}^A$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {}

    if data_type == "fHH":
        data = ed.fHH(scaled=options.scale)

        label = "$f_{HH}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {}



    if data_type == "fDsbyfD":
        data = (ed.fDs(scaled=options.scale)/ed.fD(scaled=options.scale))

        label = "$f_{D_s}/f_D$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm":phys_FDs/phys_FD,
                                                "Bottom":phys_FBs/phys_FB}

    if data_type == "fDsAbyfDA":
        data = (ed.fDsA(scaled=options.scale)/ed.fDA(scaled=options.scale))

        label = "$f_{D_s}/f_D$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"Charm":phys_FDs/phys_FD,
                                                "Bottom":phys_FBs/phys_FB}


    if data_type == "fKbyfpi":
        data = (ed.fK(scaled=options.scale)/ed.fpi(scaled=options.scale))

        label = "$f_K/f_\pi$"
        if options.scale:
            label += " "
        return data.mean(), data.std(), label, {"PDG": phys_FK/phys_Fpi}


    if data_type == "fK":
        data = ed.fK(scaled=options.scale)

        label = "$f_K$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_FK}

    if data_type == "fK_new":
        data = ed.fK()

        label = "$f_K$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_FK}



    if data_type == "fpi":
        data = ed.fpi(scaled=options.scale)

        label = "$f_\pi$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_Fpi}



    if data_type == "fpi_new":
        data = ed.fpi()

        label = "$f_\pi$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_Fpi}


    if data_type == "mpi":
        data = ed.pion_mass(scaled=options.scale)

        label = "$m_\pi$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_pion}

    if data_type == "meta":
        data = ed.eta_mass(scaled=options.scale)

        label = "$m_{eta_s}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"unphysical $s\\bar{s}$": unphys_etas}


    if data_type == "mk":
        data = ed.kaon_mass(scaled=options.scale)
        label = "$m_K$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"PDG": phys_kaon}

    if data_type == "2mksqr_mpisqr":
        kdata = ed.kaon_mass(scaled=options.scale)
        pdata = ed.pion_mass(scaled=options.scale)
        data = 2.0*(kdata**2) - (pdata**2)
        label = "$2m_K^2 - m_\pi^2 $"
        if options.scale:
            label += " [MeV^2]"
        return data.mean(), data.std(), label, {"PDG": 2.0*phys_kaon**2 - phys_pion**2}


    if data_type == "mHH":
        data = ed.HH_mass(scaled=options.scale)
        label = "$m_{HH}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_etac,
                                  "Bottom": phys_etab}

    if data_type == "mHHv":
        data = ed.HHv_mass(scaled=options.scale)
        label = "$m_{HH}^v$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_Jpsi}


    if data_type == "mHH_spinave":
        vdata = ed.HHv_mass(scaled=options.scale)
        pdata = ed.HH_mass(scaled=options.scale)

        data = (pdata + 3.0*vdata)/4.0
        label = '$(3 M_{J/\psi} + M_{\eta_c})/4$'
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": (3*phys_Jpsi + phys_etac)/4.0 ,
                                  "Bottom": (3*phys_Jpsi + phys_etac)/4.0 }

    if data_type == "1/mHH_spinave":
        vdata = ed.HHv_mass(scaled=options.scale)
        pdata = ed.HH_mass(scaled=options.scale)


        data = 4.0/(pdata + 3.0*vdata)
        label = '$1.0/\\bar{M}_{HH}$'
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm":1.0/( (3*phys_Jpsi + phys_etac)/4.0),
                                  "Bottom":1.0/( (3*phys_Upsilon + phys_etab)/4.0)}


    if data_type == "mHH_spindiff":
        vdata = ed.HHv_mass(scaled=options.scale)
        pdata = ed.HH_mass(scaled=options.scale)

        data = vdata - pdata
        label = "$m_{HH}^{V} - m_{HH}^{P}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_Jpsi - phys_etac,
                                  "Bottom": phys_Upsilon - phys_etab}



    if data_type == "1/mHH":
        data = 1/ed.HH_mass(scaled=options.scale)
        label = "$1.0/m_{HH}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_etac, "Bottom": 1.0/phys_etab}


    if data_type == "HL_diff":
        HHdata = ed.HH_mass(scaled=options.scale)
        HLdata = ed.D_mass(scaled=options.scale)
        LLdata = ed.pion_mass(scaled=options.scale)

        HLdiff = HLdata - ((HHdata + LLdata) / 2.0)


        label = "$m_{Hl} - (m_{HH} + m_{ll})/2$"
        if options.scale:
            label += " [MeV]"
        return HLdiff.mean(), HLdiff.std(), label, {"Charm": phys_D - (phys_etac +phys_pion)/2.0,
                                                    "Bottom": phys_MB - (phys_etab +phys_pion)/2.0}

    if data_type == "Hs_diff":
        HHdata = ed.HH_mass(scaled=options.scale)
        HLdata = ed.Ds_mass(scaled=options.scale)
        LLdata = ed.eta_mass(scaled=options.scale)

        HLdiff = HLdata - ((HHdata + LLdata) / 2.0)

        label = "$m_{Hs} - (m_{HH} + m_{s\\bar{s}})/2.0$"
        if options.scale:
            label += " [MeV]"
        return HLdiff.mean(), HLdiff.std(), label, {"Charm": phys_Ds - (phys_etac + unphys_etas)/2.0,
                                                    "Bottom": phys_MBs - (phys_etab + unphys_etas)/2.0}

    if data_type == "MHs_MHH":
        HHdata = ed.HH_mass(scaled=options.scale)
        HLdata = ed.Ds_mass(scaled=options.scale)
        LLdata = ed.eta_mass(scaled=options.scale)

        HLdiff = HLdata - ((HHdata) / 2.0)

        label = "$m_{Hs} - (m_{HH})/2.0$"
        if options.scale:
            label += " [MeV]"
        return HLdiff.mean(), HLdiff.std(), label, {"Charm": phys_Ds - (phys_etac)/2.0,
                                      "Bottom": phys_MBs - (phys_etab)/2.0}



    if data_type == "1/mD":
        mdata = ed.D_mass(scaled=options.scale)

        data = 1.0/mdata

        label = "$1/m_{hl}$"
        if options.scale:
            label += " [1/MeV]"

        return data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB}

    if data_type == "1/mDs":
        mdata = ed.Ds_mass(scaled=options.scale)

        data = 1.0/mdata

        label = "$1/m_{hs}$"
        if options.scale:
            label += " [1/MeV]"
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MBs}

    if data_type == "1/mDA":
        mdata = ed.D_mass_axial(scaled=options.scale)

        data = 1.0/mdata

        label = "$1/m_{hl}$"
        if options.scale:
            label += " [1/MeV]"

        return data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB}

    if data_type == "1/mDsA":
        mdata = ed.Ds_mass_axial(scaled=options.scale)

        data = 1.0/mdata

        label = "$1/m_{hs}$"
        if options.scale:
            label += " [1/MeV]"
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MBs}


    if data_type == "1/mD_corrected":
        mdata = ed.D_mass(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/(m_{hl} + m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1)*scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB}

    if data_type == "1/mD_div_corrected":
        mdata = ed.D_mass_div(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/({m}_{hl} + m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1)*scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB}

    if data_type == "1/mDs_div_corrected":
        mdata = ed.Ds_mass_div(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/({m}_{hs} + m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1) *scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MB}

    if data_type == "1/mDA_div_corrected":
        mdata = ed.DA_mass_div(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/({m}_{hl} + m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1)*scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB}

    if data_type == "1/mDsA_div_corrected":
        mdata = ed.DsA_mass_div(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/({m}_{hs} + m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1) *scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MB}


    if data_type == "1/mDs_corrected":
        mdata = ed.Ds_mass(scaled=options.scale)
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = 1.0/(mdata +(m2 - m1))
        label = "$1/(m_{hs}+m_2 - m_1)$"
        if options.scale:
            label += " [1/MeV]"
            data = 1.0/(mdata +(m2 - m1)*scale[ed.dp.beta])
        return data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MBs}

    if data_type == "mD":
        data = ed.D_mass(scaled=options.scale)
        label = "$m_{hl}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_D, "Bottom": phys_MB}

    if data_type == "mD_corrected":
        mdata = ed.D_mass(scaled=options.scale)
        m = ed.dp.heavyq_mass
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = mdata + (m2 - m1)
        label = "$m_{hl} + m_2 - m_1$"
        if options.scale:
            label += " [MeV]"
            data = mdata + (m2 - m1)*scale[ed.dp.beta]
        return data.mean(), data.std(), label, {"Charm": phys_D, "Bottom": phys_MB}

    if data_type == "m1_m2":
        data = ed.D_mass(scaled=options.scale)
        m = ed.dp.heavyq_mass
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = (m2 - m1)
        label = "$m_2 - m_1$"
        if options.scale:
            label += " [MeV]"
            data = data*scale[ed.dp.beta]
        return data.mean(), data.std(), label, {"": None}


    if data_type == "mD_div_corrected":
        mdata = ed.D_mass_div(scaled=options.scale)
        m = ed.dp.heavyq_mass
        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2
        data = mdata + (m2 - m1)
        label = "${m}_{hl} + m_2 - m_1$"
        if options.scale:
            label += " [MeV]"
            data = mdata + (m2 - m1)*scale[ed.dp.beta]
        return data.mean(), data.std(), label, {"Charm": phys_D, "Bottom": phys_MB}


    if data_type == "mDs":
        data = ed.Ds_mass(scaled=options.scale)

        label = "$m_{hs}$"
        if options.scale:
            label += " [MeV]"
        return data.mean(), data.std(), label, {"Charm": phys_Ds, "Bottom": phys_MBs}


    if data_type == "mDs_corrected":
        mdata = ed.Ds_mass(scaled=options.scale)

        m1 = ed.dp.heavy_m1
        m2 = ed.dp.heavy_m2

        data = mdata + (m2 - m1)

        label = "$m_{hs}+m_2-m_1$"
        if options.scale:
            label += " [MeV]"
            data = mdata + (m2 - m1)*scale[ed.dp.beta]

        return data.mean(), data.std(), label, {"Charm": phys_Ds, "Bottom": phys_MBs}


    if data_type == "mpisqr":
        data = ed.pion_mass(scaled=options.scale)**2

        label = "$m_\pi^2$"
        if options.scale:
            label += " [MeV^2]"
        return data.mean(), data.std(), label, {"PDG $m_\pi^2$": phys_pion**2}

    if data_type == "mKsqr":
        data = ed.kaon_mass(scaled=options.scale)**2
        label = "$m_K^2$"
        if options.scale:
            label += " [MeV^2]"
        return data.mean(), data.std(), label, {"PDG $m_K^2$": phys_kaon**2}


    if data_type == "mpisqr/mq":
        mpi = ed.pion_mass(scaled=True).mean()
        mpisqr = mpi**2
        mpierr = (ed.pion_mass(scaled=True)**2).std()
        mq = scale[ed.dp.beta] * (ed.dp.ud_mass + residual_mass(ed.dp)) / Zs[ed.dp.beta]
        res_err = scale[ed.dp.beta]*residual_mass_errors(ed.dp)

        data = mpisqr / mq
        sqr_err = ((mpierr / mq)**2 +
                   (res_err * data / (scale[ed.dp.beta]*(ed.dp.ud_mass + residual_mass(ed.dp))))**2)
        err = np.sqrt(sqr_err)
        label = "$m_\pi^2 / m_q$"
        if options.scale:
            label += " [MeV]"

        return data.mean(), data.std(), label, {"PDG": phys_pion**2 / phys_mq}

    if data_type == "mud":
        data = ed.dp.ud_mass + residual_mass(ed.dp)
        err = residual_mass_errors(ed.dp)
        label = "$m_{ud}$"
        if options.scale:
            data = scale[ed.dp.beta]*data
            label += " [MeV]"
        return data, err, label, {"PDG": phys_mq}

    if data_type == "mheavyq":
        data = ed.dp.heavyq_mass / Zs[ed.dp.beta]
        print ed.dp.latspacing
        err = 0.0
        label = "$m_{q_h}$"
        if options.scale:
            data = scale[ed.dp.beta]*data
            label += " [MeV]"
        return data, err, label, {"PDG": phys_mhq}

    if data_type == "mheavyq_bare":
        data = ed.dp.heavyq_mass
        err = 0.0
        label = "$m_{q_h}^{bare}$"
        if options.scale:
            data = scale[ed.dp.beta]*data
            label += " [MeV]"
        return data, err, label, {"PDG": phys_mhq}


    if data_type == "1/mheavyq":
        data = 1.0/(ed.dp.heavyq_mass / Zs[ed.dp.beta])
        err = 0.0
        label = "$1/\\bar{m}_{q_h}$"
        if options.scale:
            data = data / scale[ed.dp.beta]
            label += " [1/MeV]"
        return data, err, label, {"PDG": phys_mhq}


    if data_type == "xi":
        mpi = ed.pion_mass(scaled=options.scale)
        fpi = ed.fpi(scaled=options.scale)
        xi = ((mpi**2) / (8 * (np.pi**2)*(fpi**2))).mean()
        data = xi
        phys_xi = phys_pion**2 / (8 * (np.pi**2)*(phys_Fpi**2))

        return data.mean(), data.std(), "$\\xi$", {"F:flag M:pdg": phys_xi}

    if data_type == "x":
        B = 2826.1
        F_0 = 118.03
        qmass = scale[ed.dp.beta]*(ed.dp.ud_mass + residual_mass(ed.dp)) / Zs[ed.dp.beta]
        Msqr = B * (qmass + qmass)
        x = Msqr / (8 * (np.pi**2) * (F_0**2))

        data = x
        err = 0
        return data, err, "$x=B(m_q + m_q)/(4 \pi F)^2 $", {"", 0}

    raise RuntimeError("{} not supported as a data type yet".format(data_type))
