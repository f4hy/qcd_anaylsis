import logging
import numpy as np
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon

def ZA_fpi_fpiA(ed, options):
    dataA = ed.fpiA(scaled=options.scale) / ed.ep.Zv
    dataP = ed.fpi(scaled=options.scale)

    data = (dataP/dataA)

    label = "$f_{\pi}/f_{\pi}^A$"

    return (data.mean(), data.std(),
            label, {"4.17": 0.9553, "4.35": 0.9636, "4.47": 0.9699})


def ZA_fK_fKA(ed, options):
    dataA = ed.fKA(scaled=options.scale) / ed.ep.Zv
    dataP = ed.fK(scaled=options.scale)

    data = (dataP/dataA)

    label = "$f_{K}/f_{K}^A$"

    return (data.mean(), data.std(),
            label, {"4.17": 0.9553, "4.35": 0.9636, "4.47": 0.9699})

def ZA_fD_fDA(ed, options):
    dataA = ed.fDA(scaled=options.scale) / ed.ep.Zv
    dataP = ed.fD(scaled=options.scale)

    data = (dataP/dataA)

    label = "$f_{D}/f_{D}^A$"

    return (data.mean(), data.std(),
            label, {"4.17": 0.9553, "4.35": 0.9636, "4.47": 0.9699})

def ZA_fDs_fDsA(ed, options):
    dataA = ed.fDsA(scaled=options.scale) / ed.ep.Zv
    dataP = ed.fDs(scaled=options.scale)

    data = (dataP/dataA)

    label = "$f_{Ds}/f_{Ds}^A$"

    return (data.mean(), data.std(),
            label, {"4.17": 0.9553, "4.35": 0.9636, "4.47": 0.9699})
