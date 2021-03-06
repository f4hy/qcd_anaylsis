import logging
import numpy as np
from physical_values import phys_pion, phys_kaon, phys_mq, flag_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon




def fpi(ed, options):
    data = ed.fpi()

    label = "$f_\pi$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_Fpi})

def Fpi(ed, options):
    """ Other normaliztion of fpi"""
    data = ed.fpi() / np.sqrt(2)

    label = "$F_\pi$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_Fpi / np.sqrt(2)})


def fK(ed, options):
    data = ed.fK()

    label = "$f_K$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_FK})

def FK(ed, options):
    data = ed.fK() / np.sqrt(2)

    label = "$F_K$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_FK / np.sqrt(2)})


def fK_fpi(ed, options):
    data = ed.fK() / ed.fpi()

    label = "$f_K / f_\pi$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_FK / phys_Fpi})



def xi(ed, options):
    mpi = ed.pion_mass()
    fpi = ed.fpi()/np.sqrt(2)

    data = ((mpi**2) / (16. * (np.pi**2)*(fpi**2)))

    label = "$\\xi$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    phys_xi = phys_pion**2 / (8. * (np.pi**2)*(phys_Fpi**2))

    return (data.mean(), data.std(),
            label, {"PDG": phys_xi})

def chiral_x(ed, options):
    B = 2777.99374105
    B =  2817.1
    F = 85.8
    # logging.warn("PLEASE SET B and F0 for these plots")

    qmass = ed.ep.scale*(ed.ep.ud_mass + ed.ep.residual_mass)/ed.ep.Zs
    Msqr = B * (qmass + qmass)
    data = Msqr / (16. * (np.pi**2) * (F**2))

    label = "$x=B(m_q + m_q)/(4 \pi F)^2$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data, 0,
            label, {"using flag mud": B*(flag_mq*2) / (16*(np.pi**2) * (F**2) )})
