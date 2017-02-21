import logging
import numpy as np
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


def mpisqr_mq(ed, options):
    mpi = ed.pion_mass()
    mq = ed.scale*(ed.ep.ud_mass + ed.ep.residual_mass)/ed.ep.Zs

    data = mpi**2 / mq

    mpsqrerr = (mpi**2).std()
    res_err = ed.scale*ed.ep.residual_mass_error
    err = np.sqrt(( mpsqrerr / mq)**2 + (res_err * data.mean() / mq)**2 )

    label = "$m_\pi^2 / m_q$"
    if ed.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), err,
            label, {"PDG": phys_pion**2 / phys_mq})



def fpi(ed, options):
    data = ed.fpi()

    label = "$f_\pi$"
    if ed.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_Fpi})

def fK(ed, options):
    data = ed.fK()

    label = "$f_K$"
    if ed.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_FK})



def xi(ed, options):
    mpi = ed.pion_mass()
    fpi = ed.fpi()

    data = ((mpi**2) / (8 * (np.pi**2)*(fpi**2)))

    label = "$\\xi$"
    if ed.scale != 1.0:
        label += " [MeV]"

    phys_xi = phys_pion**2 / (8 * (np.pi**2)*(phys_Fpi**2))

    return (data.mean(), data.std(),
            label, {"PDG": phys_xi})

def chiral_x(ed, options):
    B = 2777.99374105
    B = 2817.51286699
    F_0 = 117.492871007
    # logging.warn("PLEASE SET B and F0 for these plots")

    qmass = ed.scale*(ed.ep.ud_mass + ed.ep.residual_mass)/ed.ep.Zs
    Msqr = B * (qmass + qmass)
    data = Msqr / (8 * (np.pi**2) * (F_0**2))

    label = "$x=B(m_q + m_q)/(4 \pi F)^2$"
    if ed.scale != 1.0:
        label += " [MeV]"

    return (data, 0,
            label, {"": 0})
