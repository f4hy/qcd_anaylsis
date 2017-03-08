from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon
import numpy as np

def mpi(ed, options):
    data = ed.pion_mass()

    label = "$m_\pi$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), data.std(),
            label, {"PDG": phys_pion})


def mK(ed, options):
    data = ed.kaon_mass()

    label = "$m_K$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
            label, {"PDG": phys_kaon})


def twomksqr_mpisqr(ed, options):
    kdata = ed.kaon_mass()
    pdata = ed.pion_mass()
    data = 2.0 * (kdata**2) - (pdata**2)
    label = "$2m_K^2 - m_\pi^2 $"
    if ed.ep.scale != 1.0:
        label += " [MeV^2]"
    return (data.mean(), data.std(),
            label, {"PDG": 2.0 * phys_kaon**2 - phys_pion**2})


def mpisqr(ed, options):
    pdata = ed.pion_mass()
    data = (pdata**2)
    label = "$m_\pi^2 $"
    if ed.ep.scale != 1.0:
        label += " [MeV^2]"
    return (data.mean(), data.std(),
            label, {"PDG": phys_pion**2})


def mpisqr_mq(ed, options):
    mpi = ed.pion_mass()
    mq = ed.ep.scale*(ed.ep.ud_mass + ed.ep.residual_mass)/ed.ep.Zs

    data = mpi**2 / mq

    mpsqrerr = (mpi**2).std()
    res_err = ed.ep.scale*ed.ep.residual_mass_error
    err = np.sqrt(( mpsqrerr / mq)**2 + (res_err * data.mean() / mq)**2 )

    label = "$m_\pi^2 / m_q$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"

    return (data.mean(), err,
            label, {"PDG": phys_pion**2 / phys_mq})
