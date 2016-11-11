from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


def mpi(ed, options):
    data = ed.pion_mass()

    label = "$m_\pi$"
    if ed.scale != 1.0:
        label += " [MeV]"
    print data
    return (data.mean(), data.std(),
            label, {"PDG": phys_pion})

def mK(ed, options):
    data = ed.kaon_mass()

    label = "$m_K$"
    if ed.scale != 1.0:
        label += " [MeV]"
    print data
    return (data.mean(), data.std(),
            label, {"PDG": phys_kaon})



def twomksqr_mpisqr(ed, options):
    kdata = ed.kaon_mass()
    pdata = ed.pion_mass()
    data = 2.0*(kdata**2) - (pdata**2)
    label = "$2m_K^2 - m_\pi^2 $"
    if ed.scale != 1.0:
        label += " [MeV^2]"
    return (data.mean(), data.std(),
            label, {"PDG": 2.0*phys_kaon**2 - phys_pion**2})

def mpisqr(ed, options):
    pdata = ed.pion_mass()
    data = (pdata**2)
    label = "$m_\pi^2 $"
    if ed.scale != 1.0:
        label += " [MeV^2]"
    return (data.mean(), data.std(),
                     label, {"PDG": phys_pion**2})