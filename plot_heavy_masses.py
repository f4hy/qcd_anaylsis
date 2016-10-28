from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


def mD(ed, options):
    data = ed.D_mass()
    label = "$m_{hl}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_D, "Bottom": phys_MB})
