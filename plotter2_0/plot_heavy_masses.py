from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


def mD(ed, options):
    data = ed.D_mass()
    label = "$m_{D}$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_D, "Bottom": phys_MB})


def inv_mhl(ed, options):
    data = ed.get_mass("heavy-ud")
    label = "$1/m_{hl}$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
    phys = {"Charm": phys_D, "Bottom": phys_MB}
    return {m.split("_")[-1]: ((1/d).mean(), (1/d).std(), label, phys ) for m,d in data.iteritems()}


def inv_mD(ed, options):
    data = 1.0 / ed.D_mass()
    label = "$1/m_{D}$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": 1/phys_D, "Bottom": 1/phys_MB})


def mDs(ed, options):
    data = ed.Ds_mass()
    label = "$m_{D_s}$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_Ds, "Bottom": phys_MBs})


def mDs_mD(ed, options):
    data = ed.Ds_mass()-ed.D_mass()
    label = "$m_{D_s}-m_{D}$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"PDG": phys_Ds-phys_D})


def inv_mhs(ed, options):
    data = ed.get_mass("heavy-s")
    label = "$1/m_{hs}$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
    phys = {"Charm": phys_Ds, "Bottom": phys_MBs}
    return {m.split("_")[-1]: ((1/d).mean(), (1/d).std(), label, phys ) for m,d in data.iteritems()}


def inv_mDs(ed, options):
    data = 1.0 / ed.Ds_mass()
    label = "$m_{D_s}$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": 1/phys_Ds, "Bottom": 1/phys_MBs})


def inv_mD_corrected(ed, options):
    mdata = ed.D_mass()
    m1 = ed.heavy_m1
    m2 = ed.heavy_m2
    data = 1.0/(mdata +(m2 - m1))
    label = "$1/(m_{hl} + m_2 - m_1)$"
    if options.scale:
        label += " [1/MeV]"
    return (data.mean(), data.std(), label, {"Charm": 1.0/phys_D, "Bottom": 1.0/phys_MB})

def inv_mhl_corrected(ed, options):
    mdata = ed.get_mass("heavy-ud")
    data = {}
    for m, m12s in ed.ep.m12s.iteritems():
        mkey = [k for k in mdata if m in k][0]
        m1,m2 = m12s
        data[m] = (mdata[mkey] + (m2 - m1)*ed.ep.scale )
    label = "$1/(m_{hl} + m_2 - m_1)$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
        phys = {"Charm": phys_D, "Bottom": phys_MB}
    return {m.split("_")[-1]: ((1/d).mean(), (1/d).std(), label, phys ) for m,d in data.iteritems()}


def inv_mDs_corrected(ed, options):
    mdata = ed.Ds_mass()
    m1 = ed.heavy_m1
    m2 = ed.heavy_m2
    data = 1.0/(mdata +(m2 - m1))
    label = "$1/(m_{hs} + m_2 - m_1)$"
    if options.scale:
        label += " [1/MeV]"
    return (data.mean(), data.std(), label, {"Charm": 1.0/phys_Ds, "Bottom": 1.0/phys_MBs})

def inv_mhs_corrected(ed, options):
    mdata = ed.get_mass("heavy-s")
    data = {}
    for m, m12s in ed.ep.m12s.iteritems():
        mkey = [k for k in mdata if m in k][0]
        m1,m2 = m12s
        data[m] = (mdata[mkey] + (m2 - m1)*ed.ep.scale )
    label = "$1/(m_{hs} + m_2 - m_1)$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
        phys = {"Charm": phys_Ds, "Bottom": phys_MBs}
    return {m.split("_")[-1]: ((1/d).mean(), (1/d).std(), label, phys ) for m,d in data.iteritems()}
