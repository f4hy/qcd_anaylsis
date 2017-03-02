from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon


def fD(ed, options):
    data = ed.fD()

    label = "$f_D$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_FD, "Bottom": phys_FB})

def fDA(ed, options):
    data = ed.fDA()

    label = "$f_D^A$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_FD, "Bottom": phys_FB})


def fDAr(ed, options):
    data = ed.fD() / ed.fDA()

    label = "$f_D^A$"
    return (data.mean(), data.std(),
            label, {})

def fDAr_Z(ed, options):
    data = ed.fD() / (ed.fDA()/ed.ep.Zv)

    label = "$f_D / f_D^A$"
    return (data.mean(), data.std(),
            label, {})

def fhl(ed, options):
    data = ed.fhl()

    label = "$f_{hl}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    phys = {"Charm": phys_FD, "Bottom": phys_FB}
    pds = {m: (d.mean(), d.std(), label, phys) for m, d in data.iteritems()}
    return pds


def fm1(ed, options):
    data = ed.fD(heavy="m1")

    label = "$f_{hl}^{m1}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )

def fm2(ed, options):
    data = ed.fD(heavy="m2")

    label = "$f_{hl}^{m2}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fm3(ed, options):
    data = ed.fD(heavy="m3")

    label = "$f_{hl}^{m3}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fm4(ed, options):
    data = ed.fD(heavy="m4")

    label = "$f_{hl}^{m4}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fm5(ed, options):
    data = ed.fD(heavy="m5")

    label = "$f_{hl}^{m5}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )


def fDs(ed, options):
    data = ed.fDs()

    label = "$f_{Ds}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_FD, "Bottom": phys_FB})

def fDsA(ed, options):
    data = ed.fDsA()

    label = "$f_{Ds}^A$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, {"Charm": phys_FD, "Bottom": phys_FB})


def fDs_m1(ed, options):
    data = ed.fDs(heavy="m1")

    label = "$f_{hs}^{m1}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )

def fDs_m2(ed, options):
    data = ed.fDs(heavy="m2")

    label = "$f_{hs}^{m2}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fDs_m3(ed, options):
    data = ed.fDs(heavy="m3")

    label = "$f_{hs}^{m3}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fDs_m4(ed, options):
    data = ed.fDs(heavy="m4")

    label = "$f_{hs}^{m4}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
def fDs_m5(ed, options):
    data = ed.fDs(heavy="m5")

    label = "$f_{hs}^{m5}$"
    if ed.scale != 1.0:
        label += " [MeV]"
    return (data.mean(), data.std(),
                     label, )
