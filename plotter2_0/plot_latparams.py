from physical_values import phys_pion, phys_kaon, phys_mq, flag_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_FB, phys_FBs, phys_FBsbyFB, phys_MB, phys_MBs, unphys_etas
from physical_values import phys_eta, phys_etac, phys_etab, phys_FK, phys_mhq, phys_Jpsi, phys_Upsilon
from physical_values import phys_mb_2gev, phys_mc_2gev


def asqr_fm(ed, options):
    data = ed.ep.latspacing**2

    label = "$a^2$"
    if ed.ep.scale != 1.0:
        label += " [fm^2]"
    return (data, 0, label, {"Continuum": 0})

def asqr(ed, options):
    data = 1/(ed.ep.scale**2)

    label = "$a^2$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV^2]"
    return (data, 0, label, {"Continuum": 0})

def a_gev_sqr(ed, options):
    data = ed.ep.a_gev**2

    label = "$a^2$"
    if ed.ep.scale != 1.0:
        label += " [1/GeV^2]"
    return (data, 0, label, {"Continuum": 0})


def mud(ed, options):
    data = ed.ep.ud_mass + ed.ep.residual_mass
    err = ed.ep.residual_mass_error
    label = "$m_{ud}$"
    data = ed.ep.scale*data
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data, err, label, {"PDG": phys_mq})

def ms(ed, options):
    data = ed.ep.s_mass + ed.ep.residual_mass
    err = ed.ep.residual_mass_error
    label = "$m_{s}$"
    data = ed.ep.scale*data
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data, err, label)

def mud_renorm(ed, options):
    data = (ed.ep.ud_mass + ed.ep.residual_mass)/ed.ep.Zs
    err = (ed.ep.residual_mass_error)/ed.ep.Zs
    label = "$m_{ud}/Z_{s}$"
    data = ed.ep.scale*data
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data, err, label, {"PDG": flag_mq})


def ms_renorm(ed, options):
    data = (ed.ep.s_mass + ed.ep.residual_mass)/ed.ep.Zs
    err = (ed.ep.residual_mass_error)/ed.ep.Zs
    label = "$m_{s}/Z_{s}$"
    data = ed.ep.scale*data
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return (data, err, label)

def mh_renorm(ed, options):
    # (ed.ep.heavyq_mass + ed.ep.residual_mass)/ed.ep.Zs
    err = 0
    label = "$m_{h}/Z_{s}$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return {m: (ed.ep.scale*(q + ed.ep.residual_mass)/ed.ep.Zs , err, label)
            for m,q in ed.ep.heavies.iteritems()}


def mheavyq_bare(ed, options):
    label = "$m_{q_h}^{bare}$"
    if ed.ep.scale != 1.0:
        label += " [MeV]"
    return {m: (ed.ep.scale*(q), err, label)
            for m,q in ed.ep.heavies.iteritems()}


def inv_mheavyq(ed, options):
    err = 0
    label = "$1/\\bar{m}_{q_h}$"
    if ed.ep.scale != 1.0:
        label += " [1/MeV]"
    phys = {"Charm": 1/phys_mc_2gev, "Bottom": 1/phys_mb_2gev}
    return {m: (1/(ed.ep.scale*(q + ed.ep.residual_mass)/ed.ep.Zs) , err, label, phys)
            for m,q in ed.ep.heavies.iteritems()}
