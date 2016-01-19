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
        return data, err, label, 0

    if data_type == "fDsqrtmD":
        fdata = ed.fD(scaled=options.scale).mean()
        ferr = ed.fD(scaled=options.scale).std()
        mdata = ed.D_mass(scaled=options.scale).mean()
        merr = ed.D_mass(scaled=options.scale).std()

        data = fdata*np.sqrt(mdata)
        err = ferr*np.sqrt(mdata) + merr*fdata*(1.0/2.0)*(mdata**(-0.5))

        label = "$f_D\, \sqrt{m_D}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data, err, label, phys_FD*np.sqrt(phys_D)

    if data_type == "fDssqrtmDs":
        fdata = ed.fDs(scaled=options.scale).mean()
        ferr = ed.fDs(scaled=options.scale).std()
        mdata = ed.Ds_mass(scaled=options.scale).mean()
        merr = ed.Ds_mass(scaled=options.scale).std()

        data = fdata*np.sqrt(mdata)
        err = ferr*np.sqrt(mdata) + merr*fdata*(1.0/2.0)*(mdata**(-0.5))

        label = "$f_{Ds}\, \sqrt{m_{Ds}}$"
        if options.scale:
            label += " [MeV^(3/2)]"
        return data, err, label, phys_FDs*np.sqrt(phys_Ds)


    if data_type == "fD":
        data = ed.fD(scaled=options.scale).mean()
        err = ed.fD(scaled=options.scale).std()
        label = "$f_D$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_FD

    if data_type == "fDs":
        data = ed.fDs(scaled=options.scale).mean()
        err = ed.fDs(scaled=options.scale).std()
        label = "$f_{D_s}$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_FDs

    if data_type == "fDsbyfD":
        data = (ed.fDs(scaled=options.scale)/ed.fD(scaled=options.scale)).mean()
        err = (ed.fDs(scaled=options.scale)/ed.fD(scaled=options.scale)).std()
        label = "$f_{D_s}/f_D$"
        if options.scale:
            label += " "
        return data, err, label, phys_FDs/phys_FD

    if data_type == "fKbyfpi":
        data = (ed.fK(scaled=options.scale)/ed.fpi(scaled=options.scale)).mean()
        err = (ed.fK(scaled=options.scale)/ed.fpi(scaled=options.scale)).std()
        label = "$f_K/f_\pi$"
        if options.scale:
            label += " "
        return data, err, label, phys_FK/phys_Fpi


    if data_type == "fK":
        data = ed.fK(scaled=options.scale).mean()
        err = ed.fK(scaled=options.scale).std()
        label = "$f_K$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_FK


    if data_type == "fpi":
        data = ed.fpi(scaled=options.scale).mean()
        err = ed.fpi(scaled=options.scale).std()
        label = "$f_\pi$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_Fpi

    if data_type == "mpi":
        data = ed.pion_mass(scaled=options.scale).mean()
        err = ed.pion_mass(scaled=options.scale).std()
        label = "$M_\pi$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_pion

    if data_type == "mk":
        data = ed.kaon_mass(scaled=options.scale).mean()
        err = ed.kaon_mass(scaled=options.scale).std()
        label = "$M_K$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_pion

    if data_type == "mHH":
        data = ed.HH_mass(scaled=options.scale).mean()
        err = ed.HH_mass(scaled=options.scale).std()
        label = "$M_{HH}$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, 2980.3

    if data_type == "mHH_spinave":
        vdata = ed.HHv_mass(scaled=options.scale).mean()
        pdata = ed.HH_mass(scaled=options.scale).mean()
        verr = ed.HH_mass(scaled=options.scale).std()
        perr = ed.HH_mass(scaled=options.scale).std()


        data = (pdata + 3.0*vdata)/4.0
        err = (perr + 3.0*verr)/4.0
        label = '$\\bar{M}_{HH}$'
        if options.scale:
            label += " [MeV]"
        return data, err, label,  (3*3096.0 + 2980.3)/4.0

    if data_type == "1/mHH_spinave":
        vdata = ed.HHv_mass(scaled=options.scale).mean()
        pdata = ed.HH_mass(scaled=options.scale).mean()
        verr = ed.HH_mass(scaled=options.scale).std()
        perr = ed.HH_mass(scaled=options.scale).std()


        data = 4.0/(pdata + 3.0*vdata)
        err = 4.0*perr/(pdata + 3.0*vdata) + 12/(pdata + 3.0*vdata)
        label = '$1.0/\\bar{M}_{HH}$'
        if options.scale:
            label += " [MeV]"
        return data, err, label,  (3*3096.0 + 2980.3)/4.0


    if data_type == "mHH_spindiff":
        vdata = ed.HHv_mass(scaled=options.scale).mean()
        pdata = ed.HH_mass(scaled=options.scale).mean()
        verr = ed.HHv_mass(scaled=options.scale).std()
        perr = ed.HH_mass(scaled=options.scale).std()


        data = vdata - pdata

        err = verr + perr
        label = "$M_{HH}^{V} - M_{HH}^{P}$"
        if options.scale:
            label += " [MeV]"
        return data, err, label,  3096.0 - 2980.3



    if data_type == "1/mHH":
        data = 1/ed.HH_mass(scaled=options.scale).mean()
        err = ed.HH_mass(scaled=options.scale).std()
        err = err/(data**2)
        label = "$1.0/M_{HH}$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, 1.0/2980.3


    if data_type == "HL_diff":
        HHdata = ed.HH_mass(scaled=options.scale).mean()
        HLdata = ed.D_mass(scaled=options.scale).mean()
        LLdata = ed.pion_mass(scaled=options.scale).mean()
        HHerr = ed.HH_mass(scaled=options.scale).std()
        HLerr = ed.D_mass(scaled=options.scale).std()
        LLerr = ed.pion_mass(scaled=options.scale).std()

        HLdiff = HLdata - ((HHdata + LLdata) / 2.0)
        HLerr = HLerr - ((HHerr + LLerr) / 2.0)


        label = "$m_{Hl} - (M_{HH} + M_{ll})/2$"
        if options.scale:
            label += " [MeV]"
        return HLdiff, HLerr, label, 0

    if data_type == "Hs_diff":
        HHdata = ed.HH_mass(scaled=options.scale).mean()
        HLdata = ed.Ds_mass(scaled=options.scale).mean()
        LLdata = ed.eta_mass(scaled=options.scale).mean()
        HHerr = ed.HH_mass(scaled=options.scale).std()
        HLerr = ed.Ds_mass(scaled=options.scale).std()
        LLerr = ed.eta_mass(scaled=options.scale).std()

        HLdiff = HLdata - ((HHdata + LLdata) / 2.0)
        HLerr = HLerr - ((HHerr + LLerr) / 2.0)

        label = "$m_{Hs} - (M_{HH} + M_{ss})/2.0$"
        if options.scale:
            label += " [MeV]"
        return HLdiff, HLerr, label, phys_Ds - (phys_etac + phys_eta)/2.0



    if data_type == "1/mD":
        mdata = ed.D_mass(scaled=options.scale).mean()
        merr = ed.D_mass(scaled=options.scale).std()

        data = 1.0/mdata
        err = merr/(mdata)**2

        label = "$1/M_D$"
        if options.scale:
            label += " [1/MeV]"
        return data, err, label, 1.0/phys_D

    if data_type == "1/mDs":
        mdata = ed.Ds_mass(scaled=options.scale).mean()
        merr = ed.Ds_mass(scaled=options.scale).std()

        data = 1.0/mdata
        err = merr/(mdata)**2

        label = "$1/M_Ds$"
        if options.scale:
            label += " [1/MeV]"
        return data, err, label, 1.0/phys_Ds


    if data_type == "mD":
        data = ed.D_mass(scaled=options.scale).mean()
        err = ed.D_mass(scaled=options.scale).std()
        label = "$M_D$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_D

    if data_type == "mDs":
        data = ed.Ds_mass(scaled=options.scale).mean()
        err = ed.Ds_mass(scaled=options.scale).std()
        label = "$M_{D_s}$"
        if options.scale:
            label += " [MeV]"
        return data, err, label, phys_Ds

    if data_type == "mpisqr":
        data = (ed.pion_mass(scaled=options.scale)**2).mean()
        err = (ed.pion_mass(scaled=options.scale)**2).std()
        label = "$M_\pi^2$"
        if options.scale:
            label += " [MeV^2]"
        return data, err, label, phys_pion**2

    if data_type == "mKsqr":
        data = (ed.kaon_mass(scaled=options.scale)**2).mean()
        err = (ed.kaon_mass(scaled=options.scale)**2).std()
        label = "$M_K^2$"
        if options.scale:
            label += " [MeV^2]"
        return data, err, label, phys_kaon**2


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
        label = "$M_\pi^2 / m_q$"
        if options.scale:
            label += " [MeV]"

        return data, err, label, phys_pion**2 / phys_mq

    if data_type == "mud":
        data = ed.dp.ud_mass + residual_mass(ed.dp)
        err = residual_mass_errors(ed.dp)
        label = "$M_{ud}$"
        if options.scale:
            data = scale[ed.dp.beta]*data
            label += " [MeV]"
        return data, err, label, phys_mq

    if data_type == "mheavyq":
        data = ed.dp.heavyq_mass / Zs[ed.dp.beta]
        err = 0.0
        label = "$M_{q_h}$"
        if options.scale:
            data = scale[ed.dp.beta]*data
            label += " [MeV]"
        return data, err, label, phys_mhq


    if data_type == "xi":
        mpi = ed.pion_mass(scaled=options.scale)
        fpi = ed.fpi(scaled=options.scale)
        xi = ((mpi**2) / (8 * (np.pi**2)*(fpi**2))).mean()
        data = xi
        err = ((mpi**2) / (8 * (np.pi**2)*(fpi**2))).std()
        phys_xi = phys_pion**2 / (8 * (np.pi**2)*(phys_Fpi**2))
        return data, err, "$\\xi$", phys_xi

    if data_type == "x":
        B = 2826.1
        F_0 = 118.03
        qmass = scale[ed.dp.beta]*(ed.dp.ud_mass + residual_mass(ed.dp)) / Zs[ed.dp.beta]
        Msqr = B * (qmass + qmass)
        x = Msqr / (8 * (np.pi**2) * (F_0**2))

        data = x
        err = 0
        return data, err, "$x=B(m_q + m_q)/(4 \pi F)^2 $", 0

    raise RuntimeError("{} not supported as a data type yet".format(data_type))
