#!/usr/bin/env python2
import logging
import argparse
import numpy as np

Nf = 3.0                      # in GeV
Lambda = 0.340              # in GeV

pi = 3.1415927
zeta3 = 1.2020569032
zeta5 = 1.0369277551

mu = 2.0; # in GeV

def beta(mu):

    beta0 = 1.0/(4.0) * (11.0 - 2.0/3*Nf)
    beta1 = 1.0/(4.0)**2 * (102.0 - 38.0/3*Nf)
    beta2 = 1.0/(4.0)**3 * (2857.0/2.0 - 5033.0/18.0*Nf + 325.0/54.0*Nf**2)
    beta3 = 1.0/(4.0)**4 * ( 149753.0/6.0 + 3564.0*zeta3 -
                          (1078361.0/162 + 6508.0/27.0*zeta3)*Nf +
                          (50065.0/162 + 6472.0/81.0*zeta3)*Nf**2 + 1093.0/729.0*Nf**3.0 )
    logging.debug("# b0, b1, b2, b3 = {} {} {} {} ".format(beta0, beta1, beta2, beta3))
    return {0: beta0, 1: beta1, 2: beta2, 3: beta3}


def alpha_s(beta, m):
    beta0, beta1, beta2, beta3 = (beta[i] for i in range(0,4))
    log = np.log

    L = log(m**2.0/Lambda**2.0);
    alpha1 = pi/(beta0*L);
    alpha2 = pi/(beta0*L) * (-beta1/beta0**2.0 * log(L)/L)
    alpha3 = pi/(beta0*L) * 1.0/(beta0**2.0*L**2.0)*(
        beta1**2.0/beta0**2.0 * ((log(L))**2 - log(L) - 1.0) + beta2/beta0)
    alpha4 = pi/(beta0*L) * 1.0/(beta0**3.0*L**3.0)*(
        beta1**3.0/beta0**3.0 * (-(log(L))**3 + 5.0/2.0*(log(L))**2.0 + 2.0*log(L)
                                 - 1.0/2.0)
        -3.0*beta1*beta2/beta0**2.0 * log(L) + beta3/(2.0*beta0) )
    asum = alpha1 + alpha2 + alpha3 + alpha4
    logging.debug("alpha_s (1, 2, 3, 4 loop at mMS) = {} {} {} {} = {}".format(alpha1, alpha2, alpha3, alpha4, asum))
    return {1: alpha1, 2: alpha2, 3: alpha3, 4: alpha4, "s": asum}


def pole_mass(alpha, beta, mbar):
    beta0, beta1, beta2, beta3 = (beta[i] for i in range(0,4))
    alpha1, alpha2, alpha3, alpha4 = (alpha[i] for i in range(1,5))
    alpha = alpha["s"]
    # pole mass
    As = alpha / pi
    Cm1 = 4.0 / 3.0
    Zm1 = Cm1 * As

    Cm2 = (6.248*beta0-3.739)
    Zm2 = Cm2*As**2

    Cm3 = (23.497*beta0**2+6.248*beta1+1.019*beta0-29.94)
    Zm3 = Cm3*As**3

    Zm = 1 + Zm1 + Zm2 + Zm3
    logging.debug("Zm to pole mass (1,2,3 loop) 1 + {} {} {} = {}".format(Zm1, Zm2, Zm3, Zm))

    M = Zm*mbar
    logging.debug("pole mass = {}".format(M))
    return M


def matching(alpha, beta, M, mbar):
    beta0, beta1, beta2, beta3 = (beta[i] for i in range(0,4))
    alpha1, alpha2, alpha3, alpha4 = (alpha[i] for i in range(1,5))
    alpha = alpha["s"]

    log = np.log

    # HQET -> QCD matching
    As = alpha/pi;

    Cm1 = 4.0 / 3.0
    Cm2 = (6.248*beta0-3.739)
    Cm3 = (23.497*beta0**2.0+6.248*beta1+1.019*beta0-29.94)


    L = log(mu**2.0/M**2.0);
    Lbar = log(mu**2.0/mbar**2.0);

    ln2 = log(2.0);
    a4 = 0.5174790617; # need to input a number Li4(1/2)

    nl = 3.0; # number of light flavors
    nm = 0.0; # number of charm flavors
    nh = 0.0; # number of heavy flavors
    # 1-loop
    C1 = - 2.0/3.0 - Lbar/2.0;
    C1Lbar = -1.0/2.0;

    # 2-loop
    CG = - 177.0/64.0 - 5.0/72.0*pi**2.0 - 1.0/18.0*pi**2.0*ln2 - 11.0/36.0*zeta3 + (-79.0/144.0 - 7.0/108*pi**2.0)*Lbar + 13.0/16.0*Lbar**2.0
    CH = 727.0/432.0 - 1.0/6.0*pi**2.0;
    CL = 47.0/288.0 + 1.0/36.0*pi**2.0 + 5.0/72*Lbar - 1.0/24.0*Lbar**2.0;
    CM = 0.0;
    C2Lbar = (-79.0/144.0 - 7.0/108.0*pi**2.0) + 5.0/72.0;
    C2Lbar2 =  13.0/16.0 - 1.0/24.0;

    C2 = CG + CH*nh + CL*nl + CM*nm;
    C2bar = C1Lbar*(-2.0*Cm1);

    # 3-loop
    CGG = (-62575.0/62208.0 - 231253.0/46656.0*pi**2.0 - 517.0/324.0*pi**2.0*ln2
    + 20.0/81.0*pi**2.0*ln2**2.0 + 5645.0/1296.0*zeta3
    + 2089.0/486.0*pi**2.0*zeta3 - 17347.0/58320.0*pi**4.0 - 49435.0/2592.0*zeta5
    + 11.0/54.0*ln2**4.0 + 44.0/9.0*a4
    + (115.0/54.0 - 121.0/648.0*pi**2.0 + 1.0/36.0*pi**2.0*ln2 + 37.0/48.0*zeta3
       - 95.0/1944.0*pi**4.0)*Lbar
    + (2257.0/576.0 + 91.0/432.0*pi**2.0)*Lbar**2.0 - 13.0/8.0*Lbar**3.0)
    CGH = (2051.0/96.0 - 24583.0/2430.0*pi**2.0 + 361.0/27.0*pi**2.0*ln2 + 10.0/81.0*pi**2.0*ln2**2.0
    - 45869.0/5184.0*zeta3 + 53.0/96.0*pi**2.0*zeta3
    - 1.0/20.0*pi**4 - 85.0/32.0*zeta5 - 10.0/81.0*ln2**4.0 - 80.0/27.0*a4
    + (-727.0/864.0 + 1.0/12.0*pi**2)*Lbar)
    CGL = (24457.0/46656.0 + 5575.0/8748.0*pi**2.0 + 19.0/324.0*pi**2.0*ln2 - 1.0/81.0*pi**2.0*ln2**2.0
    + 3181.0/1944.0*zeta3 - 379.0/116640.0*pi**4.0
    - 1.0/162.0*ln2**4.0 - 4.0/27.0*a4
    + (-319.0/5184.0 + 11.0/972.0*pi**2.0 + 83.0/216.0*zeta3)*Lbar
    + (-469.0/864.0 - 7.0/648.0*pi**2.0)*Lbar**2.0 + 25.0/144.0*Lbar**3.0)
    CHH = - 5857.0/7776.0 + 1.0/405.0*pi**2.0 + 11.0/18.0*zeta3;
    CHL = - 193.0/432.0 + 29.0/648.0*pi**2.0;
    CLL = (1751.0/46656.0 - 13.0/648.0*pi**2.0 - 7.0/108.0*zeta3 + 35.0/2592.0*L + 5.0/432.0*L**2.0
    - 1.0/216.0*L**3.0)
    CGM = 0.0
    CHM = 0.0
    CLM = 0.0
    CMM = 0.0

    C3 = (CGG + CGH*nh + CGL*nl + CHH*nh**2.0 + CHL*nh*nl + CLL*nl**2.0
    + CGM*nm + CHM*nh*nm + CLM*nl*nm + CMM*nm**2.0)
    C3bar = (C1Lbar*(-2.0*Cm2+Cm1**2.0)
    + C2Lbar*(-2.0*Cm1) + C2Lbar2*Lbar*(-2.0*Cm1))

    # total
    Cmu1 = As*C1;
    Cmu2 = As**2.0*(C2+C2bar);
    Cmu3 = As**3.0*(C3+C3bar+C1*(-beta0*2.0*Cm1));
    Cmu = 1 + Cmu1 + Cmu2 + Cmu3;
    logging.debug("C(mu)  1 + {} {} {} =  {}".format(Cmu1, Cmu2, Cmu3, Cmu))
    return {1: Cmu1, 2: Cmu2, 3:Cmu3, 's': Cmu}


def get_Cmu(m):
    b = beta(m)
    a = alpha_s(b, m)
    mp = pole_mass(a, b, m)
    c = matching(a, b, mp, m)
    return c["s"]

def get_alpha(m):
    b = beta(m)
    a = alpha_s(b, m)
    return a['s']

def main(options):

    for s in options.scales:
        cmu = get_Cmu(s)
        print "Cmu = {}".format(cmu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute alpha_s at 1-4 loops")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('scales', type=float, nargs='+',
                        help='list of scales to compute at')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(args)
