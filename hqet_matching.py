
import numpy as np
from alpha_s import *
from mpmath import *

'''
Script to calculate matching coefficients up to
alpha_s**3 as in Bekavac et. al. arxiv:0911.3356
Result is matching to get HQET values from
QCD, i.e. result * QCD = HQET.
Does not contain x=mi/m terms.
'''

def get_pole_mass(alpha_s,msbar_m):
    beta0 = 1./4. * (11. - 2./3.*Nf)
    beta1 = 1./4.**2.0 * (102. - 38./3.*Nf)
    beta2 = 1./4.**3.0 * (2857./2. - 5033./18.*Nf + 325./54.*Nf**2.0)
    beta3 = 1./4.**4.0 * (149753./6. + 3564.*zeta(3) - \
        (1078361./162. + 6508./27.*zeta(3))*Nf + \
        (50065./162. + 6472./81.*zeta(3))*Nf**2.0 + 1093./729.*Nf**3.0 )
    am = alpha_s/np.pi
    Zm1 = 1 + 4./3.*am;
    Zm2 = Zm1 + (6.248*beta0-3.739)*am**2;
    Zm3 = Zm2 + (23.497*beta0**2+6.248*beta1+1.019*beta0-29.94)*am**3.0

    mQ = msbar_m*Zm3

    return mQ

def lo_match(m,scale):
    const = -2./3.
    log_val = -1./2.*np.log(scale**2.0/m**2.0)
    if constants:
        return (const + log_val)
    else:
        return lo_val

def nlo_match(m,scale):
    value = 0
    value+=CG(m,scale)
    if constants:
        value+=CH()*n_h
    value+=CL(m,scale)*n_l
    return value

def nnlo_match(m,scale):
    value = 0
    value+=CGG(m,scale)
    value+=CGH(m,scale)*n_h
    value+=CGL(m,scale)*n_l
    if constants:
        value+=CHH()*n_h**2.0
        value+=CHL()*n_h*n_l
    value+=CLL(m,scale)*n_l**2.0
    return value

def CG(m,scale):
    # constants
    ln2=np.log(2)
    L=np.log(scale**2.0/m**2.0)
    const=-177./64.-5./72.*np.pi**2.0-1./18.*np.pi**2.0*ln2-11./36.*zeta(3)
    log1=(-79./144.-7./108.*np.pi**2.0)*L
    log2=(13./16.)*L**2.0
    log_val = log1+log2
    if constants:
        return const+log_val
    else:
        return log_val

def CH():
    const=727./432.-1./6.*np.pi**2.0
    return const

def CL(m,scale):
    L=np.log(scale**2.0/m**2.0)
    const=47./288.+1./36.*np.pi**2.0
    log1=5./72.*L
    log2=-1./24*L**2.0
    log_val = log1+log2
    if constants:
        return const+log_val
    else:
        return log_val

def CGG(m,scale):
    ln2=log(2)
    L=np.log(scale**2.0/m**2.0)
    const=-62575./62208.-231253./46656.*np.pi**2.0 \
        - 517./324.*np.pi**2.0*ln2+20./81.*np.pi**2.0*ln2**2.0 \
        +5645./1296.*zeta(3)+2089./486.*np.pi**2.0*zeta(3) \
        -17347./58320.*np.pi**4.0-49435./2592.*zeta(5) \
        +11./54.*ln2**4.0+44./9.*a4
    log1=115./54.-121./648.*np.pi**2.0+1./36*np.pi**2.0*ln2 \
          +37./48.*zeta(3)-95./1944.*np.pi**4.0
    log1*=L
    log2=2257./576.+91./432.*np.pi**2.0
    log2*=L**2.0
    log3=-13./8.*L**3.0
    log_val=log1+log2+log3
    if constants:
        return const+log_val
    else:
        return log_val

def CGH(m,scale):
    ln2=log(2)
    L=np.log(scale**2.0/m**2.0)
    const = 2051./96.-24583./2430.*np.pi**2.0+361./27.*np.pi**2.0*ln2 \
            + 10./81.*np.pi**2.0*ln2**2.0-45869./5184.*zeta(3) \
            + 53./96.*np.pi**2.0*zeta(3)-1./20.*np.pi**4.0 \
            - 85./32.*zeta(5)-10./81*ln2**4.0-80./27.*a4
    log1 = -727./864.+1./12.*np.pi**2.0
    log1 *= L
    if constants:
        return const + log1
    else:
        return log1

def CGL(m,scale):
    ln2=log(2)
    L=np.log(scale**2.0/m**2.0)
    const = 24457./46656.+5575./8748.*np.pi**2.0+19./324.*np.pi**2.0*ln2 \
            -1./81.*np.pi**2.0*ln2**2.0+3181./1944.*zeta(3) \
            -379./116640.*np.pi**4.0-1./162.*ln2**4.0-4./27.*a4
    log1 = -319./5184.+11./972.*np.pi**2.0+83./216.*zeta(3)
    log1 *= L
    log2 = -469./864.-7./648.*np.pi**2.0
    log2 *= L**2.0
    log3 = 25./144.*L**3.0
    log_val = log1+log2+log3
    if constants:
        return const + log_val
    else:
        return log_val

def CHH():
    return -5857./7776.+1./405.*np.pi**2.0+11./18.*zeta(3)

def CHL():
    return -193./432.+29./648.*np.pi**2.0

def CLL(m,scale):
    ln2=log(2)
    L=np.log(scale**2.0/m**2.0)
    const = 1751./46656.-13./648.*np.pi**2.0-7./108.*zeta(3)
    log1 = 35./2592.*L
    log2 = 5./432.*L**2.0
    log3 = -1./216.*L**3.0
    log_val = log1+log2+log3
    if constants:
        return const+log_val
    else:
        return log_val


ms_mass=[2.0]
constants = True
alpha_loop = 4 # n for alpha_s**n expansion
scale=2.0 # scale for  matching
Nf=3.00
n_h = 0 # number of heavy quarks
n_l = 3 # number of light quarks
a4=polylog(4,0.5)

def main():
    for m in ms_mass:
        alpha=alpha_s(m)[alpha_loop]
        mass=get_pole_mass(alpha,m)
        alpha=alpha_s(mass)[alpha_loop]
        result = 1.0
        result += lo_match(mass,scale)*(alpha/np.pi)
        result += nlo_match(mass,scale)*(alpha/np.pi)**2.0
        result += nnlo_match(mass,scale)*(alpha/np.pi)**3.0
        print 'For mass '+str(m)+', matching factor C(m) = '+str(result)

if __name__=='__main__':
    main()
