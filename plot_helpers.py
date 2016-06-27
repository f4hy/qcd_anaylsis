#!/usr/bin/env python2
from scipy import stats
import numpy as np
import math

from ensamble_info import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from ensamble_info import phys_eta, phys_etac, phys_FK, phys_mhq
from ensamble_info import Zs, Zv


def error(x):
    #return x.std()
    med = np.median(x)
    percentile84 = stats.scoreatpercentile(x, 84.13)
    percentile16 = stats.scoreatpercentile(x, 15.87)
    return [[med-percentile16], [percentile84-med] ]

def print_paren_error(value,error):

    try:
        digits = math.floor(math.log10(error))
    except ValueError:
        return value
    if digits >= 1.0:
        formated_string = "{m:d}({e:d})".format(m=int(value), e=int(error))
        return formated_string
    if digits == 0.0:
        formated_string = "{m:0.1f}({e:.1f})".format(m=value, e=error)
        return formated_string

    digits= -1.0*digits
    formated_error = int(round(error * (10**(digits + 1))))
    formated_value = "{m:.{d}f}".format(d=int(digits) + 1, m=value)

    formated_string = "{m}({e})".format(m=formated_value, e=formated_error)
    return formated_string

def add_mc_lines(axe, options, auto_key):
    mcline = None
    if options.ydata == "mD":
        mcline = phys_D
    if options.ydata == "mDs":
        mcline = phys_Ds
    if mcline is not None:
        mq = 1068
        for i in range(6):
            axe.plot([0, mq*(1.25)**i], [mcline*(1.15)**i,mcline*(1.15)**i], color=auto_key((i, None, None), check=False)[0], ls="--", lw=2)
            axe.annotate("$1.15^{}".format(i) + "M_{D_s}$", xy=(100, 50+mcline*(1.15)**i), fontsize=30)

            #axe.axvspan(mq*(1.25)**i, mq*(1.25)**i, ymin=0, ymax=mcline*(1.15)**i, color=auto_key((i, None, None), check=False)[0], ls="--", lw=2)
            axe.plot([mq*(1.25)**i, mq*(1.25)**i], [0, mcline*(1.15)**i], color=auto_key((i, None, None), check=False)[0], ls="--", lw=2)
            axe.annotate("$1.25^{}".format(i) + "M_{q_h}$", xy=(-150+mq*(1.25)**i, 1500), fontsize=30, rotation=90)



def test():

    print(print_paren_error(0.11111, 0.00111))
    print(print_paren_error(0.00815, 0.0042))
    print(print_paren_error(0.00815, 0.0041))
    print(print_paren_error(0.00815, 46.0))

    print(print_paren_error(353.74, 4.0))
    print(print_paren_error(353.74, 43.0))
    print(print_paren_error(353.74, 40.3))
    print(print_paren_error(353.74, 354.0))
    print(print_paren_error(353.74, 354.6))

    print(print_paren_error(0.000815, 0.1))

    print(print_paren_error(32.0, 4.3))
    print(print_paren_error(-32.0, 4.3))


if __name__ == "__main__":
    test()
