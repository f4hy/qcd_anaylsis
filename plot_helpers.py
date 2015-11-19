#!/usr/bin/env python2
from scipy import stats
import numpy as np
import math

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
