from scipy import stats
import numpy as np

def error(x):
    #return x.std()
    med = np.median(x)
    percentile84 = stats.scoreatpercentile(x, 84.13)
    percentile16 = stats.scoreatpercentile(x, 15.87)
    return [[med-percentile16], [percentile84-med] ]
