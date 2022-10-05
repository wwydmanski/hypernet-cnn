import matplotlib.pyplot as plt
import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter # add shuffle for supervised
from rpy2.robjects.conversion import localconverter

r_lpca_package = importr('logisticPCA')

# Create a converter that starts with rpy2's default converter
# to which the numpy conversion rules are added.
np_cv_rules = default_converter + numpy2ri.converter

def select_m(data, ks, ms=list(range(1, 10)), plot=True):
    np_cv_rules = default_converter + numpy2ri.converter
    with localconverter(np_cv_rules) as cv:
      
        logpca_cv = r_lpca_package.cv_lpca(data, ks = ks, ms = ms)

        if plot:
            plt.plot(ms, logpca_cv[0])

    return ms[np.argmin(logpca_cv[0])]



def get_LPCs(data, k, m):
    with localconverter(np_cv_rules) as cv:

        lpca_model = r_lpca_package.logisticPCA(data, k=k, m=m)

        return lpca_model.rx2['PCs']