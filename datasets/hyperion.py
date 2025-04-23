import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets as ds

import datasets
import datasets.util
import scipy.io

class HYPERION:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        H,W,HIM = load_data()

        self.H = H
        self.W = W
        self.HIM = self.Data(HIM)

        self.n_dims = self.HIM.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        datasets.util.plot_hist_marginals(data_split.x)
        plt.show()


def load_data():
    file_name = datasets.root  + 'Hyperion_norm.mat'
    mat = scipy.io.loadmat(file_name)
    HIM_name = 'data'


    # HIM is H*W*C, GT is H*W
    HIM = mat[HIM_name]
    # band_selected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    # HIM = HIM[:,:,band_selected]

    
    HIM_np = np.array(HIM)


    H,W,C = HIM_np.shape
    HIM_reshaped = HIM_np.reshape((H*W,C)).astype(np.float32)
    return H,W,HIM_reshaped

