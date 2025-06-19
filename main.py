##########################
# DO NOT CHANGE THIS FILE!
##########################

import numpy as np
import matplotlib.pyplot as plt

# import plot function
from plotting import plot_images

# import scaling functions
from nearest_neighbor import nearest_neighbor_scaling
from bilinear import bilinear_interpolation_scaling


if __name__ == '__main__':
    # define scaling factors
    downscaling_factor = 2/3
    upscaling_factor = 3

    # read the test image and convert to numpy array
    original = plt.imread("pixel.bmp")
    original_npa = np.array(original)

    # execute the scaling methods on the test image
    # for upscaling
    upscaled_nn = nearest_neighbor_scaling(upscaling_factor, original_npa)
    upscaled_bl = bilinear_interpolation_scaling(upscaling_factor, original_npa)

    # for downscaling
    downscaled_nn = nearest_neighbor_scaling(downscaling_factor, original_npa)
    downscaled_bl = bilinear_interpolation_scaling(downscaling_factor, original_npa)

    # plot the results
    plot_images(original, upscaled_nn, upscaled_bl, upscaling_factor, downscaled_nn, downscaled_bl, downscaling_factor)

