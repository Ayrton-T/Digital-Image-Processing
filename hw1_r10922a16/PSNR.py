from PIL import Image
import numpy as np
import math

def signal_to_noise(noise,original):
    row,col=original.shape

    MSE = 0
    MSE = np.mean((noise - original)**2)

    ans = 10*math.log10(255.0**2/MSE)

    return ans