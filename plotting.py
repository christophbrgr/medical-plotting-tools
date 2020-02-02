import os
import numpy as np
import SimpleITK as sitk

from matplotlib import colors
from scipy import ndimage

def binaryCMap(one, zero=(0,0,0,0)):
    """
    returns a new binary colormap where one and
    zero are the color values for max and min in the
    image, respectively. Default for zero is
    transparent.
    args:
        one, zero: 4-tuple, float 0...1
    returns:
        listedColormap
    """
    return colors.ListedColormap([zero, one])

def ret_CoM(segmentation, fallback=True):
    """
    returns the center of mass for the passed tumor labels
    (BraTS Standard)
    """
    mask = np.zeros(segmentation.shape)
    mask[segmentation == 1] = 1
    mask[segmentation == 4] = 1
    if np.sum(mask) == 0 and fallback: # if no tumor core is found, use the edema CoM
        mask[segmentation > 0] = 1
    com = ndimage.measurements.center_of_mass(mask) # get center of mass for tumor core
    return (int(com[0]), int(com[1]), int(com[2])) # convert to int (cuts decimals!)

def pad3D(image, newshape):
    """
    Adds padding to the passed 3D numpy array
    and places the input image in the center of
    the newly created image with shape newshape

    args:
        image(np.ndarray): arbitrary numpy array with 3 dims
        newshape(3-tuple, int): target shape for padding
                                has to be smaller than the
                                shape of the image
    returns:
        numpy array with shape=newshape
    """
    for (i,j) in zip(image.shape, newshape):
        if j < i:
            raise IOError('Target shape cannot be smaller than input shape!')
    res = np.zeros(newshape)
    offset_z = int((newshape[2]-image.shape[2])/2)
    offset_y = int((newshape[1]-image.shape[1])/2)
    offset_x = int((newshape[0]-image.shape[0])/2)
    bound_z = offset_z + image.shape[2]
    bound_y = offset_y + image.shape[1]
    bound_x = offset_x + image.shape[0]
    res[offset_x:bound_x,offset_y:bound_y,offset_z:bound_z] = image
    return res

def pad2D(image, newshape):
    """
    Adds padding to the passed 3D numpy array
    and places the input image in the center of
    the newly created image with shape newshape

    args:
        image(np.ndarray): arbitrary numpy array with 3 dims
        newshape(3-tuple, int): target shape for padding
                                has to be smaller than the
                                shape of the image
    returns:
        numpy array with shape=newshape
    """
    for (i,j) in zip(image.shape, newshape):
        if j < i:
            raise IOError('Target shape cannot be smaller than input shape!')
    res = np.zeros(newshape)
    offset_y = int((newshape[1]-image.shape[1])/2)
    offset_x = int((newshape[0]-image.shape[0])/2)
    bound_y = offset_y + image.shape[1]
    bound_x = offset_x + image.shape[0]
    res[offset_x:bound_x,offset_y:bound_y] = image
    return res
