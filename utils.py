import argparse
import os
import numpy as np

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

import pydicom
from statistics import  median_high

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def calculate_malignancy(nodule):
    # Calcula la malignidad de un nódulo con las anotaciones hechas por 4 doctores.
    # Devuelve la mediana alta de la malignidad anotada y un label True o False para el cáncer.
    # Si la mediana alta es mayor a 3, devolvemos True (cáncer).
    # Si es menor a 3, devolvemos False (no cáncer).
    # Si es 3, devolvemos 'Ambiguous', para procesamiento semisupervisado futuro.
    list_of_malignancy = []
    for annotation in nodule:
        list_of_malignancy.append(annotation.malignancy)

    malignancy = median_high(list_of_malignancy)
    if malignancy > 3:
        return malignancy, True
    elif malignancy < 3:
        return malignancy, False
    else:
        return malignancy, 'Ambiguous'

# Limit the HU (Hounsfield Unit) values to a common range
def clip_hu_range(hu_image, min_hu=-1000, max_hu=400):
    """
    Clip the Hounsfield Unit (HU) values of a CT image to a specified range.
    
    Args:
        hu_image (numpy array): The input CT image in HU.
        min_hu (int, optional): The minimum HU value to clip to. Default is -1000 (typical value for air).
        max_hu (int, optional): The maximum HU value to clip to. Default is 400 (typical value for bones).
    
    Returns:
        numpy array: The CT image with HU values clipped to the range [min_hu, max_hu].
    """
    # Clip the image values to the specified range
    hu_image = np.clip(hu_image, min_hu, max_hu)
    return hu_image


# Convert DICOM image to Hounsfield Units (HU)
def convert_to_HU(archive):
    """
    Convert a DICOM image to Hounsfield Units (HU).
    
    Args:
        archive (Path or str): Path to the DICOM file.
        
    Returns:
        numpy array: The CT image converted to Hounsfield Units (HU).
    """
    # Read the DICOM file
    dicom_file = pydicom.dcmread(archive.resolve())
    # Extract the pixel data from the DICOM file
    image_array = dicom_file.pixel_array

    # Extract rescaling values from DICOM metadata (if present)
    intercept = dicom_file.RescaleIntercept if 'RescaleIntercept' in dicom_file else 0
    slope = dicom_file.RescaleSlope if 'RescaleSlope' in dicom_file else 1

    # Apply the conversion formula to transform to Hounsfield Units (HU)
    # HU = PixelValue * RescaleSlope + RescaleIntercept
    hu_image = image_array * slope + intercept

    return hu_image


# Normalize HU values between 0 and 1
def normalize_hu(hu_image, min_hu=-1000, max_hu=400):
    """
    Normalize the Hounsfield Unit (HU) values of a CT image to the range [0, 1].
    
    Args:
        hu_image (numpy array): The input CT image in HU.
        min_hu (int, optional): The minimum HU value for normalization. Default is -1000.
        max_hu (int, optional): The maximum HU value for normalization. Default is 400.
        
    Returns:
        numpy array: The CT image with HU values normalized between 0 and 1.
    """
    # Normalize the HU values to the range [0, 1]
    hu_image = (hu_image - min_hu) / (max_hu - min_hu)
    
    # Ensure that all values remain between 0 and 1
    hu_image[hu_image > 1] = 1  # Cap values greater than 1 to 1
    hu_image[hu_image < 0] = 0  # Cap values less than 0 to 0
    
    return hu_image

def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*img

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)