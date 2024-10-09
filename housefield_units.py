import pydicom
import numpy as np
# Read DICOM file
from pathlib import Path

# Define the directory of the LIDC-IRDI files. For now we will work with just one pacient
directory = Path(".LIDC-IDRI/LIDC-IDRI-0001")

# Iterate over the files of this directory 
for archive in directory.rglob('*'):
    if archive.is_file():  # Ignorar directorys
        dicom_file = pydicom.dcmread(archive.resolve())
        image_array = dicom_file.pixel_array

        # Extraer los valores de rescaling (si están presentes en los metadatos)
        intercept = dicom_file.RescaleIntercept if 'RescaleIntercept' in dicom_file else 0
        slope = dicom_file.RescaleSlope if 'RescaleSlope' in dicom_file else 1

        # Aplicar la conversión a unidades Hounsfield
        hu_image = image_array * slope + intercept

        # Limitar los valores HU a un rango común
        def clip_hu_range(hu_image, min_hu=-1000, max_hu=400):
            hu_image = np.clip(hu_image, min_hu, max_hu)
            return hu_image

        # Aplicar el recorte de HU
        clipped_image = clip_hu_range(hu_image)


        # Normalize values between 0 y 1
        def normalize_hu(hu_image, min_hu=-1000, max_hu=400):
            hu_image = (hu_image - min_hu) / (max_hu - min_hu)
            hu_image[hu_image > 1] = 1  # Limit maximum values to 1
            hu_image[hu_image < 0] = 0  # Limit minimum values to 0
            return hu_image

        # Aplicar normalización
        normalized_image = normalize_hu(clipped_image)

        # import matplotlib.pyplot as plt

        # # Mostrar la imagen normalizada
        # plt.imshow(normalized_image, cmap='gray')
        # plt.title("Imagen CT Normalizada")
        # plt.show()
