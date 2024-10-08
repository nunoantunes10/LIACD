import pydicom
import numpy as np
# Leer archivo DICOM
from pathlib import Path

# Definir el directorio donde están los archivos
directorio = Path("D:/imag/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001")

# Iterar sobre todos los archivos en la carpeta
for archivo in directorio.rglob('*'):
    if archivo.is_file():  # Ignorar directorios
        dicom_file = pydicom.dcmread(archivo.resolve())

        # Acceder a los metadatos del archivo DICOM
        print(dicom_file)

        image_array = dicom_file.pixel_array

        print(image_array)

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


        # Normalizar los valores a un rango entre 0 y 1
        def normalize_hu(hu_image, min_hu=-1000, max_hu=400):
            hu_image = (hu_image - min_hu) / (max_hu - min_hu)
            hu_image[hu_image > 1] = 1  # Limitar los valores máximos a 1
            hu_image[hu_image < 0] = 0  # Limitar los valores mínimos a 0
            return hu_image

        # Aplicar normalización
        normalized_image = normalize_hu(clipped_image)

        import matplotlib.pyplot as plt

        # Mostrar la imagen normalizada
        plt.imshow(normalized_image, cmap='gray')
        plt.title("Imagen CT Normalizada")
        plt.show()
