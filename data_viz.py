import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_all_patients_annotations(plot = False):
    # Initialize lists to collect annotation properties across all patients
    sphericities = []
    volumes = []
    surface_areas = []
    textures = []
    malignancies = []
    calcifications = []
    internal_structures = []
    margins = []
    spiculations = []
    subtleties = []
    diameters = []
    lobulations = []

    # Query all scans in the dataset
    scans = pl.query(pl.Scan)

    # Iterate over all scans in the dataset
    for scan in scans:
        # Iterate over all nodules in the scan
        for nodule in scan.cluster_annotations():
            for annotation in nodule:
                # Collect properties of each annotation
                sphericities.append(annotation.sphericity)
                volumes.append(annotation.volume)
                surface_areas.append(annotation.surface_area)
                textures.append(annotation.texture)
                malignancies.append(annotation.malignancy)
                calcifications.append(annotation.calcification)  
                internal_structures.append(annotation.internalStructure) 
                margins.append(annotation.margin)  
                spiculations.append(annotation.spiculation)  
                subtleties.append(annotation.subtlety) 
                diameters.append(annotation.diameter) 
                lobulations.append(annotation.lobulation)  
    if plot:

        # Create subplots for the histograms
        fig, axs = plt.subplots(4, 2, figsize=(12, 16))

        sphericity_bins = np.linspace(0, 1, 11)  # Create 10 bins from 0 to 1 for sphericity
        sphericity_counts, _ = np.histogram(sphericities, bins=sphericity_bins)

        axs[0, 0].bar(np.unique(sphericities), [sphericities.count(x) for x in np.unique(sphericities)], color='skyblue')
        axs[0, 0].set_title('Sphericity')
        axs[0, 0].set_xlabel('Sphericity Value')
        axs[0, 0].set_ylabel('Frequency')

        # Bar plot for volume
        volume_bins = np.linspace(0, max(volumes), 20)  # Create 20 bins for volume
        volume_counts, _ = np.histogram(volumes, bins=volume_bins)

        axs[0, 1].bar(volume_bins[:-1], volume_counts, width=np.diff(volume_bins), color='lightgreen', align='edge')
        axs[0, 1].set_title('Volume')
        axs[0, 1].set_xlabel('Volume (cubic mm)')
        axs[0, 1].set_ylabel('Frequency')

        # Bar plot for surface area
        surface_area_bins = np.linspace(0, max(surface_areas), 20)  # Create 20 bins for surface area
        surface_area_counts, _ = np.histogram(surface_areas, bins=surface_area_bins)

        axs[1, 0].bar(surface_area_bins[:-1], surface_area_counts, width=np.diff(surface_area_bins), color='lightcoral', align='edge')
        axs[1, 0].set_title('Surface Area')
        axs[1, 0].set_xlabel('Surface Area (square mm)')
        axs[1, 0].set_ylabel('Frequency')

        # Bar plot for diameter
        diameter_bins = np.linspace(0, max(diameters), 20)  # Create 20 bins for diameter
        diameter_counts, _ = np.histogram(diameters, bins=diameter_bins)

        axs[1, 1].bar(diameter_bins[:-1], diameter_counts, width=np.diff(diameter_bins), color='lightblue', align='edge')
        axs[1, 1].set_title('Diameter')
        axs[1, 1].set_xlabel('Diameter (mm)')
        axs[1, 1].set_ylabel('Frequency')

        # Plot bar plots for categorical properties
        axs[2, 0].bar(np.unique(textures), [textures.count(x) for x in np.unique(textures)], color='lightpink')
        axs[2, 0].set_title('Texture (1-5)')
        axs[2, 0].set_xlabel('Texture Rating')
        axs[2, 0].set_ylabel('Frequency')

        axs[2, 1].bar(np.unique(malignancies), [malignancies.count(x) for x in np.unique(malignancies)], color='lightgray')
        axs[2, 1].set_title('Malignancy (1-5)')
        axs[2, 1].set_xlabel('Malignancy Rating')
        axs[2, 1].set_ylabel('Frequency')

        # Categorical properties
        axs[3, 0].bar(np.unique(calcifications), [calcifications.count(x) for x in np.unique(calcifications)], color='orange')
        axs[3, 0].set_title('Calcification')
        axs[3, 0].set_xlabel('Calcification Type')
        axs[3, 0].set_ylabel('Frequency')

        axs[3, 1].bar(np.unique(internal_structures), [internal_structures.count(x) for x in np.unique(internal_structures)], color='purple')
        axs[3, 1].set_title('Internal Structure')
        axs[3, 1].set_xlabel('Structure Type')
        axs[3, 1].set_ylabel('Frequency')

        # Adjust layout and hide unused subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle("Annotation Properties Plots for All Patients", fontsize=16)
        plt.show()
    
    return sphericities,volumes,surface_areas,textures,malignancies,calcifications,internal_structures,margins,spiculations,subtleties,diameters,lobulations