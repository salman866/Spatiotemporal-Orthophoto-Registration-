# Spatiotemporal Orthophoto Registration in UAV-based Crop Breeding Experiments

This repository presents a cost-effective method for temporally aligning UAV-derived orthophotos in plant breeding trials without relying on GCPs or high-precision GNSS. Our centroid-based registration approach, validated across multiple crops and field layouts, achieves plot-level accuracy suitable for agronomic analysis, offering a practical solution for resource-limited environments.

## SAM Model Weights

To use the SAM model, download the pre-trained weights from the following link:  
🔗 [SAM Model Weights](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view)

## Notebooks Overview

- **`Correspondence matching.ipynb`**  
  This notebook is used to generate correspondence matches between image pairs. **It must be run first**, as it produces the matched pairs required by the registration notebooks.

- **`Register Centroids.ipynb`**  
  This notebook registers the centroid shapefiles based on the previously generated correspondence matches.

- **`Register Raster Files.ipynb`**  
  This notebook registers the raster orthophotos using the matched correspondences for alignment with reference images.

Make sure to follow the sequence starting with `Correspondence matching.ipynb` before using the registration notebooks.
