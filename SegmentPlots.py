#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import geopandas as gpd
import numpy as np
from samgeo.hq_sam import SamGeo
from osgeo import gdal, ogr, osr
import rasterio


# In[ ]:


def generateSegmentationMask(imgPth, sam):
    mask_output = os.path.join(os.path.splitext(imgPth)[0] + '-mask.tif')
    annotation_output = os.path.join(os.path.splitext(imgPth)[0] + '-annotation.tif')
#     vector_output = os.path.join(os.path.splitext(imgPth)[0] + '-vector.gpkg')
    vector_output = os.path.join(os.path.splitext(imgPth)[0] + '-vector.shp')
    sam.generate(imgPth, output=mask_output, foreground=True, unique=True)
    sam.show_anns(axis="off", alpha=1, output=annotation_output)
    raster_to_polygon(annotation_output, vector_output)
#     sam.tiff_to_vector(annotation_output, vector_output)
    
def raster_to_polygon(input_raster, output_vector):
    """
    Converts a raster dataset to a polygon vector dataset.

    Parameters:
    - input_raster (str): Path to the input raster dataset.
    - output_vector (str): Path to the output vector dataset.
    - output_crs (str or osr.SpatialReference): CRS information for the output vector dataset.

    Returns:
    None
    """
    output_tif = os.path.join(os.path.splitext(input_raster)[0] + '-summed.tif')
    create_unique_values(input_raster, output_tif)
    raster_ds = None
    mem_ds = None
    output_ds = None 
    try:
        # Open the raster dataset
        raster_ds = gdal.Open(output_tif)
        # Get the spatial reference
        crs = osr.SpatialReference()
        crs.ImportFromWkt(raster_ds.GetProjection())

        # Create a memory vector layer to store the polygons
        mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
        vector_layer = mem_ds.CreateLayer('polygonized', geom_type=ogr.wkbPolygon)
        
        # Use gdal.Polygonize to convert raster to polygons
        gdal.Polygonize(raster_ds.GetRasterBand(1), None, vector_layer, 0, ["DN"], callback=None)

        # Create an output vector file and copy the features
        driver = ogr.GetDriverByName("ESRI Shapefile")
        output_ds = driver.CreateDataSource(output_vector)

        # Set the spatial reference for the output vector dataset
        if crs:
            if isinstance(crs, str):
                srs = ogr.osr.SpatialReference()
                srs.SetFromUserInput(crs)
            elif isinstance(crs, ogr.osr.SpatialReference):
                srs = crs
            else:
                raise ValueError("Invalid output_crs format. Use a CRS string or osr.SpatialReference object.")

            output_layer = output_ds.CreateLayer('polygonized', geom_type=ogr.wkbPolygon, srs=srs)
        else:
            output_layer = output_ds.CreateLayer('polygonized', geom_type=ogr.wkbPolygon)

        # Copy fields from the input layer to the output layer
        for i in range(vector_layer.GetLayerDefn().GetFieldCount()):
            field_defn = vector_layer.GetLayerDefn().GetFieldDefn(i)
            output_layer.CreateField(field_defn)

        # Copy features from the memory layer to the output layer
        for feature in vector_layer:
            output_layer.CreateFeature(feature)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Clean up and close datasets
        if raster_ds:
            raster_ds = None

        if mem_ds:
            mem_ds = None  
            
        if output_ds:
            output_ds = None



def create_unique_values(input_tif, output_tif):
    try:
        # Open the input GeoTIFF file
        with rasterio.open(input_tif) as src:
            # Read all bands
            bands = src.read()

            # Combine RGB values into a single band
            unique_values = bands[0] * 65536 + bands[1] * 256 + bands[2]

            # Update metadata for the output dataset
            profile = src.profile
            profile.update(count=1, dtype=np.uint32)

            # Create the output GeoTIFF file
            with rasterio.open(output_tif, 'w', **profile) as dst:
                # Write the unique values band to the output file
                dst.write(unique_values, 1)

        print(f"Unique values saved to: {output_tif}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
def selectPolygons(inFilePath):
    output_path = os.path.join(os.path.splitext(inFilePath)[0] + '-selected.shp')
    gdf = gpd.read_file(inFilePath)
    gdf['area'] = gdf.area
    median_area = np.median(gdf['area'])
#     iqr = np.percentile(gdf['area'], plot_selection_percentile[1]) - np.percentile(gdf['area'], plot_selection_percentile[0])
#     lower_thresh = np.percentile(gdf['area'], plot_selection_percentile[0])
#     upper_thresh = median_area + 3*iqr
    iqr = median_area*0.25
    selected_polygons = gdf[(gdf['area'] >= median_area-iqr) & (gdf['area'] <= median_area+3.5*iqr)]
    selected_polygons.to_file(output_path, driver='ESRI Shapefile')
    return selected_polygons

def generateCentroids(selected_polygons, inFilePath):
    output_path = os.path.join(os.path.splitext(inFilePath)[0] + '-Centroids.shp')
    selected_polygons['centroid'] = selected_polygons['geometry'].centroid
    centroids_gdf = gpd.GeoDataFrame(geometry=selected_polygons['centroid'])
    centroids_gdf.to_file(output_path, driver='ESRI Shapefile')
    return centroids_gdf

