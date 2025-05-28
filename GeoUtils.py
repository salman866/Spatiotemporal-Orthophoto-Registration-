#!/usr/bin/env python
# coding: utf-8

# In[3]:



import fiona
import rasterio
import os
import pandas as pd
import numpy as np
from fiona.crs import from_epsg
from fiona.transform import transform_geom
from pyproj import CRS
from pyproj.enums import WktVersion
from packaging import version
from functools import partial
from shapely.geometry import mapping, shape
from math import radians, cos, sin, asin, sqrt, atan2
from rasterio.enums import Resampling as r_sample
from rasterio.transform import from_origin, from_gcps
from osgeo import gdal, osr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Transformer
from rasterio.transform import Affine


# In[4]:


def getDistance(coordStart, coordEnd):
    R = 6372800 # 3959.87433 is in miles.  For Earth radius in kilometers use 6372.8 km
    dLat = radians(coordEnd[0] - coordStart[0])
    dLon = radians(coordEnd[1] - coordStart[1])
    lat1 = radians(coordStart[0])
    lat2 = radians(coordEnd[0])
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

def getBearing(coordStart, coordEnd):
    # Coordinates in (latitude, Longitude) format
    dLat = radians(coordEnd[0] - coordStart[0])
    dLon = radians(coordEnd[1] - coordStart[1])
    lat1 = radians(coordStart[0])
    lat2 = radians(coordEnd[0])
    x = cos(lat2) * sin(dLon)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    brng = np.arctan2(x,y)
    brng = (np.degrees(brng) + 360)%360
    
    return brng

def getCoordinates(coordStart, distance, bearing):
    R = 6372800 # 3959.87433 is in miles.  For Earth radius in kilometers use 6372.8 km
    brng = radians(bearing)
    d = distance
    lat1 = radians(coordStart[0])
    lon1 = radians(coordStart[1])
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(brng))
    lon2 = lon1 + atan2(sin(brng) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2))
    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    coordEnd = (lat2, lon2)
    return coordEnd

# set up Fiona transformer
def crs_to_fiona(proj_crs):
    proj_crs = CRS.from_user_input(proj_crs)
    if version.parse(fiona.__gdal_version__) < version.parse("3.0.0"):
        fio_crs = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
    else:
        # GDAL 3+ can use WKT2
        fio_crs = proj_crs.to_wkt()
    return fio_crs

def base_transformer(geom, src_crs, dst_crs):
    return shape(
        transform_geom(
            src_crs=crs_to_fiona(src_crs),
            dst_crs=crs_to_fiona(dst_crs),
            geom=mapping(geom),
            antimeridian_cutting=True,
        )
    )

def transformCRS(gdf, desCRS):
    destination_crs = desCRS
    forward_transformer = partial(base_transformer, src_crs=gdf.crs, dst_crs=destination_crs)
    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION="YES"):
        transformed_gdf = gdf.set_geometry(gdf.geometry.apply(forward_transformer), crs=destination_crs)
    return transformed_gdf
# Sort Geo Dataframe based on latitude or longitude
def sortGeoDataframe(gdf, lat=True):
    if lat:
        gdf['y_coordinate'] = gdf['geometry'].apply(lambda geom: geom.y)
        sorted_gdf = gdf.sort_values(by='y_coordinate')
        sorted_gdf = sorted_gdf.drop(columns=['y_coordinate'])
    else:
        gdf['x_coordinate'] = gdf['geometry'].apply(lambda geom: geom.x)
        sorted_gdf = gdf.sort_values(by='x_coordinate')
        sorted_gdf = sorted_gdf.drop(columns=['x_coordinate'])
    return sorted_gdf

def resampledGeoTiff(geotiff_path, scale_factor):
    output_path = os.path.join(os.path.splitext(geotiff_path)[0] + '-resampled.tif')
    with rasterio.open(geotiff_path) as src:
        # Calculate new dimensions
        new_width = src.width // scale_factor
        new_height = src.height // scale_factor

        # Calculate the transform for the new dimensions
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Prepare the output metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'height': new_height,
            'width': new_width,
            'transform': transform
        })
    
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for band in range(1, src.count + 1):
                # Read data and downsample using bilinear interpolation
                data = src.read(band, out_shape=(new_height, new_width), resampling=Resampling.bilinear)

                # Convert to appropriate data type
                data = data.astype(rasterio.uint8)

                # Write to the output file
                dst.write(data, band)
            return output_path
        
def change_CRS_raster(input_file, crs):
    output_path = os.path.join(os.path.splitext(input_file)[0] + '-reprojected.tif') 
    # Define the desired CRS (destination CRS)
#     dst_crs = CRS.from_epsg(crs)  # Example: EPSG 4326 is for WGS 84
    dst_crs = crs
    with rasterio.open(input_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return output_path
    
def geo_reference_with_gcps(input_path, output_path, gcps):
    # Open the input raster
    with rasterio.open(input_path) as src:
        # Define the transformation from GCPs
        transform = from_gcps(gcps)

        # Update metadata with new transformation
        kwargs = src.meta.copy()
        kwargs.update({
            'transform': transform
        })

        # Create the output raster with the new transformation
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            # Copy bands and other metadata
            dst.write(src.read(), indexes=range(1, src.count + 1))
    

def change_crs(input_path, output_path, new_crs_epsg):
    # Open the input raster
    with rasterio.open(input_path) as src:
        # Define the new CRS
        new_crs = f"EPSG:{new_crs_epsg}"

        # Reproject the raster
        transform, width, height = calculate_default_transform(
            src.crs, new_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': new_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=Resampling.nearest
                )


def shift_vector_file(gdf, outfile, shift_lon, shift_lat):

    # Define a function to shift a geometry in the northeast direction
    def shift_geometry(geometry, shift_degrees_lon, shift_degrees_lat):
        lon, lat = geometry.x, geometry.y
        new_lon = lon + (shift_degrees_lon / (np.cos(np.radians(lat)) * 111319.9))
        new_lat = lat + (shift_degrees_lat / 111319.9)
        return Point(new_lon, new_lat)

    # Apply the shift to all geometries in the GeoDataFrame
    gdf['geometry'] = gdf['geometry'].apply(shift_geometry, shift_degrees_lon=shift_lon, shift_degrees_lat=shift_lat)

    # Save the shifted GeoDataFrame back to a Shapefile
    gdf.to_file(outfile)
    
def write_GCP_file(template_file, out_file, matched_inliers_List):
    # Open the file for reading
    with open(template_file, 'r') as file:
        content = file.read()
    if os.path.exists(out_file):
        os.remove(out_file)
        print(f"File {out_file} has been deleted.")
    else:
        print(f"File {out_file} does not exist.")
    # Open (or create) a file for appending
    with open(out_file, 'a') as file:
        file.write(content)
        for i, val in enumerate(matched_inliers_List):
            file.write('\n')
            file.write(str(val[0][0]))
            file.write(', ')
            file.write(str(val[0][1]))
            file.write(', ')
            file.write(str(val[1][0]))
            file.write(', ')
            file.write(str(val[1][1]))
            file.write(', 1')