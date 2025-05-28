import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from pyproj import Transformer
import GeoUtils
from shapely.geometry import Point
import geopandas as gpd




def parse_points_file(filepath):
    sensor1, sensor2 = [], []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    for line in data_lines[1:]:  # skip header
        parts = line.split(',')
        lon1, lat1, lon2, lat2 = map(float, parts[:4])
        sensor1.append((lon1, lat1))
        sensor2.append((lon2, lat2))
    return sensor1, sensor2


def get_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def estimate_affine_transform(sensor1_utm, sensor2_utm):
    center = np.mean(sensor2_utm, axis=0)
    sensor1_centered = sensor1_utm - center
    sensor2_centered = sensor2_utm - center

    A, B = [], []
    for (x1, y1), (x2, y2) in zip(sensor2_centered, sensor1_centered):
        A.append([x1, y1, 0, 0, 1, 0])
        A.append([0, 0, x1, y1, 0, 1])
        B.append(x2)
        B.append(y2)

    A = np.array(A)
    B = np.array(B)
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, tx, ty = params

    T = np.array([
        [a, b, tx],
        [c, d, ty],
        [0, 0, 1]
    ])
    return T, center


def apply_affine(points, T, center):
    centered = points - center
    homog = np.hstack([centered, np.ones((centered.shape[0], 1))])
    transformed_centered = (T @ homog.T).T[:, :2]
    return transformed_centered + center

def reproject_to_utm(src_path, utm_crs, temp_path):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.bilinear)
                
def apply_affine_to_raster(raster_path_utm, output_path, T, center):
    with rasterio.open(raster_path_utm) as src:
        profile = src.profile
        data = src.read()
        original_transform = src.transform

        offset_affine = Affine.translation(-center[0], -center[1])
        recenter_affine = Affine.translation(center[0], center[1])
        affine_from_matrix = Affine(T[0, 0], T[0, 1], T[0, 2],
                                    T[1, 0], T[1, 1], T[1, 2])
        total_transform = recenter_affine * affine_from_matrix * offset_affine * original_transform

        profile.update(transform=total_transform)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)


def register_shapefiles(shp1_path, shp2_path, match_file_path, output_path):
    # Step 1: Load matched point pairs
    sensor1_wgs84, sensor2_wgs84 = parse_points_file(match_file_path)

    # Step 2: Determine UTM zone based on first point
    utm_epsg = get_utm_epsg(*sensor1_wgs84[0])
    transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    transformer_to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    # Step 3: Project points to UTM
    sensor1_utm = np.array([transformer_to_utm.transform(lon, lat) for lon, lat in sensor1_wgs84])
    sensor2_utm = np.array([transformer_to_utm.transform(lon, lat) for lon, lat in sensor2_wgs84])

    # Step 4: Estimate affine transform
    T, center = estimate_affine_transform(sensor1_utm, sensor2_utm)

    # Step 5: Load shapefile 2 and transform its geometry
    shp2 = gpd.read_file(shp2_path)
    shp2_coords = np.array([transformer_to_utm.transform(geom.x, geom.y) for geom in shp2.geometry])
    shp2_registered_coords = apply_affine(shp2_coords, T, center)
    shp2_registered_lonlat = [transformer_to_wgs.transform(x, y) for x, y in shp2_registered_coords]
    shp2_transformed = shp2.copy()
    shp2_transformed['geometry'] = [Point(lon, lat) for lon, lat in shp2_registered_lonlat]

    # Step 6: Save output
    shp2_transformed.to_file(output_path)
    print(f"Registered shapefile saved to: {output_path}")
    
def register_raster(raster1_path, raster2_path, match_file_path, output_path_wgs84):
    # Load matched pairs
    sensor1_wgs84, sensor2_wgs84 = parse_points_file(match_file_path)

    # Use UTM based on sensor1
    utm_epsg = get_utm_epsg(*sensor1_wgs84[0])
    transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    # Convert matched WGS84 points to UTM
    sensor1_utm = np.array([transformer_to_utm.transform(lon, lat) for lon, lat in sensor1_wgs84])
    sensor2_utm = np.array([transformer_to_utm.transform(lon, lat) for lon, lat in sensor2_wgs84])

    # Compute affine matrix in UTM
    T, center = estimate_affine_transform(sensor1_utm, sensor2_utm)

    # Step 1: Reproject raster2 to UTM
    temp_utm_raster = "temp_utm_raster.tif"
    reproject_to_utm(raster2_path, utm_epsg, temp_utm_raster)

    # Step 2: Apply affine transform in UTM
    registered_utm_raster = "registered_utm_raster.tif"
    apply_affine_to_raster(temp_utm_raster, registered_utm_raster, T, center)

    # Step 3: Reproject registered UTM raster to WGS84
    with rasterio.open(registered_utm_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': "EPSG:4326",
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path_wgs84, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear)
    print(f"Registered raster written to: {output_path_wgs84}")
    # Cleanup
    os.remove(temp_utm_raster)
    os.remove(registered_utm_raster)