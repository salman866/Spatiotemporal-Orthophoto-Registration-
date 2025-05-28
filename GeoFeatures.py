#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import ast
from GeoUtils import getDistance, getBearing
from tqdm import tqdm
import time
import geopandas as gpd
from shapely.geometry import LineString
import math


# In[ ]:


def getDistanceDictionary(gdf):
    dist_dict = {}
    for i, geomSrc in enumerate(gdf.geometry):
        dist_list = []
        coordinatesSrc = (geomSrc.y,geomSrc.x)
        for j, geomDst in enumerate(gdf.geometry):
            coordinatesDest = (geomDst.y,geomDst.x)
            dist = getDistance(coordinatesSrc, coordinatesDest)
            dist_list.append(dist)
        sorted_list = sorted(dist_list)[1:]
        dist_dict["P"+str(i+1)] = sorted_list
    return dist_dict    

# Returns the max distance threshold for which neighbor_percentile% of the points contains at least minNeighbors
def getDistThresholdNeighbors(dist_dict, minNeighbors, neighbor_percentile):
    dist_closestPoint = [] # list of distances  
    for i, key in enumerate(dist_dict):
        dist_closestPoint.append(dist_dict[key][minNeighbors-1])

    distThresholdNeighbors = np.percentile(dist_closestPoint, neighbor_percentile)
    return distThresholdNeighbors

# def calculateFeatureDescriptor(gdf, distanceThresholdNeighbors):
#     featureDescriptor = {}
#     # Loop through all the points
#     for i, geomSrc in enumerate(gdf.geometry):
#         featureDescriptor["P"+str(i+1)] = []
#         coordinatesSrc = (geomSrc.y,geomSrc.x)
#         # Loop through all the points and calculate distance and angle between two points
#         for j, geomDst in enumerate(gdf.geometry):
#             coordinatesDest = (geomDst.y,geomDst.x)
#             dist = getDistance(coordinatesSrc, coordinatesDest)
#             if dist < distanceThresholdNeighbors:
#                 featureDescriptor["P"+str(i+1)].append((j+1,dist, getBearing(coordinatesSrc, coordinatesDest)))
        
#         # Sort wrt to distance
# #         sorted_list = sorted(featureDescriptor["P"+str(i+1)], key=lambda x: x[1])
# #         sorted_list_selected = sorted_list[1:noOfNearestNeighbors+1]
#         # Sort wrt to angle
#         sorted_list = sorted(featureDescriptor["P"+str(i+1)], key=lambda x: x[2])[1:]
#         featureDescriptor["P"+str(i+1)] = sorted_list
            
#     return featureDescriptor
def get_feature_descriptor(gdf,centroid_neighbor_dist):
    featureDescriptor = {}
    for i, val in enumerate(centroid_neighbor_dist):
        featureDescriptor[val] = []
        feat_desc_list = []
        center_id = int(val[1:])
        ref_id = centroid_neighbor_dist[val][0][0]
        geometery_center = gdf.loc[gdf['id'] == center_id].geometry.values[0]
        geometery_ref = gdf.loc[gdf['id'] == ref_id].geometry.values[0]
        ref_angle = getBearing((geometery_center.y,geometery_center.x), (geometery_ref.y,geometery_ref.x))

    #     print('center: (' + str(geometery_center.x)+','+str(geometery_center.y)+')')
    #     print('ref: (' + str(geometery_ref.x)+','+str(geometery_ref.y)+')')
        for j, point in enumerate(centroid_neighbor_dist[val]):
            point_id = point[0]
            geometery_point = gdf.loc[gdf['id'] == point_id].geometry.values[0]
            point_angle = getBearing((geometery_center.y,geometery_center.x), (geometery_point.y,geometery_point.x))
            angle = point_angle-ref_angle
            if (angle < 0):
                angle = angle+360
    #         featureDescriptor[val].append((point[0],point[1], angle))
            feat_desc_list.append((point[0],point[1], angle))
        sorted_list = sorted(feat_desc_list, key=lambda x: x[2])
        featureDescriptor[val] = sorted_list
    return featureDescriptor

# Append feature descriptor in dataframe of geopandas
def appendFeatureDescriptor(gdf, fd):
    fdList = list(fd.values())
    gdf['fd'] = fdList
    gdf['fd'] = gdf['fd'].astype(str)
    return gdf

# def matchFeatures(gdf_S1, gdf_S2, gps_error_dia, residualErrorThreshold):
#     featureDescriptor_S1 = gdf_S1['fd'].apply(ast.literal_eval).tolist()
#     featureDescriptor_S2 = gdf_S2['fd'].apply(ast.literal_eval).tolist()
#     matchNetFeatures_dict = {}
#     matchIndvFeatures_dict = {}
#     print('Total iterations: ' + str(len(featureDescriptor_S1)))
#     for i,fd_S1 in tqdm(enumerate(featureDescriptor_S1)):
#         srcPoint_feature = fd_S1
#         coordinatesSrc = (gdf_S1.loc[gdf_S1['id'] == i+1].geometry.y.values[0],
#                       gdf_S1.loc[gdf_S1['id'] == i+1].geometry.x.values[0])

#         matchNetFeatures_dict["P"+str(i+1)] = []
#         matchIndvFeatures_dict["P"+str(i+1)] = []
#         matchNetFeatures_list = []
#         matchIndvFeatures_list = []
#         for j,fd_S2 in enumerate(featureDescriptor_S2):
#             dstPoint_feature = fd_S2
#             coordinatesDest = (gdf_S2.loc[gdf_S2['id'] == j+1].geometry.y.values[0],
#                       gdf_S2.loc[gdf_S2['id'] == j+1].geometry.x.values[0])
#             distance_PS1_PS2 = getDistance(coordinatesSrc, coordinatesDest)
#             if (distance_PS1_PS2 <= gps_error_dia) and (len(srcPoint_feature) == len(dstPoint_feature)):
#                 errorDistance = 0
#                 errorAngle = 0
#                 for k,dstPoint_feature_instance in enumerate(dstPoint_feature):
#                     resDistSq = ((srcPoint_feature[k][1]-dstPoint_feature_instance[1])*10)**2
#                     resAngle = abs(srcPoint_feature[k][2]-dstPoint_feature_instance[2])
#                     errorDistance = errorDistance + resDistSq
#                     errorAngle = errorAngle + resAngle
# #                 netFeatureDistance = distanceWeightage*errorDistance + (1-distanceWeightage)*errorAngle
#                 netFeatureDistance = errorDistance
#                 if netFeatureDistance < residualErrorThreshold:
#                     matchNetFeatures_list.append((j+1,netFeatureDistance))
#         if matchNetFeatures_list:
#             matchNetFeatures_list = sorted(matchNetFeatures_list, key=lambda x: x[1])
#             matchNetFeatures_dict["P"+str(i+1)] = matchNetFeatures_list[0]

#     return matchNetFeatures_dict

def matchFeatures(gdf_S1, gdf_S2):
    featureDescriptor_S1 = gdf_S1['fd'].apply(ast.literal_eval).tolist()
    featureDescriptor_S2 = gdf_S2['fd'].apply(ast.literal_eval).tolist()
    matchNetFeatures_dict = {}
    matchIndvFeatures_dict = {}
    print('Total iterations: ' + str(len(featureDescriptor_S1)))
    for i,fd_S1 in tqdm(enumerate(featureDescriptor_S1)):
        srcPoint_feature = fd_S1
        coordinatesSrc = (gdf_S1.loc[gdf_S1['id'] == i+1].geometry.y.values[0],
                      gdf_S1.loc[gdf_S1['id'] == i+1].geometry.x.values[0])

        matchNetFeatures_dict["P"+str(i+1)] = []
        matchIndvFeatures_dict["P"+str(i+1)] = []
        matchNetFeatures_list = []
        matchIndvFeatures_list = []
        for j,fd_S2 in enumerate(featureDescriptor_S2):
            dstPoint_feature = fd_S2
            coordinatesDest = (gdf_S2.loc[gdf_S2['id'] == j+1].geometry.y.values[0],
                      gdf_S2.loc[gdf_S2['id'] == j+1].geometry.x.values[0])
            distance_PS1_PS2 = getDistance(coordinatesSrc, coordinatesDest)
            max_gps_error_dia = 16
            if (distance_PS1_PS2 <= max_gps_error_dia) and (len(srcPoint_feature) == len(dstPoint_feature)):
                errorDistance = 0
                errorAngle = 0
                src_dist = [t[1] for t in srcPoint_feature]
                src_brg = [t[2] for t in srcPoint_feature]
                dst_dist = [t[1] for t in dstPoint_feature]
                dst_brg = [t[2] for t in dstPoint_feature]
                brg_diff = abs(np.array(src_brg) - np.array(dst_brg))
                dist_diff = abs(np.array(src_dist) - np.array(dst_dist))
                brg_flag = np.any(brg_diff > 3)
                dist_tol = np.array(src_dist)*0.075
                dist_flag = np.any(dist_diff > dist_tol)
                if ~brg_flag and ~dist_flag:
#                     dist_mse = ((np.array(src_dist) - np.array(dst_dist))*10)**2
                    netFeatureDistance = np.sum(dist_diff)
                    matchNetFeatures_list.append((j+1,netFeatureDistance))
                    
        if matchNetFeatures_list:
            matchNetFeatures_list = sorted(matchNetFeatures_list, key=lambda x: x[1])
            matchNetFeatures_dict["P"+str(i+1)] = matchNetFeatures_list[0]

    return matchNetFeatures_dict



# Generate a list of source destination pair matched points
def generateSrcToDstPointsList(gdf_S1, gdf_S2, matched_dict):
    coordinatesList = []
    for i, key in enumerate(matched_dict):
        pointId_src = int(key[1:])
        pointId_dst = matched_dict[key][0]
        coordinatesSrc = (gdf_S1.loc[gdf_S1['id'] == pointId_src].geometry.x.values[0],
                      gdf_S1.loc[gdf_S1['id'] == pointId_src].geometry.y.values[0])
        coordinatesDst = (gdf_S2.loc[gdf_S2['id'] == pointId_dst].geometry.x.values[0],
                      gdf_S2.loc[gdf_S2['id'] == pointId_dst].geometry.y.values[0])
        coordinatesList.append((coordinatesSrc,coordinatesDst))
    return coordinatesList

def getDistBearingList(matchedPointsLists):
    distList = []
    brgList = []
    for i, val in enumerate(matchedPointsLists):
        src_point = (val[0][1], val[0][0])
        dst_point = (val[1][1], val[1][0])
        distList.append(getDistance(src_point, dst_point))
        brgList.append(getBearing(src_point, dst_point))
    return (distList, brgList)

# def findMostEntriesRange(input_list):
#     input_list.sort()  # Sort the list to ensure it is in ascending order
#     differences = [j - i for i, j in zip(input_list[:-1], input_list[1:])]
#     diff_Thresh = np.median(differences)
#     max_entries_count = 0
#     current_entries_count = 1
#     current_range_start = input_list[0]
#     max_range_start = input_list[0]

#     for i in range(1, len(input_list)):
#         if input_list[i] - input_list[i - 1] <= diff_Thresh:
#             current_entries_count += 1
#         else:
#             current_entries_count = 1
#             current_range_start = input_list[i]

#         if current_entries_count > max_entries_count:
#             max_entries_count = current_entries_count
#             max_range_start = current_range_start

#     max_range_start_index = input_list.index(max_range_start)
#     max_range_end = input_list[max_range_start_index+max_entries_count-1]
#     return max_range_start, max_range_end

def rangeThresholdBearing(input_list, bins):
#     bins = int((max(input_list)-min(input_list))/bin_length)
    bins = 24
    # Create a histogram
    hist, bins = np.histogram(input_list, bins=bins, density=True)
    # Find the bin with the highest density
    max_density_bin = np.argmax(hist)

    if (max_density_bin == len(bins)-1):
        center_val = (bins[max_density_bin]+bins[max_density_bin-1])/2
    else:
        center_val = (bins[max_density_bin]+bins[max_density_bin+1])/2
        
    lower_val = (center_val - 12) % 360.0
    upper_val = (center_val + 12) % 360.0

    return lower_val, upper_val

def rangeThresholdDistance(input_list, bins):
#     bins = int((max(input_list)-min(input_list))/bin_length)
    bins = 24
    # Create a histogram
    hist, bins = np.histogram(input_list, bins=bins, density=True)
    # Find the bin with the highest density
    max_density_bin = np.argmax(hist)
    if (max_density_bin == len(bins)-1):
        center_val = (bins[max_density_bin]+bins[max_density_bin-1])/2
    else:
        center_val = (bins[max_density_bin]+bins[max_density_bin+1])/2
        
    lower_val = center_val - center_val*0.2
    upper_val = center_val + center_val*0.2

    return lower_val, upper_val

def getDistBearingThresholds(distList, brgList, bins_dist, bins_brg):
    dist_lower_thresh, dist_upper_thresh = rangeThresholdDistance(distList, bins_dist)
    brg_lower_thresh, brg_upper_thresh = rangeThresholdBearing(brgList, bins_brg)
    return (dist_lower_thresh, dist_upper_thresh, brg_lower_thresh, brg_upper_thresh)

def removeOutlierPoints(gdf_S1, gdf_S2, matched_dict, thresholds):
    matched_dict = removeDuplicates(matched_dict)
    matched_dict_inliers = {}
    for i, key in enumerate(matched_dict):
        pointId_src = int(key[1:])
        pointId_dst = matched_dict[key][0]
        coordinatesSrc = (gdf_S1.loc[gdf_S1['id'] == pointId_src].geometry.y.values[0],
                      gdf_S1.loc[gdf_S1['id'] == pointId_src].geometry.x.values[0])
        coordinatesDst = (gdf_S2.loc[gdf_S2['id'] == pointId_dst].geometry.y.values[0],
                      gdf_S2.loc[gdf_S2['id'] == pointId_dst].geometry.x.values[0])
        distance = getDistance(coordinatesSrc, coordinatesDst)
        bearing = getBearing(coordinatesSrc, coordinatesDst)
        dist_flag = (distance >= thresholds[0]) and (distance <= thresholds[1])
        angle_flag = angle_in_range(bearing, thresholds[2], thresholds[3])
        if (dist_flag and angle_flag):
            matched_dict_inliers[key] = matched_dict[key]
    return matched_dict_inliers

# Remove duplicates if the dictionary contains two or more same destination points 
#than the smallest residual distance point is retained 
def removeDuplicates(input_dict):
    seen = {}
    smallest_second = {}

    for key, (value1, value2) in input_dict.items():
        if value1 not in seen or value2 < seen[value1][1]:
            seen[value1] = (key, value2)
            smallest_second[value1] = key

    result_dict = {smallest_second[value1]: (value1, seen[value1][1]) for value1 in seen}

    return result_dict

def angle_in_range(alpha, lower, upper):
    return (alpha - lower) % 360 <= (upper - lower) % 360

# Draw lines between matched pair points
# def drawLinesMatchedPoints(coordinatesList, lineShp):
#     for i, val in enumerate(coordinatesList):
#         coordinatesSrc = val[0]
#         coordinatesDst = val[1]
#         #get list of points
#         xyList = []
#         rowName = str(i)
#         xyList.append(coordinatesSrc)
#         xyList.append(coordinatesDst)
#         #save record and close shapefile
#         rowDict = {
#         'geometry' : {'type':'LineString',
#                          'coordinates': xyList},
#         'properties': {'Name' : rowName},
#         }
#         lineShp.write(rowDict)
#     #close fiona object
#     lineShp.close()

def drawLinesMatchedPoints(coordinates_list, out_file_path, crs):
    # Create a GeoDataFrame with LineString geometry column for each pair of coordinates
    geometries = [LineString(coords) for coords in coordinates_list];
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs);
    gdf.to_file(out_file_path);
    return gdf

def get_angle(point1, point2, point3):
    # Define vectors representing the lines
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [point2[0] - point3[0], point2[1] - point3[1]]

    # Calculate dot product and cross product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Calculate angle between the vectors
    angle_rad = math.atan2(cross_product, dot_product)
    angle_deg = math.degrees(angle_rad)

    # Ensure the angle is between 0 and 360
    if angle_deg < 0:
        angle_deg += 360
        
    return angle_deg

def get_neighbor_distances(gdf, distanceThresholdNeighbors):
    neighbor_dist = {}
    # Loop through all the points
    for i, geomSrc in enumerate(gdf.geometry):
        neighbor_dist["P"+str(i+1)] = []
        coordinatesSrc = (geomSrc.y,geomSrc.x)
        # Loop through all the points and calculate distance and angle between two points
        for j, geomDst in enumerate(gdf.geometry):
            coordinatesDest = (geomDst.y,geomDst.x)
            dist = getDistance(coordinatesSrc, coordinatesDest)
            if dist < distanceThresholdNeighbors:
                neighbor_dist["P"+str(i+1)].append((j+1,dist))
        
        # Sort wrt to distance
#         sorted_list = sorted(featureDescriptor["P"+str(i+1)], key=lambda x: x[1])
#         sorted_list_selected = sorted_list[1:noOfNearestNeighbors+1]
        # Sort wrt to dist
        sorted_list = sorted(neighbor_dist["P"+str(i+1)], key=lambda x: x[1])[1:]
        neighbor_dist["P"+str(i+1)] = sorted_list
            
    return neighbor_dist

def get_add_matches_theshold(matched_inliers_List):
    dist_list = []
    brg_list = []
    for i, points in enumerate(matched_inliers_List):
        dst = getDistance((points[0][1], points[0][0]), (points[1][1], points[1][0]))
        brg = getBearing((points[0][1], points[0][0]), (points[1][1], points[1][0]))
        dist_list.append(dst)
        brg_list.append(brg)
    dst_mean = np.mean(dist_list)
    brg_mean = np.median(brg_list)
    add_thresholds = [dst_mean-dst_mean*0.15, dst_mean+dst_mean*0.15, brg_mean-5, brg_mean+5]
    return add_thresholds

def add_feature_matches(s1_centroids_transformed, s2_centroids_transformed, thresh):
    add_matched_inliers_dict = {}

    for _, s1_id in tqdm(enumerate(s1_centroids_transformed.id)):
#     for _, s1_id in enumerate(s1_centroids_transformed.id):
        # Access the geometry and other columns in each row
        s1_coord = (s1_centroids_transformed.loc[s1_centroids_transformed['id'] == s1_id].geometry.y.values[0],
                          s1_centroids_transformed.loc[s1_centroids_transformed['id'] == s1_id].geometry.x.values[0])
        for _, s2_id in enumerate(s2_centroids_transformed.id):
            s2_coord = (s2_centroids_transformed.loc[s2_centroids_transformed['id'] == s2_id].geometry.y.values[0],
                          s2_centroids_transformed.loc[s2_centroids_transformed['id'] == s2_id].geometry.x.values[0])
            dist = getDistance(s1_coord, s2_coord)
            bearing = getBearing(s1_coord, s2_coord)
            dist_flag = (dist >= thresh[0]) and (dist <= thresh[1])
            angle_flag = angle_in_range(bearing, thresh[2], thresh[3])
            if (dist_flag and angle_flag):
                #s1_coord = (s1_coord[1],s1_coord[0])
                #s2_coord = (s2_coord[1],s2_coord[0])
                #match = (s1_coord,s2_coord)
#                 add_matched_inliers_list.append(match)
                brg_diff = abs(bearing - (thresh[2]+thresh[3])/2)
                match_point = (s2_id,brg_diff)
                add_matched_inliers_dict['P'+str(s1_id)] = match_point
                break
    add_matched_inliers_dict = removeDuplicates(add_matched_inliers_dict)        
    return add_matched_inliers_dict