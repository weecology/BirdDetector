#Prepare all training sets
import rasterio as rio
import numpy as np
import os
from shapely.geometry import Point, box
import geopandas as gpd
import cv2
import pandas as pd

def split_test_train(annotations, split = 0.9):
    """Split annotation in train and test by image"""
    #Currently want to mantain the random split
    np.random.seed(0)
    
    #unique annotations for the bird detector
    #annotations = annotations.groupby("selected_i").apply(lambda x: x.head(1))
    
    #add to train_names until reach target split threshold
    image_names = annotations.image_path.unique()
    target = int(annotations.shape[0] * split)
    counter = 0
    train_names = []
    for x in image_names:
        if target > counter:
            train_names.append(x)
            counter+=annotations[annotations.image_path == x].shape[0]
        else:
            break
        
    train = annotations[annotations.image_path.isin(train_names)]
    test = annotations[~(annotations.image_path.isin(train_names))]
    
    return train, test

def shapefile_to_annotations(shapefile, rgb, savedir=".", box_points=False, confidence_filter=False,buffer_size=0.25):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
        buffer_size: size of point to box expansion in map units of the target object, meters for projected data, pixels for unprojected data 
    Returns:
        results: a pandas dataframe
    """
    #Read shapefile
    gdf = gpd.read_file(shapefile)
    if confidence_filter:
        gdf = gdf[gdf.Confidence==1]    
    gdf = gdf[~gdf.geometry.isnull()]
        
    #raster bounds
    with rio.open(rgb) as src:
        left, bottom, right, top = src.bounds
        resolution = src.res[0]
        
    #define in image coordinates and buffer to create a box
    if box_points:
        gdf["geometry"] = gdf.geometry.boundary.centroid
        gdf["geometry"] =[Point(x,y) for x,y in zip(gdf.geometry.x.astype(float), gdf.geometry.y.astype(float))]
    
    gdf["geometry"] = [box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(buffer_size).bounds.values]
        
    #get coordinates
    df = gdf.geometry.bounds
    df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
    
    #Transform project coordinates to image coordinates
    df["tile_xmin"] = (df.xmin - left)/resolution
    df["tile_xmin"] = df["tile_xmin"].astype(int)
    
    df["tile_xmax"] = (df.xmax - left)/resolution
    df["tile_xmax"] = df["tile_xmax"].astype(int)
    
    #UTM is given from the top, but origin of an image is top left
    
    df["tile_ymax"] = (top - df.ymin)/resolution
    df["tile_ymax"] = df["tile_ymax"].astype(int)
    
    df["tile_ymin"] = (top - df.ymax)/resolution
    df["tile_ymin"] = df["tile_ymin"].astype(int)    
    
    df["label"] = "Bird"
    
    #add filename
    df["image_path"] = os.path.basename(rgb)
    
    #select columns
    result = df[["image_path","tile_xmin","tile_ymin","tile_xmax","tile_ymax","label"]]
    result = result.rename(columns={"tile_xmin":"xmin","tile_ymin":"ymin","tile_xmax":"xmax","tile_ymax":"ymax"})
    
    #ensure no zero area polygons due to rounding to pixel size
    result = result[~(result.xmin == result.xmax)]
    result = result[~(result.ymin == result.ymax)]
    
    return result

def check_shape(df):
    """Make sure no annotations fall off edges of crops
    Args:
        df: a pandas dataframe with image_path, xmax, ymax
    """
    updated_data = []
    for name, group in df.groupby("image_path"):
        img = cv2.imread("/orange/ewhite/b.weinstein/generalization/crops/{}".format(name))
        height = img.shape[0]
        width = img.shape[1]
        group["xmax"] = group.xmax.apply(lambda x: x if x < width else width)
        group["ymax"] = group.ymax.apply(lambda y: y if y < height else height)
        updated_data.append(group)
        
    df = pd.concat(updated_data)    