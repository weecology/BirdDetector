#Prepare all training sets
import comet_ml
import cv2
from model import BirdDetector
import glob
from PIL import ImageFile
from pytorch_lightning.loggers import CometLogger
from deepforest import preprocess
from deepforest import visualize
from deepforest.model import create_model
from deepforest import main
from datetime import datetime
from augmentation import get_transform
from shapely.geometry import Point, box
from utils import start_cluster

import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import os
import PIL
import tempfile
import torch
import gc
from pytorch_lightning.plugins import DDPPlugin

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

def prepare_palmyra(generate=True):
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/palmyra_test.csv"
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/palmyra_train.csv"      
    if generate:      
        df = shapefile_to_annotations(
            shapefile="/orange/ewhite/everglades/Palmyra/Dudley_projected.shp",
            rgb="/orange/ewhite/everglades/Palmyra/Dudley_projected.tif", box_points=False, confidence_filter=False, buffer_size=0.15)
        df.to_csv("Figures/test_annotations.csv",index=False)
        
        src = rio.open("/orange/ewhite/everglades/Palmyra/Dudley_projected.tif")
        numpy_image = src.read()
        numpy_image = np.moveaxis(numpy_image,0,2)
        numpy_image = numpy_image[:,:,:3].astype("uint8")
        
        test_annotations = preprocess.split_raster(numpy_image=numpy_image,
                                                   annotations_file="Figures/test_annotations.csv",
                                                   patch_size=1400, patch_overlap=0.05, base_dir="/orange/ewhite/b.weinstein/generalization/crops/", image_name="Dudley_projected.tif")
        
        test_annotations.to_csv(test_path,index=False)
        
        src = rio.open("/orange/ewhite/everglades/Palmyra/CooperStrawn_53m_tile_clip_projected.tif")
        numpy_image = src.read()
        numpy_image = np.moveaxis(numpy_image,0,2)
        training_image = numpy_image[:,:,:3].astype("uint8")
        
        df = shapefile_to_annotations(
            shapefile="/orange/ewhite/everglades/Palmyra/TNC_Cooper_annotation_03192021.shp", 
            rgb="/orange/ewhite/everglades/Palmyra/CooperStrawn_53m_tile_clip_projected.tif", box_points=True,
            confidence_filter=True, buffer_size=0.15
        )
    
        df.to_csv("Figures/training_annotations.csv",index=False)
        
        train_annotations_1 = preprocess.split_raster(
            numpy_image=training_image,
            annotations_file="Figures/training_annotations.csv",
            patch_size=1400,
            patch_overlap=0.05,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
            image_name="CooperStrawn_53m_tile_clip_projected.tif",
            allow_empty=False
        )
        
        train_annotations = pd.concat([train_annotations_1])
        train_annotations.to_csv(train_path,index=False)
            
    return {"train":train_path, "test":test_path}

def prepare_penguin(generate=True):
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/penguins_test.csv"
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/penguins_train.csv"
    
    if generate:
        df = shapefile_to_annotations(
            shapefile="/orange/ewhite/b.weinstein/penguins/cape_wallace_survey_8.shp",
            rgb="/orange/ewhite/b.weinstein/penguins/cape_wallace_survey_8.tif", buffer_size=0.15)
        df.to_csv("/orange/ewhite/b.weinstein/penguins/test_annotations.csv",index=False)
        
        src = rio.open("/orange/ewhite/b.weinstein/penguins/cape_wallace_survey_8.tif")
        numpy_image = src.read()
        numpy_image = np.moveaxis(numpy_image,0,2)
        numpy_image = numpy_image[:,:,:3].astype("uint8")
        
        test_annotations = preprocess.split_raster(numpy_image=numpy_image, annotations_file="/orange/ewhite/b.weinstein/penguins/test_annotations.csv", patch_size=500, patch_overlap=0.05,
                                                   base_dir="/orange/ewhite/b.weinstein/generalization/crops", image_name="cape_wallace_survey_8.tif")
        
        test_annotations.to_csv(test_path,index=False)
    
        src = rio.open("/orange/ewhite/b.weinstein/penguins/offshore_rocks_cape_wallace_survey_4.tif")
        numpy_image = src.read()
        numpy_image = np.moveaxis(numpy_image,0,2)
        training_image = numpy_image[:,:,:3].astype("uint8")
        
        df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/penguins/offshore_rocks_cape_wallace_survey_4.shp", rgb="/orange/ewhite/b.weinstein/penguins/offshore_rocks_cape_wallace_survey_4.tif", buffer_size=0.1)
    
        df.to_csv("/orange/ewhite/b.weinstein/penguins/training_annotations.csv",index=False)
        
        train_annotations = preprocess.split_raster(
            numpy_image=training_image,
            annotations_file="/orange/ewhite/b.weinstein/penguins/training_annotations.csv",
            patch_size=500,
            patch_overlap=0.05,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops",
            image_name="offshore_rocks_cape_wallace_survey_4.tif",
            allow_empty=False
        )
        
        train_annotations.to_csv(train_path,index=False)
        
    return {"train":train_path, "test":test_path}

def prepare_everglades():
    
    #too large to repeat here, see create_model.py
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/everglades_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/everglades_test.csv"
    
    return {"train":train_path, "test":test_path}

def prepare_terns(generate=True):
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/tern_test.csv"
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/terns_train.csv"        
    if generate:   
        df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/terns/birds.shp",
                                      rgb="/orange/ewhite/b.weinstein/terns/seabirds_rgb.tif", buffer_size=0.15)
        df.to_csv("/orange/ewhite/b.weinstein/terns/seabirds_rgb.csv")
        
        annotations = preprocess.split_raster(
            path_to_raster="/orange/ewhite/b.weinstein/terns/seabirds_rgb.tif",
            annotations_file="/orange/ewhite/b.weinstein/terns/seabirds_rgb.csv",
            patch_size=700,
            patch_overlap=0,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops",
            image_name="seabirds_rgb.tif",
            allow_empty=False
        )
        
        #split into train test
        train, test = split_test_train(annotations)
        train.to_csv(train_path,index=False)    
        
        #Test        
        test.to_csv(test_path,index=False)
    
    return {"train":train_path, "test":test_path}

def prepare_hayes(generate=True):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/hayes_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/hayes_test.csv"    
    if generate:
        hayes_albatross_train = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Albatross_Labels/albatross_train_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])
                   
        hayes_albatross_test = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Albatross_Labels/albatross_test_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])
        
        hayes_penguin_train = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Penguin_Labels/penguin_train_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])
                   
        hayes_penguin_val = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Penguin_Labels/penguin_val_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])
            
        hayes_penguin_test = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Penguin_Labels/penguin_test_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])
        
        
        hayes_albatross_val = pd.read_csv("/orange/ewhite/b.weinstein/Hayes/Label/Albatross_Labels/albatross_val_annotations_final.csv",
                      names=["image_path","xmin","ymin","xmax","ymax","label"])    
        
        train_annotations = pd.concat([hayes_albatross_train, hayes_albatross_test, hayes_penguin_train, hayes_penguin_test, hayes_penguin_val])
        train_annotations.label = "Bird"
        
        
        train_images = train_annotations.image_path.sample(n=500)
        train_annotations = train_annotations[train_annotations.image_path.isin(train_images)]
        train_annotations.to_csv(train_path, index=False)
        
        hayes_albatross_val.label="Bird"
        hayes_albatross_val_images = hayes_albatross_val.image_path.sample(n=100)
        hayes_albatross_val = hayes_albatross_val[hayes_albatross_val.image_path.isin(hayes_albatross_val_images)]
        hayes_albatross_val.to_csv(test_path, index=False)
    
    return {"train":train_path, "test":test_path}
    
def prepare_pfeifer(generate=True):
    
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/pfeifer_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/pfeifer_test.csv"
    
    train_annotations = []
    test_annotations = []
    if generate:   
        for x in glob.glob("/orange/ewhite/b.weinstein/pfeifer/*.shp")[:1]:
            basename = os.path.splitext(os.path.basename(x))[0]
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/pfeifer/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/pfeifer/{}.tif".format(basename))
            df.to_csv("/orange/ewhite/b.weinstein/pfeifer/{}.csv".format(basename))
            
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/pfeifer/{}.tif".format(basename),
                annotations_file="/orange/ewhite/b.weinstein/pfeifer/{}.csv".format(basename),
                patch_size=450,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            test_annotations.append(annotations)
        test_annotations = pd.concat(test_annotations)
        test_annotations.to_csv(test_path, index=False)
            
        for x in glob.glob("/orange/ewhite/b.weinstein/pfeifer/*.shp")[1:]:
            print(x)
            basename = os.path.splitext(os.path.basename(x))[0]
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/pfeifer/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/pfeifer/{}.tif".format(basename))
            df.to_csv("/orange/ewhite/b.weinstein/pfeifer/{}.csv".format(basename))
            
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/pfeifer/{}.tif".format(basename),
                annotations_file="/orange/ewhite/b.weinstein/pfeifer/{}.csv".format(basename),
                patch_size=450,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            train_annotations.append(annotations)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations.to_csv(train_path, index=False)
        
    return {"train":train_path, "test":test_path}
        
def prepare_murres(generate=True):
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/murres_test.csv"
    if generate:   
        df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/murres/DJI_0019.shp",
                                      rgb="/orange/ewhite/b.weinstein/murres/DJI_0019.JPG", buffer_size=25)
        df.to_csv("/orange/ewhite/b.weinstein/murres/DJI_0019.csv")
        
        annotations = preprocess.split_raster(
            path_to_raster="/orange/ewhite/b.weinstein/murres/DJI_0019.JPG",
            annotations_file="/orange/ewhite/b.weinstein/murres/DJI_0019.csv",
            patch_size=800,
            patch_overlap=0,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops",
            allow_empty=False
        )
        
        #Test        
        annotations.to_csv(test_path,index=False)    
    
    return {"test":test_path}

def prepare_pelicans(generate=True):
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/pelicans_test.csv"
    if generate:   
        df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/pelicans/AWPE_Pigeon_Lake_2020_DJI_0005.shp",
                                      rgb="/orange/ewhite/b.weinstein/pelicans/AWPE_Pigeon_Lake_2020_DJI_0005.JPG", buffer_size=30)
        df.to_csv("/orange/ewhite/b.weinstein/pelicans/AWPE_Pigeon_Lake_2020_DJI_0005.csv")
        
        annotations = preprocess.split_raster(
            path_to_raster="/orange/ewhite/b.weinstein/pelicans/AWPE_Pigeon_Lake_2020_DJI_0005.JPG",
            annotations_file="/orange/ewhite/b.weinstein/pelicans/AWPE_Pigeon_Lake_2020_DJI_0005.csv",
            patch_size=800,
            patch_overlap=0,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops",
            allow_empty=False
        )
        
        #Test        
        annotations.to_csv(test_path,index=False)    
    
    return {"test":test_path}

def prepare_schedl(generate=True):
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/schedl_test.csv"
    
    test_annotations = []
    if generate:   
        for x in glob.glob("/orange/ewhite/b.weinstein/schedl/*.shp")[:1]:
            basename = os.path.splitext(os.path.basename(x))[0]
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/schedl/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/schedl/{}.JPG".format(basename), buffer_size=30)
            df.to_csv("/orange/ewhite/b.weinstein/schedl/{}.csv".format(basename))
            
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/schedl/{}.JPG".format(basename),
                annotations_file="/orange/ewhite/b.weinstein/schedl/{}.csv".format(basename),
                patch_size=800,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            test_annotations.append(annotations)
        test_annotations = pd.concat(test_annotations)
        test_annotations.to_csv(test_path)
        
    return {"test":test_path}

def prepare_monash(generate=True):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/Monash_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/Monash_test.csv"
        
    if generate:
        client = start_cluster.start(cpus=30, mem_size="10GB")
  
        #Find all the shapefiles
        shps = glob.glob("/orange/ewhite/b.weinstein/Monash/**/*.shp",recursive=True)
        shps = [x for x in shps if not "AQUA" in x]
        
        annotation_list = []
        matched_tiles = []
        for x in shps:
            components = os.path.basename(x).split("_")
            tif_path = "/orange/ewhite/b.weinstein/Monash/Transect {letter}/Transect {letter} {year}/Transect_{letter}_{year}.tif".format(letter=components[1],year=components[2])
            jpg_path = "/orange/ewhite/b.weinstein/Monash/Transect {letter}/Transect {letter} {year}/Transect_{letter}_{year}.jpg".format(letter=components[1],year=components[2])
            
            if os.path.exists(tif_path):
                rgb_path = tif_path
            elif os.path.exists(jpg_path):
                rgb_path = jpg_path
            else:
                print("Cannot find corresponding image to annotations {}".format(x))
                continue
                
            annotations = shapefile_to_annotations(shapefile=x, rgb=rgb_path)
            annotations["image_path"] = os.path.basename(rgb_path)
            
            if "Claire" in  x:
                annotations["annotator"] = "Claire"
            else:
                annotations["annotator"] = "Karina"
                
            annotation_list.append(annotations)
            matched_tiles.append(rgb_path)
            
        input_data = pd.concat(annotation_list)
        
        #Remove duplicates, Rohan said to choose karina over claire
        final_frame = []
        for name, group in input_data.groupby("image_path"):
            if len(group.annotator.unique())==2:
                final_frame.append(group[group.annotator == "Karina"])
            else:
                final_frame.append(group)
                
        final_frame = pd.concat(final_frame)
        final_frame.to_csv("/orange/ewhite/b.weinstein/Monash/annotations.csv")
        
        def cut(x):
            annotations = preprocess.split_raster(
                path_to_raster=x,
                annotations_file="/orange/ewhite/b.weinstein/Monash/annotations.csv",
                patch_size=1000,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            return annotations
        
        crop_annotations = []
        futures = client.map(cut,matched_tiles)
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
        
        df = pd.concat(crop_annotations)
        df.label = "Bird"
        
        train_annotations = df[~(df.image_path.str.contains("Transect_A_2020"))]
        train_annotations.to_csv(train_path, index=False)    
        
        test_annotations = df[df.image_path.str.contains("Transect_A_2020")]
        test_annotations.to_csv(test_path, index=False)
        client.close()

    return {"train":train_path, "test":test_path}

def prepare_USGS(generate=True):
    
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/USGS_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/USGS_test.csv"
    
    if generate:
        
        client = start_cluster.start(cpus=30, mem_size="10GB")

        input_data = pd.read_csv("/orange/ewhite/b.weinstein/USGS/migbirds/migbirds2020_07_31.csv")
        input_data["image_path"] = input_data.file_basename
        input_data.to_csv("/orange/ewhite/b.weinstein/USGS/migbirds/annotations.csv")
        
        def cut(x):
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/USGS/migbirds/migbirds/{}".format(x),
                annotations_file="/orange/ewhite/b.weinstein/USGS/migbirds/annotations.csv",
                patch_size=1200,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            return annotations
        
        crop_annotations = []
        futures = client.map(cut,input_data.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
        
        df = pd.concat(crop_annotations)
        df.label = "Bird"
        train_images = df.image_path.sample(frac=0.75)
        train_annotations = df[df.image_path.isin(train_images)]
        train_annotations.to_csv(train_path, index=False)    
    
        test_annotations = df[~(df.image_path.isin(train_images))]
        test_annotations.to_csv(test_path, index=False)   
        client.close()
    
    return {"train":train_path, "test":test_path}

def view_training(paths,comet_logger, n=10):
    """For each site, grab three images and view annotations
    Args:
        n: number of images to load
    """
    m = BirdDetector(transforms=get_transform)
    
    with comet_logger.experiment.context_manager("view_training"):
        for site in paths:
            for split in ["train","test"]:
                if split == "train":
                    augment = True
                else:
                    augment = False
                try:
                    x = paths[site][split]
                    ds = m.load_dataset(csv_file=x, root_dir=os.path.dirname(x), shuffle=True, augment=augment)
                    for i in np.arange(10):
                        batch = next(iter(ds))
                        image_path, image, targets = batch
                        df = visualize.format_boxes(targets[0], scores=False)
                        image = np.moveaxis(image[0].numpy(),0,2)[:,:,::-1] * 255
                        image = visualize.plot_predictions(image, df)
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            cv2.imwrite("{}/{}".format(tmpdirname, image_path[0]),image )
                            comet_logger.experiment.log_image("{}/{}".format(tmpdirname, image_path[0]),image_scale=0.25)                
                except Exception as e:
                    print(e)
                    continue
def prepare():
    paths = {}
    paths["terns"] = prepare_terns(generate=False)
    paths["everglades"] = prepare_everglades()
    paths["penguins"] = prepare_penguin(generate=False)
    paths["palmyra"] = prepare_palmyra(generate=False)
    paths["pelicans"] = prepare_pelicans(generate=False)
    paths["murres"] = prepare_murres(generate=False)
    paths["schedl"] = prepare_schedl(generate=False)
    paths["pfeifer"] = prepare_pfeifer(generate=False)    
    paths["hayes"] = prepare_hayes(generate=False)
    paths["USGS"] = prepare_USGS(generate=False)
    paths["monash"] = prepare_monash(generate=False)
    return paths

def train(path_dict, config, train_sets = ["penguins","terns","everglades","palmyra"],test_sets=["everglades"], comet_logger=None, save_dir=None):
        
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.log_parameter("train_set",train_sets)
    comet_logger.experiment.log_parameter("test_set",test_sets)
    comet_logger.experiment.add_tag("Generalization")
    
    all_sets = []
    print("Train sets: {}".format(train_sets))
    for x in train_sets:
        
        try:
            df = pd.read_csv(path_dict[x]["train"])
            df_test = pd.read_csv(path_dict[x]["test"])
            
        except:
            raise ValueError("No training path supplied for {}".format(x))
        all_sets.append(df)
        all_sets.append(df_test)
    
    train_annotations = pd.concat(all_sets)
    
    #A couple illegal boxes, make slightly smaller
    train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
    train_annotations["xmax"] = train_annotations["xmax"].astype(float) - 3
    train_annotations["ymin"] = train_annotations["ymin"].astype(float)
    train_annotations["ymax"] = train_annotations["ymax"].astype(float) - 3
    
    train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
    train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
    
    train_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv")

    all_val_sets = []
    for x in test_sets:
        df = pd.read_csv(path_dict[x]["test"])
        all_val_sets.append(df)
    
    test_annotations = pd.concat(all_val_sets)
    
    test_annotations.to_csv("/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv")

    comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
    comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])
    
    train_df =  pd.read_csv("/orange/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
    label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
    pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
    
    model = BirdDetector(transforms = get_transform)
    
    #update backbone weights with new Retinanet head
    model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
    
    model.config = config

    model.config["train"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/training_annotations.csv"
    model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops"    
    #model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/test_annotations.csv"
    #model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops"
        
    model.create_trainer(logger=comet_logger, plugins=DDPPlugin(find_unused_parameters=False))
    comet_logger.experiment.log_parameters(model.config)
    
    model.trainer.fit(model)
    
    for x in test_sets:
        test_results = model.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops/", iou_threshold=0.25, savedir=savedir)
        if comet_logger is not None:
            try:
                test_results["results"].to_csv("{}/iou_dataframe.csv".format(savedir))
                comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(savedir))
                
                test_results["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
                comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))
                
                for index, row in test_results["class_recall"].iterrows():
                    comet_logger.experiment.log_metric("{}_Recall".format(row["label"]),row["recall"])
                    comet_logger.experiment.log_metric("{}_Precision".format(row["label"]),row["precision"])
                
                comet_logger.experiment.log_metric("Average Class Recall",test_results["class_recall"].recall.mean())
                comet_logger.experiment.log_metric("{} Box Recall".format(x),test_results["box_recall"])
                comet_logger.experiment.log_metric("{} Box Precision".format(x),test_results["box_precision"])
            except Exception as e:
                print(e)    
    if save_dir:
        try:
            model.trainer.save_checkpoint("{}/{}.pl".format(save_dir,"_".join(train_sets)))
        except Exception as e:
            print(e)        
            
    #delete model and free up memory
    del model
    torch.cuda.empty_cache()
    
    #The last position in the loop is the LOO score
    return test_results["box_recall"], test_results["box_precision"]

if __name__ =="__main__":
    #save original config during loop
    #comet_logger=None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir="/orange/ewhite/b.weinstein/generalization/"
    savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(savedir)
    except Exception as e:
        print(e)
        
    model = BirdDetector(transforms=get_transform)
    config = model.config
    
    path_dict = prepare()
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")
    
    view_training(path_dict, comet_logger=comet_logger)
    ###leave one out
    train_list = ["USGS","terns","palmyra","penguins","pfeifer","hayes"]
    results = []
    for x in train_list:
        train_sets = [y for y in train_list if not y==x]
        train_sets.append("everglades")
        test_sets = [x]
        recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)
        torch.cuda.empty_cache()
        gc.collect()
        result = pd.DataFrame({"test_sets":[x],"recall":[recall],"precision":[precision]})
        results.append(result)
    
    results = pd.concat(results)
    results.to_csv("Figures/generalization.csv")
    comet_logger.experiment.log_asset(file_data="Figures/generalization.csv", file_name="results.csv")
    comet_logger.experiment.log_metric(name="Mean LOO Recall", value=results.recall.mean())
    comet_logger.experiment.log_metric(name="Mean LOO Precision", value=results.precision.mean())
    
    #Joint model
    train_sets = ["terns","palmyra","penguins","pfeifer","hayes","everglades","USGS"]
    test_sets = ["murres","pelicans","schedl"]
    recall, precision = train(path_dict=path_dict, config=config, train_sets=train_sets, test_sets=test_sets, comet_logger=comet_logger, save_dir=savedir)
    #Don't log validation scores till the end of project
    
    #log images
    with comet_logger.experiment.context_manager("validation"):
        images = glob.glob("{}/*.png".format(savedir))
        for img in images:
            comet_logger.experiment.log_image(img, image_scale=0.25)    
