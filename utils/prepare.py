#Prepare all training sets
import glob
from deepforest import preprocess
from utils import start_cluster
from utils.preprocess import *

import pandas as pd
import random
import rasterio as rio
import numpy as np
import os
import PIL
import tempfile
import cv2

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
                                                   patch_size=1300, patch_overlap=0.05, base_dir="/orange/ewhite/b.weinstein/generalization/crops/", image_name="Dudley_projected.tif")
        
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
            patch_size=1300,
            patch_overlap=0.05,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
            image_name="CooperStrawn_53m_tile_clip_projected.tif",
            allow_empty=False
        )
        
        src = rio.open("/orange/ewhite/everglades/Palmyra/CooperEelPond_53M.tif")
        numpy_image = src.read()
        numpy_image = np.moveaxis(numpy_image,0,2)
        training_image = numpy_image[:,:,:3].astype("uint8")
        
        df = shapefile_to_annotations(shapefile="/orange/ewhite/everglades/Palmyra/CooperEelPond_53m_annotation.shp", 
                                      rgb="/orange/ewhite/everglades/Palmyra/CooperEelPond_53M.tif", buffer_size=0.15)
    
        df.to_csv("Figures/training_annotations.csv",index=False)        
        train_annotations2 = preprocess.split_raster(
            numpy_image=training_image,
            annotations_file="Figures/training_annotations.csv",
            patch_size=1300,
            patch_overlap=0.05,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
            image_name="CooperEelPond_53M.tif",
            allow_empty=False
        )

        
        train_annotations = pd.concat([train_annotations_1, train_annotations2])
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
    
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/terns_test.csv"
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
        
        train_annotations = check_shape(train_annotations)
        train_images = train_annotations.image_path.sample(n=500)
        train_annotations = train_annotations[train_annotations.image_path.isin(train_images)]
        train_annotations.to_csv(train_path, index=False)
        
        hayes_albatross_val.label="Bird"
        hayes_albatross_val_images = hayes_albatross_val.image_path.sample(n=100)
        hayes_albatross_val = hayes_albatross_val[hayes_albatross_val.image_path.isin(hayes_albatross_val_images)]
        hayes_albatross_val = check_shape(hayes_albatross_val)                
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
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/murres_train.csv"
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
        annotations.to_csv(train_path,index=False)    
    
    return {"train":train_path}

        
def prepare_valle(generate=True):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/valle_train.csv"
    if generate:   
        df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/valle/terns_italy.shp",
                                      rgb="/orange/ewhite/b.weinstein/valle/terns_italy.png", buffer_size=15)
        df.to_csv("/orange/ewhite/b.weinstein/valle/terns_italy.csv")
        
        annotations = preprocess.split_raster(
            path_to_raster="/orange/ewhite/b.weinstein/valle/terns_italy.png",
            annotations_file="/orange/ewhite/b.weinstein/valle/terns_italy.csv",
            patch_size=800,
            patch_overlap=0,
            base_dir="/orange/ewhite/b.weinstein/generalization/crops",
            allow_empty=False
        )
        
        #Test        
        annotations.to_csv(train_path,index=False)    
    
    return {"train":train_path}

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
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/schedl_train.csv"
    
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
        test_annotations.to_csv(train_path)
        
    return {"test":train_path}

def prepare_monash(generate=True):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/monash_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/monash_test.csv"
        
    if generate:
        client = start_cluster.start(cpus=30, mem_size="10GB")
  
        #Find all the shapefiles
        shps = glob.glob("/orange/ewhite/b.weinstein/Monash/**/*.shp",recursive=True)
        shps = [x for x in shps if not "AQUA" in x]
        
        annotation_list = []
        matched_tiles = []
        for x in shps:
            same_path = "{}.tif".format(os.path.splitext(x)[0])
            components = os.path.basename(x).split("_")
            tif_path = "/orange/ewhite/b.weinstein/Monash/Transect {letter}/Transect {letter} {year}/Transect_{letter}_{year}.tif".format(letter=components[1],year=components[2])
            jpg_path = "/orange/ewhite/b.weinstein/Monash/Transect {letter}/Transect {letter} {year}/Transect_{letter}_{year}.jpg".format(letter=components[1],year=components[2])
            
            if os.path.exists(same_path):
                rgb_path = same_path
            elif os.path.exists(tif_path):
                rgb_path = tif_path
            elif os.path.exists(jpg_path):
                rgb_path = jpg_path
            else:
                print("Cannot find corresponding image to annotations {}".format(x))
                continue
                
            annotations = shapefile_to_annotations(shapefile=x, rgb=rgb_path, buffer_size=0.15)
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
        df = df.drop_duplicates()
        
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
                patch_size=1100,
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
        df = df[~(df.xmin >= df.xmax)]
        df = df[~(df.ymin >= df.ymax)]
        
        #pad the edges by a few pixels
        df = check_shape(df)
                
        train_images = df.image_path.sample(frac=0.85)
        train_annotations = df[df.image_path.isin(train_images)]
        train_annotations.to_csv(train_path, index=False)    
    
        test_annotations = df[~(df.image_path.isin(train_images))]
        test_annotations.to_csv(test_path, index=False)   
        client.close()
    
    return {"train":train_path, "test":test_path}

def prepare_mckellar(generate=True):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/mckellar_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/mckellar_test.csv"
    
    train_annotations = []
    test_annotations = []
    if generate:   
        for x in glob.glob("/orange/ewhite/b.weinstein/mckellar/*.shp")[:1]:
            basename = os.path.splitext(os.path.basename(x))[0]
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/mckellar/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename), buffer_size=0.15)
            df.to_csv("/orange/ewhite/b.weinstein/mckellar/{}.csv".format(basename))
            
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename),
                annotations_file="/orange/ewhite/b.weinstein/mckellar/{}.csv".format(basename),
                patch_size=700,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            test_annotations.append(annotations)
        test_annotations = pd.concat(test_annotations)
        test_annotations.to_csv(test_path, index=False)
            
        for x in glob.glob("/orange/ewhite/b.weinstein/mckellar/*.shp")[1:]:
            print(x)
            basename = os.path.splitext(os.path.basename(x))[0]
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/mckellar/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename), buffer_size=0.15)
            df.to_csv("/orange/ewhite/b.weinstein/mckellar/{}.csv".format(basename))
            
            annotations = preprocess.split_raster(
                path_to_raster="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename),
                annotations_file="/orange/ewhite/b.weinstein/mckellar/{}.csv".format(basename),
                patch_size=700,
                patch_overlap=0,
                base_dir="/orange/ewhite/b.weinstein/generalization/crops",
                allow_empty=False
            )
            
            train_annotations.append(annotations)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations.to_csv(train_path, index=False)
        
    return {"train":train_path, "test":test_path}

def prepare_seabirdwatch(generate):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/seabirdwatch_test.csv"
    
    if generate:   
        
        client = start_cluster.start(cpus=20)
        shps = glob.glob("/orange/ewhite/b.weinstein/seabirdwatch/parsed/*.shp")
        
        #Hold one year out
        test_shps = [x for x in shps if "KIPPa" in x]
        train_shps = [x for x in shps if not "KIPPa" in x]
        
        train_annotations = []
        test_annotations = []

        for x in train_shps:
            annotations = gpd.read_file(x)
            df = annotations.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]
            df["image_path"] = "{}.JPG".format(os.path.splitext(os.path.basename(x))[0])            
            train_annotations.append(df)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations = check_shape(train_annotations)
        train_annotations.to_csv("/orange/ewhite/b.weinstein/seabirdwatch/train_images.csv")
        
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/seabirdwatch/train_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/generalization/crops/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=500)
            return result

        crop_annotations = []
        futures = client.map(cut, train_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                        
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(train_path)        
         
        for x in test_shps:
            annotations = gpd.read_file(x)
            df = annotations.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]            
            df["image_path"] = "{}.JPG".format(os.path.splitext(os.path.basename(x))[0])            
            test_annotations.append(df)
            
        test_annotations = pd.concat(test_annotations)
        test_annotations = check_shape(test_annotations)
        test_annotations.to_csv("/orange/ewhite/b.weinstein/seabirdwatch/test_images.csv")
                
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/seabirdwatch/test_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/generalization/crops/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=500)
            return result
        
        crop_annotations = []
        futures = client.map(cut, test_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                        
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(test_path)
        client.close()
        
    return {"train":train_path, "test":test_path}

def prepare_newmexico(generate):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/newmexico_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/newmexico_test.csv"
    
    if generate:   
        client = start_cluster.start(cpus=5)
        gdf = gpd.read_file("/orange/ewhite/b.weinstein/newmexico/annotations.shp")
        
        train_gdf = gdf[~(gdf.image_path == "BDA_18A4_20181107_1.JPG")]
        test_gdf = gdf[gdf.image_path == "BDA_18A4_20181107_1.JPG"]
        
        train_annotations = []
        test_annotations = []
        
        for name, group in train_gdf.groupby("image_path"):
            df = group.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]            
            df["image_path"] = name        
            train_annotations.append(df)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations.to_csv("/orange/ewhite/b.weinstein/newmexico/train_images.csv")
        
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/newmexico/train_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/newmexico/Imagery/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=800)
            return result

        #Split into crops
        crop_annotations = []
        futures = client.map(cut, train_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                        
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(train_path)

        for name, group in test_gdf.groupby("image_path"):
            df = group.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]            
            df["image_path"] = name
            test_annotations.append(df)
        
        test_annotations = pd.concat(test_annotations)        
        test_annotations.to_csv("/orange/ewhite/b.weinstein/newmexico/test_images.csv")
        
        #Too large for GPU memory, cut into pieces
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/newmexico/test_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/newmexico/Imagery/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=800)
            return result

        #Split into crops
        crop_annotations = []
        futures = client.map(cut, test_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                                
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(test_path)
        client.close()
    return {"train":train_path, "test":test_path}

def prepare_neill(generate):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/neill_train.csv"
    test_path = "/orange/ewhite/b.weinstein/generalization/crops/neill_test.csv"
    
    if generate:   
        client = start_cluster.start(cpus=30)
        shps = glob.glob("/orange/ewhite/b.weinstein/neill/parsed/*.shp")
        
        #Hold one year out
        test_shps = [x for x in shps if "2017" in x]
        train_shps = [x for x in shps if not "2017" in x]
        
        train_annotations = []
        test_annotations = []
        
        for x in train_shps:
            annotations = gpd.read_file(x)
            df = annotations.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]            
            df["image_path"] = "{}.JPG".format(os.path.splitext(os.path.basename(x))[0])            
            train_annotations.append(df)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations = check_shape(train_annotations)
        train_annotations.to_csv("/orange/ewhite/b.weinstein/neill/train_images.csv")
        
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/neill/train_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/generalization/crops/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=700)
            return result

        #Split into crops
        crop_annotations = []
        futures = client.map(cut, train_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                        
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(train_path)

        for x in test_shps:
            annotations = gpd.read_file(x)
            df = annotations.geometry.bounds
            df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})    
            df["label"] = "Bird"
            df = df[~(df.xmin >= df.xmax)]
            df = df[~(df.ymin >= df.ymax)]            
            df["image_path"] = "{}.JPG".format(os.path.splitext(os.path.basename(x))[0])
            test_annotations.append(df)
        
        test_annotations = pd.concat(test_annotations)        
        test_annotations = check_shape(test_annotations)
        test_annotations.to_csv("/orange/ewhite/b.weinstein/neill/test_images.csv")
        
        #Too large for GPU memory, cut into pieces
        def cut(x):
            result = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/neill/test_images.csv",
                                                path_to_raster="/orange/ewhite/b.weinstein/generalization/crops/{}".format(x),
                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                allow_empty=False,
                                                patch_size=700)
            return result

        #Split into crops
        crop_annotations = []
        futures = client.map(cut, test_annotations.image_path.unique())
        for x in futures:
            try:
                crop_annotations.append(x.result())
            except Exception as e:
                print(e)
                pass
                                
        crop_annotations = pd.concat(crop_annotations)
        crop_annotations.to_csv(test_path)
        client.close()
    return {"train":train_path, "test":test_path}

def prepare_cros(generate):
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/cros_train.csv"
    
    if generate:   
        txts = glob.glob("/orange/ewhite/b.weinstein/cros/*.txt")
        train_annotations = []
        for x in txts:
            df = pd.read_csv(x, delim_whitespace=True, names=["xmin","ymin","width","height"])
            basename = os.path.splitext(os.path.basename(x))[0]
            df["image_path"] = "{}.jpg".format(basename)
            img = cv2.imread("/orange/ewhite/b.weinstein/cros/{}.jpg".format(basename))
            height, width, channels = img.shape
            df["xmax"]  = df["xmin"] * width + (df["width"] * width)
            df["ymax"]  = df["ymin"] * height + (df["height"] * height)
            df["xmin"]  = df["xmin"] * width 
            df["ymin"]  = df["ymin"] * height         
            df["label"] = "Bird"
            df = df[["xmin","ymin","xmax","ymax","label","image_path"]]
            df.to_csv("/orange/ewhite/b.weinstein/cros/{}.csv".format(basename))
            cropped_df = preprocess.split_raster(annotations_file="/orange/ewhite/b.weinstein/cros/{}.csv".format(basename),
                                                                path_to_raster="/orange/ewhite/b.weinstein/cros/{}.jpg".format(basename),
                                                                base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
                                                                allow_empty=False,
                                                                patch_size=700)
            train_annotations.append(cropped_df)
            
        train_annotations = pd.concat(train_annotations)
        train_annotations.to_csv(train_path)
        
    return {"train":train_path}
        
def prepare():
    paths = {}
    paths["terns"] = prepare_terns(generate=False)
    paths["everglades"] = prepare_everglades()
    paths["penguins"] = prepare_penguin(generate=False)
    paths["palmyra"] = prepare_palmyra(generate=False)
    paths["neill"] = prepare_pelicans(generate=False)
    paths["murres"] = prepare_murres(generate=False)
    paths["schedl"] = prepare_schedl(generate=False)
    paths["pfeifer"] = prepare_pfeifer(generate=False)    
    paths["hayes"] = prepare_hayes(generate=False)
    paths["USGS"] = prepare_USGS(generate=False)
    paths["monash"] = prepare_monash(generate=False)
    paths["mckellar"] = prepare_mckellar(generate=False)
    paths["seabirdwatch"] = prepare_seabirdwatch(generate=False)
    paths["neill"] = prepare_neill(generate=False)
    paths["newmexico"] = prepare_newmexico(generate=False)
    paths["valle"] = prepare_valle(generate=False)
    #paths["cros"] = prepare_cros(generate=True)
    
    return paths
