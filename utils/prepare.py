#Prepare all training sets
import cv2
from model import BirdDetector
import glob
from deepforest import preprocess
from deepforest import visualize
from augmentation import get_transform
from utils import start_cluster
from utils.preprocess import *

import pandas as pd
import rasterio as rio
import numpy as np
import os
import PIL
import tempfile

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
                                          rgb="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename))
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
                                          rgb="/orange/ewhite/b.weinstein/mckellar/{}.tif".format(basename))
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
    
    gdf = gpd.read_file("/orange/ewhite/b.weinstein/seabirdwatch/annotations.shp")
    
    test_gdf = gdf[gdf.colonyname == "KIPPa"]
    train_gdf = gdf[~(gdf.colonyname == "KIPPa")]
    
    train_annotations = []
    test_annotations = []
    if generate:   
        for name, group in train_gdf.groupby("image_path"):
            basename = os.path.splitext(os.path.basename(name))[0]
            group.to_file("/orange/ewhite/b.weinstein/seabirdwatch/{}.shp".format(basename))
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/seabirdwatch/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/seabirdwatch/{}.tif".format(basename))
            train_annotations.append(df)
        
        train_annotations = pd.concat(train_annotations)
        train_annotations.to_csv(train_path)
         
        for name, group in test_gdf.groupby.image_path():
            basename = os.path.splitext(os.path.basename(name))[0]
            group.to_file("/orange/ewhite/b.weinstein/seabirdwatch/{}.shp".format(basename))
            df = shapefile_to_annotations(shapefile="/orange/ewhite/b.weinstein/seabirdwatch/{}.shp".format(basename),
                                          rgb="/orange/ewhite/b.weinstein/seabirdwatch/{}.tif".format(basename))
            test_annotations.append(df)
        
        test_annotations = pd.concat(test_annotations)
        test_annotations.to_csv(test_path)
        
        test_annotations.to_csv(train_path, index=False)
        
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
    paths["mckellar"] = prepare_mckellar(generate=False)
    paths["seabirdwatch"] = prepare_seabirdwatch(generate=True)
    
    return paths
