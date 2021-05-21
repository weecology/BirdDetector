"""Script to take the trained everglades model and predict the Palmyra data"""
#srun -p gpu --gpus=2 --mem 70GB --time 5:00:00 --pty -u bash -i
# conda activate Zooniverse_pytorch
import pandas as pd
import comet_ml
import gc
import start_cluster
from distributed import wait
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest import preprocess
import glob
from shapely.geometry import Point, box
import geopandas as gpd

import rasterio as rio
import numpy as np
import os
import shutil
from datetime import datetime
from matplotlib import pyplot as plt
import random
import torch

def shapefile_to_annotations(shapefile, rgb, box_points=True, savedir=".", confidence_filter=True):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        results: a pandas dataframe
    """
    #Read shapefile
    gdf = gpd.read_file(shapefile)
    
    #confidence levels
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
    gdf["geometry"] = [box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.15).bounds.values]
        
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
    
    #Add labels is they exist
    if "label" in gdf.columns:
        df["label"] = gdf["label"]
    else:
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
    train_path = "/orange/ewhite/b.weinstein/generalization/crops/palmyra_finetune.csv"      
    
    src = rio.open("/orange/ewhite/everglades/Palmyra/CooperEelPond_53M.tif")
    numpy_image = src.read()
    numpy_image = np.moveaxis(numpy_image,0,2)
    training_image = numpy_image[:,:,:3].astype("uint8")
    
    df = shapefile_to_annotations(shapefile="/orange/ewhite/everglades/Palmyra/CooperEelPond_53m_annotation.shp", 
                                  rgb="/orange/ewhite/everglades/Palmyra/CooperEelPond_53M.tif", buffer_size=0.15)

    df.to_csv("Figures/training_annotations.csv",index=False)        
    train_annotations = preprocess.split_raster(
        numpy_image=training_image,
        annotations_file="Figures/training_annotations.csv",
        patch_size=1500,
        patch_overlap=0.05,
        base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
        image_name="CooperEelPond_53M.tif",
        allow_empty=False
    )
    
    train_annotations.to_csv(train_path,index=False)
        
    return {"train":train_path, "test":test_path}
    
def training(proportion, epochs=20, patch_size=2000,pretrained=True, iteration=None):

    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="everglades", workspace="bw4sz")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir="/orange/ewhite/everglades/Palmyra/"
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
    
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.log_parameter("proportion",proportion)
    comet_logger.experiment.log_parameter("patch_size",patch_size)
    comet_logger.experiment.log_parameter("pretrained", pretrained)
    
    comet_logger.experiment.add_tag("Palmyra")
    
    train_annotations = pd.read_csv("crops/full_training_annotations.csv")
    crops = train_annotations.image_path.unique()    
    
    if not proportion == 0:
        if proportion < 1:  
            selected_crops = np.random.choice(crops, size = int(proportion*len(crops)),replace=False)
            train_annotations = train_annotations[train_annotations.image_path.isin(selected_crops)]
    
    train_annotations.to_csv("crops/training_annotations.csv", index=False)
    
    comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
    comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])
        
    if pretrained:
        model = main.deepforest.load_from_checkpoint("/orange/ewhite/everglades/Zooniverse/predictions/20210414_225905/species_model.pl")
        model.label_dict = {"Bird":0}
        
    else:
        model = main.deepforest(label_dict={"Bird":0})
    try:
        os.mkdir("/orange/ewhite/everglades/Palmyra/{}/".format(proportion))
    except:
        pass
    
    model.config["train"]["epochs"] = epochs
    model.config["train"]["csv_file"] = "crops/training_annotations.csv"
    model.config["train"]["root_dir"] = "crops"    
    model.config["validation"]["csv_file"] = "crops/test_annotations.csv"
    model.config["validation"]["root_dir"] = "crops"
    
    model.create_trainer(logger=comet_logger)
    comet_logger.experiment.log_parameters(model.config)
    
    if not proportion == 0:
        model.trainer.fit(model)
    
    test_results = model.evaluate(csv_file="crops/test_annotations.csv", root_dir="crops/", iou_threshold=0.25)
    
    if comet_logger is not None:
        try:
            test_results["results"].to_csv("{}/iou_dataframe.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(model_savedir))
            
            test_results["class_recall"].to_csv("{}/class_recall.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/class_recall.csv".format(model_savedir))
            
            for index, row in test_results["class_recall"].iterrows():
                comet_logger.experiment.log_metric("{}_Recall".format(row["label"]),row["recall"])
                comet_logger.experiment.log_metric("{}_Precision".format(row["label"]),row["precision"])
            
            comet_logger.experiment.log_metric("Average Class Recall",test_results["class_recall"].recall.mean())
            comet_logger.experiment.log_metric("Box Recall",test_results["box_recall"])
            comet_logger.experiment.log_metric("Box Precision",test_results["box_precision"])
        except Exception as e:
            print(e)
    
    recall = test_results["box_recall"]
    precision = test_results["box_precision"]
    
    print("Recall is {}".format(recall))
    print("Precision is {}".format(precision))
    
    comet_logger.experiment.log_metric("precision",precision)
    comet_logger.experiment.log_metric("recall", recall)
    
    #log images
    model.predict_file(csv_file = model.config["validation"]["csv_file"], root_dir = model.config["validation"]["root_dir"], savedir=model_savedir)
    images = glob.glob("{}/*.png".format(model_savedir))
    for img in images:
        comet_logger.experiment.log_image(img)
    
    #log training images
    #model.predict_file(csv_file = model.config["train"]["csv_file"], root_dir = model.config["train"]["root_dir"], savedir=model_savedir)
    #images = glob.glob("{}/*.png".format(model_savedir))
    #random.shuffle(images)
    #for img in images[:20]:
        #comet_logger.experiment.log_image(img)  
        
    comet_logger.experiment.end()

    formatted_results = pd.DataFrame({"proportion":[proportion], "pretrained": [pretrained], "annotations": [train_annotations.shape[0]],"precision": [precision],"recall": [recall], "iteration":[iteration]})
    
    #free up
    del model
    torch.cuda.empty_cache()
    
    return formatted_results

def run(patch_size=2500, generate=False, client=None, epochs=10, ratio=2, pretrained=True):
    if generate:
        folder = 'crops/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
        prepare_test(patch_size=patch_size)
        prepare_train(patch_size=int(patch_size * ratio))
    
    iteration_result = []
    futures = []    
    print("ratio is {}".format(ratio))
    result_df = training(proportion=1, epochs=epochs, patch_size=patch_size, pretrained=pretrained)
    result_df = training(proportion=1,epochs=epochs,patch_size=patch_size, pretrained=pretrained)
    iteration_result.append(result_df)
        
    #future = client.submit(training, pretrained=True, patch_size=patch_size, proportion=0)
    #futures.append(future)
    #future = client.submit(training, pretrained=False, patch_size=patch_size, proportion=0)
    #futures.append(future)
    
    #iteration = 0
    #while iteration < 2:
        #for x in [1]:
            #for y in [True, False]: 
                #if client is not None:
                    #future = client.submit(training,proportion=x, patch_size=patch_size, pretrained=y, iteration = iteration)
                    #futures.append(future)
                #else:
                    #experiment_result = training(proportion=x, patch_size=patch_size, pretrained=y, iteration = iteration)
                    #iteration_result.append(experiment_result)
        #iteration+=1
                    
    #if client is not None:
    #    wait(futures)
    #    for future in futures:
    #        iteration_result.append(future.result())

    #results = pd.concat(iteration_result)
    #results.to_csv("Figures/Palmyra_results_{}.csv".format(patch_size)) 

if __name__ == "__main__":
    for x in [1000, 1500]:
        run(patch_size=x, epochs=1, ratio=0.75, pretrained=False, generate=True)
        run(patch_size=x, epochs=1, ratio=0.75, pretrained=True, generate=False)
        run(patch_size=x, epochs=30, ratio=0.75, pretrained=True, generate=False)
        run(patch_size=x, epochs=30, ratio=0.75, pretrained=True, generate=False)
