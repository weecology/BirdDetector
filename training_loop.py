"""Script to take the trained everglades model and predict the Palmyra data"""
#srun -p gpu --gpus=2 --mem 70GB --time 5:00:00 --pty -u bash -i
# conda activate Zooniverse_pytorch
import comet_ml
from deepforest import main
import pandas as pd
import gc
from pytorch_lightning.loggers import CometLogger
from deepforest import preprocess
from generalization import shapefile_to_annotations
import rasterio as rio
import numpy as np
import os
import shutil
from datetime import datetime
import torch
 
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
        patch_size=1300,
        patch_overlap=0.05,
        base_dir="/orange/ewhite/b.weinstein/generalization/crops/",
        image_name="CooperEelPond_53M.tif",
        allow_empty=False
    )
    
    previous_train = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/palmyra_train.csv")
    train_annotations = pd.concat([train_annotations, previous_train])
    train_annotations.to_csv(train_path,index=False)
        
    return {"train":train_path, "test":test_path}
    
def training(proportion,pretrained=True, comet_logger=None):

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir="/orange/ewhite/everglades/Palmyra/"
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
    
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.log_parameter("proportion",proportion)
    comet_logger.experiment.log_parameter("pretrained", pretrained)
    
    comet_logger.experiment.add_tag("Palmyra")
    train_annotations = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/palmyra_finetune.csv")
    crops = train_annotations.image_path.unique()    
    
    if not proportion == 0:
        if proportion < 1:  
            selected_crops = np.random.choice(crops, size = int(proportion*len(crops)),replace=False)
            train_annotations = train_annotations[train_annotations.image_path.isin(selected_crops)]
    
    train_annotations.to_csv("crops/loop_training_annotations.csv", index=False)
    
    comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
    comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])
        
    if pretrained:
        model = main.deepforest.load_from_checkpoint("/orange/ewhite/b.weinstein/generalization/20210531_231758/terns_palmyra_penguins_pfeifer_everglades.pl")
        model.label_dict = {"Bird":0}
        
    else:
        model = main.deepforest(label_dict={"Bird":0})
    try:
        os.mkdir("/orange/ewhite/everglades/Palmyra/{}/".format(proportion))
    except:
        pass
    
    model.config["train"]["csv_file"] = "crops/loop_training_annotations.csv"
    model.config["train"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"    
    model.config["validation"]["csv_file"] = "/orange/ewhite/b.weinstein/generalization/crops/palmyra_test.csv"
    model.config["validation"]["root_dir"] = "/orange/ewhite/b.weinstein/generalization/crops/"
    
    model.create_trainer(logger=comet_logger)
    comet_logger.experiment.log_parameters(model.config)
    
    if not proportion == 0:
        model.trainer.fit(model)
    
    test_results = model.evaluate(csv_file=model.config["validation"]["csv_file"], root_dir=model.config["validation"]["root_dir"], iou_threshold=0.25)
    
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
    #model.predict_file(csv_file = model.config["validation"]["csv_file"], root_dir = model.config["validation"]["root_dir"], savedir=model_savedir)
    #images = glob.glob("{}/*.png".format(model_savedir))
    #for img in images:
        #comet_logger.experiment.log_image(img, image_scale=0.2)
     
    formatted_results = pd.DataFrame({"proportion":[proportion], "pretrained": [pretrained], "annotations": [train_annotations.shape[0]],"precision": [precision],"recall": [recall]})
    
    #free up
    #model.trainer.save_checkpoint("{}/Palmyra_proportion_{}_pretrained_{}.pl".format(model_savedir,proportion, pretrained))
    
    del model
    torch.cuda.empty_cache()
    
    return formatted_results

def run(generate=False, pretrained=True):
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                project_name="everglades", workspace="bw4sz",auto_output_logging = "simple") 
    
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
        
        prepare_palmyra(generate=generate)
        
    training(proportion=1, pretrained=pretrained, comet_logger=comet_logger)
    result_df = []
    for y in range(5):  
        print("Iteration {}".format(y))
        for x in np.arange(0.25, 1.25, 0.25):
            results = training(proportion=x, pretrained=pretrained, comet_logger=comet_logger)
            result_df.append(results)
    result_df = pd.concat(result_df)
    result_df.to_csv("Figures/Palmyra_results_pretrained_{}.csv".format(pretrained))
    
if __name__ == "__main__":   
    run(pretrained=False, generate=False)
    run(pretrained=True, generate=False)
