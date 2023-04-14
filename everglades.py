#DeepForest bird detection from extracted Zooniverse predictions
import comet_ml
import pandas as pd
from pytorch_lightning.loggers import CometLogger
from deepforest.callbacks import images_callback
from deepforest import visualize
from deepforest import main
import traceback
import os
import numpy as np
import torch
from datetime import datetime

def is_empty(precision_curve, threshold):
    precision_curve.score = precision_curve.score.astype(float)
    precision_curve = precision_curve[precision_curve.score > threshold]
    
    return precision_curve.empty

def empty_image(precision_curve, threshold):
    empty_true_positives = 0
    empty_false_negatives = 0
    for name, group in precision_curve.groupby('image'): 
        if is_empty(group, threshold):
            empty_true_positives +=1
        else:
            empty_false_negatives+=1
    empty_recall = empty_true_positives/float(empty_true_positives + empty_false_negatives)
    
    return empty_recall

def plot_recall_curve(precision_curve, invert=False):
    """Plot recall at fixed interval 0:1"""
    recalls = {}
    for i in np.linspace(0,1,11):
        recalls[i] = empty_image(precision_curve=precision_curve, threshold=i)
    
    recalls = pd.DataFrame(list(recalls.items()), columns=["threshold","recall"])
    
    if invert:
        recalls["recall"] = 1 - recalls["recall"].astype(float)
    
    ax1 = recalls.plot.scatter("threshold","recall")
    
    return ax1
    
def predict_empty_frames(model, empty_images, comet_experiment, invert=False):
    """Optionally read a set of empty frames and predict
        Args:
            invert: whether the recall should be relative to empty images (default) or non-empty images (1-value)"""
    
    #Create PR curve
    precision_curve = [ ]
    for path in empty_images:
        boxes = model.predict_image(path=path, return_plot=False)
        if boxes is not None:    
            boxes["image"] = path
            precision_curve.append(boxes)
    
    #if no boxes, skip plot
    try:
        precision_curve = pd.concat(precision_curve)
    except:
        return None
    
    recall_plot = plot_recall_curve(precision_curve, invert=invert)
    value = empty_image(precision_curve, threshold=0.4)
    
    if invert:
        value = 1 - value
        metric_name = "BirdRecall_at_0.4"
        recall_plot.set_title("Atleast One Bird Recall")
    else:
        metric_name = "EmptyRecall_at_0.4"
        recall_plot.set_title("Empty Recall")        
        
    comet_experiment.experiment.log_metric(metric_name,value)
    comet_experiment.experiment.log_figure(recall_plot)   
    
def train_model(train_path, test_path, empty_images_path=None, save_dir=".", debug = False):
    """Train a DeepForest model"""
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="everglades", workspace="bw4sz")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
    
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.add_tag("Bird Detector")
    
    #Log the number of training and test
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    #Set config and train'    
    label_dict = {key:value for value, key in enumerate(train.label.unique())}
    model = main.deepforest(num_classes=len(train.label.unique()),label_dict=label_dict)
    
    model.config["train"]["csv_file"] = train_path
    model.config["train"]["root_dir"] = os.path.dirname(train_path)
    
    #Set config and train
    model.config["validation"]["csv_file"] = test_path
    model.config["validation"]["root_dir"] = os.path.dirname(test_path)
    
    if debug:
        print("DEBUG")
        model.config["train"]["fast_dev_run"] = False
        model.config["gpus"] = None
        model.config["workers"] = 0
        model.config["batch_size"] = 1    
    if comet_logger is not None:
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_parameter("Training_Annotations",train.shape[0])    
        comet_logger.experiment.log_parameter("Testing_Annotations",test.shape[0])
        
    #im_callback = images_callback(csv_file=model.config["validation"]["csv_file"], root_dir=model.config["validation"]["root_dir"], savedir=model_savedir, n=20)    
    model.create_trainer(logger=comet_logger)
    
    model.trainer.fit(model)
    
    #Manually convert model
    results = model.evaluate(test_path, root_dir = os.path.dirname(test_path))
    
    if comet_logger is not None:
        try:
            results["results"].to_csv("{}/iou_dataframe.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(model_savedir))
            
            results["class_recall"].to_csv("{}/class_recall.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/class_recall.csv".format(model_savedir))
            
            for index, row in results["class_recall"].iterrows():
                comet_logger.experiment.log_metric("{}_Recall".format(row["label"]),row["recall"])
                comet_logger.experiment.log_metric("{}_Precision".format(row["label"]),row["precision"])
            
            comet_logger.experiment.log_metric("Average Class Recall",results["class_recall"].recall.mean())
            comet_logger.experiment.log_metric("Box Recall",results["box_recall"])
            comet_logger.experiment.log_metric("Box Precision",results["box_precision"])
            
            comet_logger.experiment.log_parameter("saved_checkpoint","{}/species_model.pl".format(model_savedir))
            
            ypred = results["results"].predicted_label.astype('category').cat.codes.to_numpy()            
            ypred = torch.from_numpy(ypred)
            ypred = torch.nn.functional.one_hot(ypred, num_classes = model.num_classes).numpy()
            
            ytrue = results["results"].true_label.astype('category').cat.codes.to_numpy()
            ytrue = torch.from_numpy(ytrue)
            ytrue = torch.nn.functional.one_hot(ytrue, num_classes = model.num_classes).numpy()
            comet_logger.experiment.log_confusion_matrix(y_true=ytrue, y_predicted=ypred, labels = list(model.label_dict.keys()))
        except Exception as e:
            print("logger exception: {} with traceback \n {}".format(e, traceback.print_exc()))
    
    #Create a positive bird recall curve
    test_frame_df = pd.read_csv(test_path)
    dirname = os.path.dirname(test_path)
    test_frame_df["image_path"] = test_frame_df["image_path"].apply(lambda x: os.path.join(dirname,x))
    empty_images = test_frame_df.image_path.unique()    
    predict_empty_frames(model, empty_images, comet_logger, invert=True)
    
    #Test on empy frames
    if empty_images_path:
        empty_frame_df = pd.read_csv(empty_images_path)
        empty_frame_df["image_path"] = empty_frame_df["image_path"].apply(lambda x: os.path.join(dirname,x))        
        empty_images = empty_frame_df.image_path.unique()
        predict_empty_frames(model, empty_images, comet_logger)
    
    #save model
    model.trainer.save_checkpoint("{}/bird_detector.pl".format(model_savedir))
    
    #Save a full set of predictions to file.
    boxes = model.predict_file(model.config["validation"]["csv_file"], root_dir=model.config["validation"]["root_dir"])
    visualize.plot_prediction_dataframe(df = boxes, savedir = model_savedir, root_dir=model.config["validation"]["root_dir"])
    
    return model

if __name__ == "__main__":
    model = train_model(train_path="/orange/ewhite/everglades/Zooniverse/parsed_images/train.csv",
                        test_path="/orange/ewhite/everglades/Zooniverse/parsed_images/test.csv",
                        empty_images_path="/orange/ewhite/everglades/Zooniverse/parsed_images/empty_test.csv",
                        save_dir="/orange/ewhite/everglades/Zooniverse/predictions/")
    
    
