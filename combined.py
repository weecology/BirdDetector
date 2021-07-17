#Single run
from generalization import *
comet_logger = CometLogger(project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")

ImageFile.LOAD_TRUNCATED_IMAGES = True
savedir = "/blue/ewhite/b.weinstein/generalization/snapshots/"

model = BirdDetector(transforms=get_transform)
config = model.config
config["gpus"] = 6
path_dict = prepare()

#Log commit
comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
comet_logger.experiment.log_parameters(model.config)
view_training(path_dict, comet_logger=comet_logger)

#Train Models
train_sets = ["seabirdwatch","neill","USGS","hayes","terns","penguins","pfeifer","palmyra","mckellar","monash","everglades","murres","schedl","valle","poland"]
test_sets =["palmyra"]
all_sets = []
print("Train sets: {}".format(train_sets))
for x in train_sets:
    try:
        df = pd.read_csv(path_dict[x]["train"])
        all_sets.append(df)            
    except:
        raise ValueError("No training path supplied for {}".format(x))
    try:
        df_test = pd.read_csv(path_dict[x]["test"])
        all_sets.append(df_test)
    except Exception as e:
        print("No test set for {}".format(x))
train_annotations = pd.concat(all_sets)

#A couple illegal boxes, make slightly smaller
train_annotations["xmin"] = train_annotations["xmin"].astype(float) 
train_annotations["xmax"] = train_annotations["xmax"].astype(float) - 3
train_annotations["ymin"] = train_annotations["ymin"].astype(float)
train_annotations["ymax"] = train_annotations["ymax"].astype(float) - 3

train_annotations = train_annotations[~(train_annotations.xmin >= train_annotations.xmax)]
train_annotations = train_annotations[~(train_annotations.ymin >= train_annotations.ymax)]
train_annotations.to_csv("/blue/ewhite/b.weinstein/generalization/crops/training_annotations.csv")

all_val_sets = []
for x in test_sets:
    df = pd.read_csv(path_dict[x]["test"])
    all_val_sets.append(df)
test_annotations = pd.concat(all_val_sets)
test_annotations.to_csv("/blue/ewhite/b.weinstein/generalization/crops/test_annotations.csv")

comet_logger.experiment.log_parameter("training_images",len(train_annotations.image_path.unique()))
comet_logger.experiment.log_parameter("training_annotations",train_annotations.shape[0])

train_df = pd.read_csv("/blue/ewhite/b.weinstein/AerialDetection/data/trainval1024/train.csv")
label_dict = {x: index for index, x in enumerate(train_df.label.unique())}    
pretrained_DOTA = main.deepforest(num_classes=15, label_dict=label_dict)
model = BirdDetector(transforms = get_transform)

#update backbone weights with new Retinanet head
model.model = create_model(num_classes=1, nms_thresh=model.config["nms_thresh"], score_thresh=model.config["score_thresh"], backbone=pretrained_DOTA.model.backbone)
model.config = config
model = fit(model, train_annotations, comet_logger, "palmyra_combined")
torch.save(model.model.state_dict(),"/blue/ewhite/b.weinstein/generalization/snapshots/combined.pt")
