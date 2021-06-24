#Single run
from generalization import *
comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                            project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")

ImageFile.LOAD_TRUNCATED_IMAGES = True
savedir = "/orange/ewhite/b.weinstein/generalization/snapshots/"

model = BirdDetector(transforms=get_transform)
config = model.config
config["gpus"] = 6
path_dict = prepare()

#Log commit
comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
comet_logger.experiment.log_parameters(model.config)
view_training(path_dict, comet_logger=comet_logger)

#Train Models
train_sets = ["seabirdwatch","neill","USGS","hayes","terns","penguins","pfeifer","palmyra","mckellar","monash","everglades"]
test_sets =["palmyra"]
result = zero_shot(path_dict=path_dict,
             config=config,
             train_sets=train_sets,
             test_sets=test_sets,
             comet_logger=comet_logger,
             savedir=savedir)