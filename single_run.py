#Single run
from generalization import *
import argparse

parser = argparse.ArgumentParser(description='single dataset run')
parser.add_argument("dataset", type=str)
args = parser.parse_args()
print("Running {}".format(args.dataset))
comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                            project_name="everglades", workspace="bw4sz",auto_output_logging = "simple")

ImageFile.LOAD_TRUNCATED_IMAGES = True
savedir = "/orange/ewhite/b.weinstein/generalization/snapshots/"

model = BirdDetector(transforms=get_transform)
config = model.config

path_dict = prepare()

#Log commit
comet_logger.experiment.log_parameter("commit hash",subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
comet_logger.experiment.log_parameters(model.config)
#view_training(path_dict, comet_logger=comet_logger)

#Train Models
train_list = ["seabirdwatch","neill","USGS","hayes","terns","penguins","pfeifer","palmyra","mckellar","monash","everglades"]
train_sets = [y for y in train_list if not y==args.dataset]
test_sets =[args.dataset]
result = run(path_dict=path_dict,
             config=config,
             train_sets=train_sets,
             test_sets=test_sets,
             comet_logger=comet_logger,
             savedir=savedir)

result.to_csv("Figures/result_{}.csv".format(args.dataset))