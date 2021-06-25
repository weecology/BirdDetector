#Comparison figures of images
from generalization import *
from deepforest import main
import torch
import os    
    
def run(x):
    path_dict = prepare()
    m = main.deepforest(label_dict={"Bird":0})
    m.model.load_state_dict(torch.load("/orange/ewhite/b.weinstein/generalization/snapshots/{}_finetune.pt".format(x)))    
    boxes = m.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops", savedir="/orange/ewhite/b.weinstein/generalization/snapshots/finetune/") 
    m.model.load_state_dict(torch.load("/orange/ewhite/b.weinstein/generalization/snapshots/{}_zeroshot.pt".format(x)))
    boxes = m.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops", savedir="/orange/ewhite/b.weinstein/generalization/snapshots/zeroshot/") 
    m.model.load_state_dict(torch.load("/orange/ewhite/b.weinstein/generalization/snapshots/{}_mini_1.pt".format(x)))
    boxes = m.evaluate(csv_file=path_dict[x]["test"], root_dir="/orange/ewhite/b.weinstein/generalization/crops", savedir="/orange/ewhite/b.weinstein/generalization/snapshots/1000bird/") 

if __name__ == "__main__":
    try:
        os.mkdir("/orange/ewhite/b.weinstein/generalization/snapshots/zeroshot/")
        os.mkdir("/orange/ewhite/b.weinstein/generalization/snapshots/finetune/")
        os.mkdir("/orange/ewhite/b.weinstein/generalization/snapshots/1000bird/")
    except:
        pass
    
    path_dict = prepare()
    for x in path_dict:
        run(x)