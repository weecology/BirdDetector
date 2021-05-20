# Sythnetic counts from hogson 2018.

from deepforest import main
import os
import glob
import pandas as pd
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

df = pd.read_csv("/Users/benweinstein/Dropbox/Weecology/bird_detector/hodgson2018/doi_10.5061_dryad.rd736__v1/MASTER_AllCountData.csv")
true_counts = df[df.Count_type == "Retreived_skewers"]
m = main.deepforest.load_from_checkpoint("snapshot/species_model.pl")

results = []
for x in glob.glob("/Users/benweinstein/Dropbox/Weecology/bird_detector/hodgson2018/doi_10.5061_dryad.rd736__v1/Colony_imagery/30 m/*"):
    img = m.predict_tile(x, patch_size = 1000, return_plot=True)
    boxes = m.predict_tile(x, patch_size = 1000, return_plot=False)
    basename = os.path.splitext(x)[0]
    colony = int(x.split(".")[-2][-1])
    fname = "{}_prediction.png".format(os.path.splitext(x)[0])
    results.append(pd.DataFrame({"Colony":[colony],"predicted":[boxes.shape[0]]}))
    img.savefig(fname)
    
results = pd.concat(results)
results = results.merge(true_counts)
results["difference"] = results.Count - results.predicted
print(np.mean(abs(results["difference"])))

results.to_csv("Figures/hogdson2018.csv")
print(results)


    
