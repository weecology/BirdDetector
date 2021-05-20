from deepforest import main
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from matplotlib import pyplot
m = main.deepforest.load_from_checkpoint("snapshot/species_model.pl")
m.model.score_thresh = 0.05
img = m.predict_tile("/Users/benweinstein/Downloads/Do Arzereti eargheta, 22 maggio 2018.png",patch_size=800,patch_overlap=0, return_plot=True)
pyplot.show()
img.savefig("/Users/benweinstein/Downloads/Do Arzereti eargheta, 22 maggio 2018_prediction.png", dpi=300)