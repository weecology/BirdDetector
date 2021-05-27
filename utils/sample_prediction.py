from deepforest import main
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from matplotlib import pyplot
m = main.deepforest.load_from_checkpoint("/Users/benweinstein/Downloads/terns_palmyra_penguins_pfeifer_hayes.pl")
m.model.score_thresh = 0.2
img = m.predict_tile("/Users/benweinstein/Downloads/AWPE counting images/IMG_0922-Subcolony U 2019-05-29.JPG",patch_size=1500,patch_overlap=0, return_plot=True)
#img = m.predict_image(path = "/Users/benweinstein/Downloads/AWPE counting images/IMG_0922-Subcolony U 2019-05-29.JPG", return_plot=True)
pyplot.show()
cv2.imwrite("/Users/benweinstein/Downloads/AWPE counting images/IMG_0922-Subcolony U 2019-05-29_prediction.JPG", img)