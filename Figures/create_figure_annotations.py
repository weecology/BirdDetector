#Create shapefiles for figures to allow greater control.
from deepforest import main
import torch
import geopandas as gpd
import pandas as pd
from shapely import geometry
import shutil


# Figure 3
#A
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/seabirds_rgb_893.png", dst="seabirds_rgb_809.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/terns_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "seabirds_rgb_809.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("seabirds_rgb_809_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/terns_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/seabirds_rgb_893.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("seabirds_rgb_809_predictions.shp")

#B
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/Rzepecki Islands_south_2016_Chinstrap_penguins_76.png", dst="Rzepecki Islands_south_2016_Chinstrap_penguins_76.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/pfeifer_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "Rzepecki Islands_south_2016_Chinstrap_penguins_76.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("Rzepecki Islands_south_2016_Chinstrap_penguins_76_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/pfeifer_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/Rzepecki Islands_south_2016_Chinstrap_penguins_76.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("Rzepecki Islands_south_2016_Chinstrap_penguins_76_predictions.shp")

#C
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/Dudley_projected_307.png", dst="Dudley_projected_307.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/palmyra_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "Dudley_projected_307.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("Dudley_projected_311_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/palmyra_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/Dudley_projected_307.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("Dudley_projected_307_predictions.shp")

#D
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/SteepleJason_Hump_Nov2019_transparent_mosaic_group1---381.png", dst="SteepleJason_Hump_Nov2019_transparent_mosaic_group1---381.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/hayes_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "SteepleJason_Hump_Nov2019_transparent_mosaic_group1---381.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("SteepleJason_Hump_Nov2019_transparent_mosaic_group1---381_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/hayes_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/SteepleJason_Hump_Nov2019_transparent_mosaic_group1---381.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("SteepleJason_Hump_Nov2019_transparent_mosaic_group1_predictions.shp")
