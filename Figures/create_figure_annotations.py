#Create shapefiles for figures to allow greater control.
from deepforest import main
import torch
import geopandas as gpd
import pandas as pd
from shapely import geometry
import shutil


# Figure 3
#A
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/seabirds_rgb_775.png", dst="seabirds_rgb_775.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/terns_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "seabirds_rgb_775.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("seabirds_rgb_775_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/terns_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/seabirds_rgb_775.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("seabirds_rgb_775_predictions.shp")

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
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/Dudley_projected_763.png", dst="Dudley_projected_763.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/palmyra_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "Dudley_projected_763.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("Dudley_projected_763_annotations.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/palmyra_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/Dudley_projected_763.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("Dudley_projected_763_predictions.shp")

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


#Figure 4
#A
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/BDA_18A4_20181107_1_38.png", dst="BDA_18A4_20181107_1_38.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/newmexico_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "BDA_18A4_20181107_1_38.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("BDA_18A4_20181107_1_38.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/newmexico_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/BDA_18A4_20181107_1_38.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("BDA_18A4_20181107_1_38_predictions.shp")

#B
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/KIPPa2016a_000144_3.png", dst="KIPPa2016a_000144_3.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/seabirdwatch_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "KIPPa2016a_000144_3.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("KIPPa2016a_000144_3.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/seabirdwatch_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/KIPPa2016a_000144_3.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("KIPPa2016a_000144_3_predictions.shp")

#C
shutil.copy2(src="/blue/ewhite/b.weinstein/generalization/crops/JackfishLakeBLTE_Sony_1_373.png", dst="JackfishLakeBLTE_Sony_1_373.png")
ground_truth = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/mckellar_test.csv")
ground_truth = ground_truth[ground_truth.image_path == "JackfishLakeBLTE_Sony_1_373.png"]
ground_truth["geometry"] = ground_truth.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
ground_truth = gpd.GeoDataFrame(ground_truth)
ground_truth.to_file("JackfishLakeBLTE_Sony_1_373.shp")

m = main.deepforest()
m.model.load_state_dict(torch.load("/blue/ewhite/b.weinstein/generalization/snapshots/mckellar_zeroshot.pt"))
boxes = m.predict_image(path = "/blue/ewhite/b.weinstein/generalization/crops/JackfishLakeBLTE_Sony_1_373.png")
boxes["geometry"] = boxes.apply(lambda x: geometry.box(x["xmin"],-x["ymin"],x["xmax"],-x["ymax"]), axis = 1)
boxes = gpd.GeoDataFrame(boxes)
boxes.to_file("JackfishLakeBLTE_Sony_1_373_predictions.shp")