#Format the seabird watch data
import glob
import os
import pandas as pd
import geopandas as gpd
from shapely import geometry

files = glob.glob("/orange/ewhite/b.weinstein/seabirdwatch/*.csv")
f = pd.concat([pd.read_csv(x) for x in files])
df = f[["image_id","cluster_x","cluster_y","colonyname"]]
df = df.dropna()
df = df.rename(columns = {"image_id":"image_path"})
df["label"] = "Bird"   

for name, group in df.groupby("image_path"):   
    if os.path.exists("/orange/ewhite/b.weinstein/generalization/crops/{}".format(name)):   
        group["geometry"] =[geometry.Point(x,y) for x,y in zip(group.cluster_x.astype(float), group.cluster_y.astype(float))]
        gdf = gpd.GeoDataFrame(group)    
        gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(10).bounds.values]                            
        image_basename = os.path.splitext(gdf.image_path.unique()[0])[0]
        gdf.to_file("{}/{}.shp".format("/orange/ewhite/b.weinstein/seabirdwatch/parsed",image_basename))
        