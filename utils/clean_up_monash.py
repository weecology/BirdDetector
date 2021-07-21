import glob
import os
import geopandas as gpd
import pandas as pd
shps = glob.glob("*.shp", recursive=True)
tif = glob.glob("*.tif")[0]
annotations = []
for x in shps:
    gdf = gpd.read_file(x)
    gdf.label="Bird"
    annotations.append(gdf)
annotations = gpd.GeoDataFrame(pd.concat(annotations))
annotations = annotations[annotations.geometry.type=="Point"]
annotations.to_file("{}.shp".format(os.path.splitext(tif)[0]))
