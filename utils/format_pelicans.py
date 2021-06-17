#Format pelicans
import glob
import os
import pandas as pd
import shutil
import xmltodict
from shapely import geometry
import geopandas as gpd

def create_geodataframe(annotations):
    annotations["geometry"] =[geometry.Point(x,y) for x,y in zip(annotations.x.astype(float), annotations.y.astype(float))]
    gdf = gpd.GeoDataFrame(annotations)
    
    return gdf

def find_files():
    return glob.glob("/orange/ewhite/b.weinstein/neill/**/*.xml",recursive=True)

def find_image(name, year):
    image_pool = glob.glob("/orange/ewhite/b.weinstein/neill/**/*.JPG",recursive=True)
    matching = [x for x in image_pool if name in x]
    matched = [x for x in matching if year in x]
    
    if len(matched) == 0:
        print("Cannot find image for {} {}".format(name, year))
        return None
        
    return matched
    
def parse(x):
    with open(x) as fd:
        doc = xmltodict.parse(fd.read())
    filename = doc["CellCounter_Marker_File"]["Image_Properties"]["Image_Filename"]
    
    x = []
    y = []
    for annotation in doc["CellCounter_Marker_File"]["Marker_Data"]["Marker_Type"][6]["Marker"]:
        x.append(annotation["MarkerX"])
        y.append(annotation["MarkerY"])
    
    annotations = pd.DataFrame({"x":x,"y":y})
    annotations["label"] = "Bird"
    
    return filename, annotations

def get_year(x):
    year = os.path.dirname(x).split("/")[-2].split(" ")[1]
    
    return year

def run():
    files = find_files()
    for x in files:
        #Load image and save file foldder info on year
        year = get_year(x)
        image_basename = os.path.splitext(image_path)[0]
        image_name, annotations = parse(x)
        image_path = find_image(image_name, year)
        
        #run if image exists
        if image_path: 
            gdf = create_geodataframe(annotations)
            gdf["image_path"] = "{}_{}.JPG".format(image_basename, year)
            shp_path = "/orange/ewhite/b.weinstein/generalization/neill/parsed/{}.shp".format(os.basename(image_path))
            image_rename = "/orange/ewhite/b.weinstein/generalization/neill/parsed/{}_{}.JPG".format(image_basename, year)
            shutil.copy2(image_path, image_rename)
            gdf.to_file(shp_path)