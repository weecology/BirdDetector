#Format pelicans
import glob
import os
import pandas as pd
import shutil
import xmltodict
from shapely import geometry
import geopandas as gpd
import cv2

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
    
    if not len(matched) == 1:
        print("Cannot find image for {} {}".format(name, year))
        return None
        
    return matched[0]
    
def parse(x):
    with open(x) as fd:
        doc = xmltodict.parse(fd.read())
    filename = doc["CellCounter_Marker_File"]["Image_Properties"]["Image_Filename"]
    
    x = []
    y = []
    for annotation in doc["CellCounter_Marker_File"]["Marker_Data"]["Marker_Type"]:
        try:
            objects = annotation["Marker"]
            for Bird in objects:   
                x.append(Bird["MarkerX"])
                y.append(Bird["MarkerY"])
        except:
            pass
    
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
        image_name, annotations = parse(x)
        image_path = find_image(image_name, year)
        
        #run if image exists
        if image_path: 
            image_basename = os.path.basename(os.path.splitext(image_path)[0])            
            gdf = create_geodataframe(annotations)
            gdf["image_path"] = "{}_{}.JPG".format(image_basename, year)
            gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(30).bounds.values]            
            shp_path = "/orange/ewhite/b.weinstein/neill/parsed/{}_{}.shp".format(image_basename, year)
            image_rename = "/orange/ewhite/b.weinstein/neill/parsed/{}_{}.JPG".format(image_basename, year)
            shutil.copy2(image_path, image_rename)
            gdf.to_file(shp_path)
 
def check_shape():
    df = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/neill_train.csv")
    for name, group in df.groupby("image_path"):
        img = cv2.imread("/orange/ewhite/b.weinstein/generalization/crops/{}".format(name))
        height = img.shape[0]
        width = img.shape[1]
        if any(group.xmax > width):
            print(name)
            df.loc[df.image_path == name, "xmax"] = width -1 
        elif any(group.ymax > height):
            print(name)
            df.loc[df.image_path == name, "ymax"] = height -1 
    #df.to_csv("/orange/ewhite/b.weinstein/generalization/crops/neill_train.csv")
            
if __name__ == "__main__":
    run()
    check_shape()