### Zenodo upload
import requests
import glob
import os
import pandas as pd
import zipfile

def zip_dataset(dataset):
    train = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))
    test = pd.read_csv("/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset))
    df = pd.concat([train, test])
    images_to_upload = df.image_path.unique()
    zipname = "/blue/ewhite/b.weinstein/generalization/zenodo/{}.zip".format(dataset)
    z = zipfile.ZipFile(zipname,'w')
    z.write("/blue/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset), arcname="{}_train.csv".format(dataset))
    z.write("/blue/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), arcname="{}_test.csv".format(dataset))    
    for x in images_to_upload:
        z.write("/blue/ewhite/b.weinstein/generalization/crops/{}".format(x), arcname=x)
    z.close()
    
    return zipname

def get_token():
    token = os.environ.get('ZENODO_TOKEN')
    return token

def upload(ACCESS_TOKEN, path):
    """Upload an item to zenodo"""    
    # New API
    filename = os.path.basename(path)
    bucket_url = "https://zenodo.org/api/files/c86cef3b-ff4c-4a86-ab90-836f420d367a"
    
    # The target URL is a combination of the bucket link with the desired filename
    # seperated by a slash.
    with open(path, "rb") as fp:
        r = requests.put(
            "%s/%s" % (bucket_url, filename),
            data=fp,
            params={'access_token': ACCESS_TOKEN},
        )
    r.json()    
    print("request of path {} returns {}".format(path, r.json()))
    
if __name__== "__main__":
    
    #zipped_datasets = []
    #for x in ['mckellar',"seabirdwatch","palmyra","penguins","monash"]:
    #    z = zip_dataset(x)
    #    zipped_datasets.append(z)
    
    zipped_datasets = glob.glob("/blue/ewhite/b.weinstein/generalization/zenodo/*.zip")
    ACCESS_TOKEN = get_token()    
    for f in zipped_datasets:
        upload(ACCESS_TOKEN, f)
