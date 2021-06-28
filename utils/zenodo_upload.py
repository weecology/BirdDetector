### Zenodo upload
import requests
import glob
import os
import pandas as pd
import zipfile

def zip_dataset(dataset):
    train = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset))
    test = pd.read_csv("/orange/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset))
    df = pd.concat([train, test])
    images_to_upload = df.image_path.unique()
    zipname = "/orange/ewhite/b.weinstein/generalization/zenodo/{}.zip".format(dataset)
    z = zipfile.ZipFile(zipname,'w')
    z.write("/orange/ewhite/b.weinstein/generalization/crops/{}_train.csv".format(dataset), arcname="{}_train.csv".format(dataset))
    z.write("/orange/ewhite/b.weinstein/generalization/crops/{}_test.csv".format(dataset), arcname="{}_test.csv".format(dataset))    
    for x in images_to_upload:
        z.write("/orange/ewhite/b.weinstein/generalization/crops/{}".format(x), arcname=x)
    z.close()
    
    return zipname

def get_token():
    token = os.environ.get('ACCESS_TOKEN')
    return token

def upload(ACCESS_TOKEN, path):
    """Upload an item to zenodo"""
    
     # Get the deposition id from the already created record
    deposition_id = "5033174"
    data = {'name': os.path.basename(path)}
    files = {'file': open(path, 'rb')}
    r = requests.post('https://zenodo.org/api/deposit/depositions/%s/files' % deposition_id,
                      params={'access_token': ACCESS_TOKEN}, data=data, files=files)
    print("request of path {} returns {}".format(path, r.json()))
    
    
    with open('path', 'rb') as fp:
        res = requests.put(
            bucket_url + '/my-file.zip', 
            data=fp,
            # No headers included in the request, since it's a raw byte request
            params=params,
        )
    print(res.json())
    
    
if __name__== "__main__":
    
    zipped_datasets = []
    for x in ['pfeifer',"everglades","hayes","terns"]:
        z = zip_dataset(x)
        zipped_datasets.append(z)
        
    ACCESS_TOKEN = get_token()    
    for f in zipped_datasets:
        upload(ACCESS_TOKEN, f)
