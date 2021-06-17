#rename pelican data
import os
import glob

files = glob.glob("**/*.JPG",recursive=True)
for x in files:
    full_path = os.path.abspath(x)
    dirname = os.path.dirname(full_path)
    basename = os.path.basename(full_path)
    new_basename = basename.split(" ")[-1]
    os.rename(full_path, "{}/{}".format(dirname, new_basename))