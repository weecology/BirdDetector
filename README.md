# BirdDetector

A global bird detector model for the [deepforest package](https://deepforest.readthedocs.io/).
This repo recreates the analysis for the [manuscript](https://docs.google.com/document/d/15zqYo-3hmOMDLRKiUfkQwx7jesIhz0AypJLa30d8nA8/edit?usp=sharing).

## Getting Started

Here is a colab tutorial that demonstrates the basic package workflow.

https://colab.research.google.com/drive/1e9_pZM0n_v3MkZpSjVRjm55-LuCE2IYE?usp=sharing


## Environment

In a clean conda env.

```
pip install deepforest
```

## Data

Where possible we have released data to make a 'machine learning ready' dataset. See the [zenodo archive](https://zenodo.org/record/5033174). Each dataset was zipped into an archive. Each contains images and two .csv files one for train and one for test. 

## Everglades Model

The everglades data was annotated on Zooniverse and parsed using the utilities in the EvergladesWadingBird [Zooniverse repo](https://github.com/weecology/EvergladesWadingBird/blob/master/Zooniverse/aggregate.py). This repo starts after the data have been downloaded and split into training/test.
The everglades.py script creates a [comet experiment](https://www.comet.ml/bw4sz/everglades).




