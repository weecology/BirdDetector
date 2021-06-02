# BirdDetector

A global bird detector model for the DeepForest-pytorch package.
This repo recreates the analysis for the [manuscript](https://docs.google.com/document/d/15zqYo-3hmOMDLRKiUfkQwx7jesIhz0AypJLa30d8nA8/edit?usp=sharing).

## Getting Started

Here is a colab tutorial that demonstrates the basic package workflow.

https://colab.research.google.com/drive/1e9_pZM0n_v3MkZpSjVRjm55-LuCE2IYE?usp=sharing


## Environment

```
conda env create --f=environment.yml
conda activate BirdDetector
```

## Data

This repo largely deals with projects from the Weecology Lab at the University of Florida. In most cases the data can be made available, but given its large size (>1TB), it is not hosted publically. 

## Everglades Model

The everglades data was annotated on Zooniverse and parsed using the utilities in the EvergladesWadingBird [Zooniverse repo](https://github.com/weecology/EvergladesWadingBird/blob/master/Zooniverse/aggregate.py). This repo starts after the data have been downloaded and split into training/test.
The everglades.py script creates a [comet experiment](https://www.comet.ml/bw4sz/everglades).

### Species model

## Leave-one-out Generalizåtion

To test the generalization of the global bird detection, the generalization.py script loads all the datasets

## Validation 





