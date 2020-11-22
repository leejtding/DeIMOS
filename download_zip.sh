#!/bin/bash

#this creates data directory where the data zip file will be downloaded to
mkdir data
cd data

#this downloads the zip file that contains the data
curl https://zenodo.org/record/2538136/files/hirise-map-proj-v3.zip?download=1 --output deimos.zip

cd ..
echo downloaded data
