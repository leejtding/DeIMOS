#!/bin/bash

mkdir data
cd data

curl https://zenodo.org/record/2538136/files/hirise-map-proj-v3.zip?download=1 --output deimos.zip

cd ..
echo downloaded data
