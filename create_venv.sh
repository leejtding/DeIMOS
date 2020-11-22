#!/bin/bash

# this installs the virtualenv module
python3 -m pip install virtualenv
# this creates a virtual environment named "env"
#python3 -m venv env
virtualenv --python=/opt/anaconda3/bin/python3.7 env
# this activates the created virtual environment
source env/bin/activate
# updates pip
pip install -U pip
# this installs the required python packages to the virtual environment
pip install -r requirements.txt

echo created environment
