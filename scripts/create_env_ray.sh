#!/bin/bash

# Build environment

ENV_NAME=env_ray
PROJ_ROOT=..

# Start at project root
cd $PROJ_ROOT

# Remove old env
if [ -d $ENV_NAME ]; then
    rm -rf $ENV_NAME
fi

# Make new virtualenv
virtualenv $ENV_NAME
source $ENV_NAME/bin/activate

# Install required packages
pip3 install -r requirements_common.txt
pip3 install -r requirements_ray.txt
