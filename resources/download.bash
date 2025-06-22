#!/bin/bash --login

git clone git@github.com:natural-hazards/earthquakes_datasets.git
mv earthquakes_datasets/data .
rm -rf earthquakes_datasets