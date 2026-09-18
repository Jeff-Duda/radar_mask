#!/bin/bash

source /home/jduda/miniforge_source
conda activate astropy

python -u /work/noaa/wrfruc/jdduda/radar_mask/process_refl_workflow.py ${TIME}


