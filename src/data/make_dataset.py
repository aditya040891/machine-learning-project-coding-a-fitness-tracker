#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:59:04 2026

@author: aditya
"""
import pandas as pd
from glob import glob
pd.set_option('display.max_columns', None)

#------------------------------------------------------------------------
# Read single csv file
#------------------------------------------------------------------------

single_file_acc = pd.read_csv('data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_gyr = pd.read_csv('data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')


# -------------------------------------------------------------------------
# List all data in data/raw/MetaMotion
# -------------------------------------------------------------------------

files = glob("data/raw/MetaMotion/*.csv")

len(files)


# -----------------------------------------------------------------------------
# Extract Features from the filenames
# -----------------------------------------------------------------------------

files[0]

data_path = "data/raw/MetaMotion/"

f = files[0]


participant = f.split('-')[0].replace(data_path, '')

label = f.split('-')[1]

category = f.split('-')[2].rstrip('123').replace('_MetaWear_2019', '')


df = pd.read_csv(f)

df['participant'] = participant
df['label'] = label
df['category'] = category






















