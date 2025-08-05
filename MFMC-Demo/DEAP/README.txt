# DEAP Dataset Setup Instructions

## Overview
This directory should contain the preprocessed DEAP dataset files (s01.dat to s32.dat) required for running the MFMC experiments.

## How to Download and Extract DEAP Dataset

### Step 1: Download the DEAP Dataset
1. Visit the official DEAP dataset website: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Download the **data_preprocessed_python.zip** files

### Step 2: Extract the Required Files
1. Extract the downloaded `data_preprocessed_python.zip` file
2. Inside the extracted folder, you will find files named `s01.dat`, `s02.dat`, ..., `s32.dat`
3. Copy all 32 .dat files (s01.dat through s32.dat) into this DEAP directory

### Expected Files
After completing the setup, this directory should contain:
- s01.dat
- s02.dat
- s03.dat
- ...
- s32.dat
(Total: 32 files, one for each subject in the DEAP dataset)

### Note for Reviewers
These files contain the preprocessed EEG and peripheral signals from 32 participants who watched 40 music videos. Each .dat file corresponds to one participant's complete session data.
