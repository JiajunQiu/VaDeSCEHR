#!/bin/bash

# Assign command line arguments to variables
primary_file=$1
hospital_file=$2
hospital_info_file=$3
basic_info_file=$4

# Execute python scripts with the appropriate arguments
python3 ./scripts/UKB_primary_EHR.py $primary_file
python3 ./scripts/UKB_hospital_EHR.py $hospital_file $hospital_info_file
python3 ./scripts/interleave.py $basic_info_file
python3 ./scripts/map.py

#./main.sh /UKB/primary_care/gp_clinical.txt /UKB/categories/2002/ukb41808.enc_ukb/Summary_Diagnoses.tab /UKB/categories/2002/fields_annotation.tsv /UKB/categories/100094/ukb41808.enc_ukb/Baseline_characteristics.tab