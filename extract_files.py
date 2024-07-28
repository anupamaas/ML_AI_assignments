# extract_files.py

import zipfile
import os

# Paths to the zip files and extraction directory
model_zip_path = r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\attendapp\static\model_files\fine_tuned_model.h5.zip'
label_encoder_zip_path = r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\attendapp\static\model_files\label_encoder.pkl.zip'
extraction_dir = r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\attendapp\static\model_files'

# Extract the zip files
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Extract both files
print('Extracting model zip...')
extract_zip(model_zip_path, extraction_dir)
print('Model zip extracted.')

print('Extracting label encoder zip...')
extract_zip(label_encoder_zip_path, extraction_dir)
print('Label encoder zip extracted.')

print('Extraction completed successfully.')
