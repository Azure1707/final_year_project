# Run through of code

## Data extraction and splitting
### For stage 1:
prep_stage_1.py extracts images from the LIDC dataset, converting DICOM images to PNG and saving the 
create_stage_1_labels.py labels slices and stores the labels in a csv file 
sort_whole_clices_10each.py used the csv file containing labels to extract 10 slices each patient for the final dataset
split_while_slices.py splits the dataset into training, validation and test sets

### For stage 2:
prep_stage_2.py extracts malignant and benign crops from LIDC-IDRI dataset and saves them 
split_stage_2.py creates the splits for training, validation and test sets, saving them as a csv file.
copy_into_dir_stage2.py uses the splits csv file to copy the actual cropped ROI images into folders

## Preprocessing
preprocess_lung.py contains code to prepare ct slice images for stage 1.
preprocess_roi.py contains code to prepare ROI crops for stage 2.

##Diffusion model folder
Contains code for diffusion model

##WGAN-GP folder
Contains code for WGAN-GP model

##Models folder
Contains all the classification models for both stages

##Dashboard
Contains the code for the final dashboard





