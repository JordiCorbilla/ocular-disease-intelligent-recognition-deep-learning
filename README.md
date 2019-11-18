# Ocular disease image recognition tensorflow

### Works on:
[tensorflow-2.0](https://github.com/tensorflow/tensorflow) - use branch `master`

All the training images must be in JPEG format.

### Usage

## Image Treatment Process

Place all the files in the following folders (Training and Validation images):

```cmd
c:\temp\ODIR-5K_Training_Dataset
c:\temp\ODIR-5K_Testing_Images
```

Create the following folders:

```cmd
c:\temp\ODIR-5K_Testing_Images_cropped
c:\temp\ODIR-5K_Testing_Images_treated
c:\temp\ODIR-5K_Training_Dataset_cropped
c:\temp\ODIR-5K_Training_Dataset_treated
```

run the following command to treat the training and validation images:

```cmd
>python odir_image_crop_job.py
>python odir_image_training_treatment_job.py
>python odir_image_testing_treatment_job.py
```

## Image to tf.Data conversion and .npy storage

run the following command to generate the dataset for training and validation:

```cmd
>python odir_image_crop_job.py
```

```cmd
python.exe odir_runner.py
```

Put all the training images under dataset folder.

#### License
Apache License, Version 2.0
