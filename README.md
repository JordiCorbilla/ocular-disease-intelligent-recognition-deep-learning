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

```cmd
python.exe odir_runner.py
```

Put all the training images under dataset folder.

#### License
Apache License, Version 2.0
