# CNN-ObjectDetection
In this project, I used YOLO algorithm trained on [Dataset](https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset?resource=download) for object detection task with Inference on video data using Convolutional Neural Network (CNN). I used pretrained Yolov3 model which can downloaded from the official YOLO [Website](https://pjreddie.com/darknet/yolo/)
<center>Examples</center>
<p align="center">
  <img src="https://user-images.githubusercontent.com/130169662/230648205-b70e44ca-2d14-4398-a7ad-fbadebdf8418.png" width="400" height="400" />
  <img src="https://user-images.githubusercontent.com/130169662/230648217-ad46d4bd-7277-404a-982d-045927d49188.png" width="400" height="400" /> 
</p>

### System Requirements

To run the model, it is recommended that you have the following system requirements:

- A good CPU and a GPU with at least 4GB memory
- At least 8GB of RAM
- An active internet connection to download the YOLOv2 weights and cfg file.

### Required Libraries

The following Python libraries were used in the making and testing of this project, along with their version numbers:

- Python: 3.6.7
- Numpy: 1.16.4
- Tensorflow: 1.13.1
- Keras: 2.2.4
- PIL: 4.3.0

## The Dataset

The dataset used in this project is the Weed Crop Image Dataset. It contains over 2822 images.

### Subset

Weed are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:

Auto-orientation of pixel data (with EXIF-orientation stripping)
The following augmentation was applied to create 3 versions of each source image:

Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
Random shear of between -15° to +15° horizontally and -15° to +15° vertically
Random brigthness adjustment of between -25 and +25 percent


### Example

Here's an example of how the original images look:

![alt text](insert-image-url-here)

### Downloading the Data

To download the type of data used in this project, follow these steps:

1. Click on the 'Download' button.
2. Then click on 'Download from Figure Eight'.
3. Next, click on 'Download Options' in the top right corner.
4. Under 'Train_00.zip', download 'train-annotations-bbox.csv' and 'train-images-boxable.csv'.
5. Scroll down the page to the 'Please Note' section, and click on the hyperlink in the second paragraph labeled 'You can download it here'.

