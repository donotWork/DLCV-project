# DLCV-project
class project 

# System Setup
Install latest mini-conda for your system
Once installed activate it, and make sure it is the default python.

Update pip to the latest version, a sample command command is given below
pip install pip / pip install upgrade-pip
I have used python 3.12 for my code, so make sure to have that or any compatible version.

Make a conda environment, using following command, 
conda create --name <my-env>

> [!NOTE]
> I made the environment named loco for my code. But if you want to make a environment
by some other name you can but, but make sure to call it in the .sh file appropriately.

Install the following libraries in the environment.
 
pillow , numpy, matplotlib, ipython, pytorch-msssim, torch, torchvision, ultralytics, albumentations


> [!NOTE]
>  If you get a error on the vgg16 module missing, download pretrained vgg16 model, and save it into
the cache of the device, so that model is able to recognize it.\
> 					Or\
> Run the PyTorch download part of the vgg16 model, in the machine with internet, and transfer the model 
from there to your device. 

* Download the pretrained yolov8m model from online as well, and save it in your main directory.
Below is the link that I used for my use, you can run it directly in your terminal in the current working directory.

curl -L -o yolov8m.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8m.pt 

Once it is done, environment setup is Done.

# Data setup
Now setup the data as follows, 
<!--  Download the data from the given drive link. -->


**You can find the data online as well, on the online platforms whose links are as follows,**

Dark_face dataset:	https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset \
WIDER_FACE dataset:	http://shuoyang1213.me/WIDERFACE/ \
RetinexNet dataset:	https://daooshee.github.io/BMVC2018website/


Note: Even though images and all is fine, kindly convert the labels of the dataset into the required YOLO format
 when you download the dataset from the links given above.

Structure the dataset as follows, \
main dir: \
--------WIDER_train: \
--------------------images: \
--------------------labels: \
--------WIDER_val: \
------------------images: \
------------------labels: \
--------dark_face: \
------------------images: \
------------------labels: \
--------RetinexNet_Dataset: \
------------------retinexdata: \
------------------------our485: \ 
------------------------syn: \
------------------------test: \



Note: I am not using the dynamic way of inputting the directory main name in my code. So 
kindly check that carefully. Otherwise code will give some error. On my end I have used
full paths for all of my directories related to my system.


Training the RetinexNet code, for generating well lit images from low light images.

Save all the codes in the main dir your system, and run the code.


# Run
## RetinexNet
Train the RetinexNet model on the RetinexNet data whose download link is provided to you.
After training the model, run the prediction model on the dark_face dataset, to get the relit images for the darkface dataset.

## Objection Detection
After getting relit images, change the formatting of labels to the YOLO acceptable format. A python file is provided in code that converts the dark_face annotations, to the YOLO acceptable annotations, and saves them in a new folder.
Now split the darkface dataset, using the data split code provided in the repo.
> [!NOTE]
> It will only split data into 2 parts, so run the code twice, first for train val split and then for val test split, but change the folder paths in the code, accordingly.
> The labels will be split together with the data only.


Once the above steps are done run the yolo train model, for getting the final results.
