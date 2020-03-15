# Face Check-in 
Automated Face Check-in using Face Recognition Machine Learning

## About this project
This project is part of Facebook Machine Learning hackathon. The program processes the images of various individuals' faces, recognizes the face, classifies the images using a pre-trained machine learning model, and provides the matching class result.

## Inspiration
Our project’s inspiration came from a common issue that we noticed in the ticketing system, specifically the ticket checking process, in Indian railway and bus systems. Currently, the process of checking tickets is being done manually by a person, leading to more wait time for the passengers and less efficiency. We wanted to find a way to automate the ticket checking process, by simply having a user check-in at a metro/bus by having his/her picture taken or from a video frame image, making the process quick and efficient. 

## What it does
The underlying machine learning model analyzes the picture taken by the user at the check-in at a train or a bus, matches it with the user-submitted picture at the time of user registration, and recognizes the correct user. At this point, further back-end ticket checking processes can occur. Since the pictures at the check-in are taken from slightly different angles, in different light conditions, and perhaps even different orientations, our model should be able to work through those situations and predict the correct individual. 

## How to use it:

clone this repo, removing the '-' to allow python imports:
```bash git clone https://github.com/kmorawala/Face-Check-in.git facenet_pytorch```

Please be sure to import all of the following items in Python:
* cv2 (it's best to download it through conda installation: conda install -c menpo opencv)
* tkinter (Run "pip install tkinter" in command line)
* facenet_pytorch (Run "pip install facenet-pytorch" in command line)
* torch (Run "pip install torch" in command line)
* torchvision (Run "pip install torchvision" in command line)
* numpy (Run "pip install numpy" in command line)
* pandas (Run "pip install pandas" in command line)

Once installed, run the capture_image.py program to capture any images using your laptop camera. If you already have several images that can be used to used to create several classes and have images to evaluate, feel free to skip this step. 

The classes can be defined by creating a folder in the "Classes" folder and placing an image of an individual in the appropriate class. For example, "Person01" should be the name of the person and the subfolder name whose image is being stored. Similarly, the images to classify should be placed in the "Test Faces" folder under "Test" folder. The folder names can be changed or existing folders can be utilized since the program will prompt the users to select a folder each time a folder path is required. Feel free to delete unused folders and/or images when you begin your own! 

At this point, be sure to remove any files other than image files must be removed from above mentioned folders. .DS_Store file is known to download from github.

Now, you may run the app.py program in Python and select the necessary folders as requested:
 * First, you will be prompted to select the folder which has classes of images. In the example, it would be the "Classes" folder.
 * Second, you will be prompted to select the folder where a subfolder has images to be evaluated. In the example, it would be the "Test" folder
 * Third, you will be prompted to select the folder where the images to be evaluated are directly placed. In the example, it would be the "Test_Faces" folder
 * Fourth, you will be prompted to select the folder where you would like the result csv file to be stored.
 
 The results will also be displyed on the termine if running the program on terminal! 

## How we built it
We are using a pre-trained model, Inception Resnet (V1) model in PyTorch, pre-trained on VGGFace2 for this purpose from https://github.com/timesler/facenet-pytorch. For each user that can be recognized, a class is set up with an image or images of the individual. Futher, another testing class in a separate folder should be set up, where an image or images that need to be recognized against the existing classes are placed. For each image, a matching "class/label" will be outputted for which a matching class is found, otherwise the unknown category will be printed. Further, for a detailed explanation, a result table will follow that will calculate the distance and in turn, recognize which individual’s image is there. 

A second Python program to capture user image is also included, so the user can easily have their images taken and stored at a folder of the user's choice. 

Further, we used the images Head Pose Image Database (at http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html) to test our model for various face angles:

We used the following technologies:
* Python
* PyTorch
* TorchVision
* facenet_pytorch
* numpy
* pandas
* tkinter
* cv2 (OpenCV)
* Google Collaboratory
* Github

## Pretrained models

See: [models/inception_resnet_v1.py](models/inception_resnet_v1.py)

The following model has been ported from facenet-pytorch (with links to download pytorch state_dict's):

|Model name|LFW accuracy (as listed [here](https://github.com/timesler/facenet-pytorch))|Training dataset|
| :- | :-: | -: |
|[20180402-114759](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1) (107MB)|0.9965|VGGFace2|

The pretrained model was trained on 160x160 px images, so will perform best if applied to images resized to this shape, which is done in the data loading portion of our program.

## Challenges we ran into
All of our team members do not have extensive experience in this area, and some of us are completely new to machine learning. The challenge was to familiarize ourselves quickly with machine learning concepts, PyTorch, image processing, deep learning, etc. Second, finding an appropriate dataset to test was difficult since we needed to account for items such as different face angles, train data balancing in terms of gender, lighting, orientation, etc. and finding high-quality images or database was quite difficult to test the model. Third, picking an appropriate model was also quite a challenge and required us to do a lot of research.

## Accomplishments that we're proud of
Having no background in machine learning/image processing and having built this project, we feel proud of how much progress we have made and how much we have learned along the way. This project can have quite a few different applications and future uses besides only the one use case that we have found so far. We are excited to help others using this image processing technology! 

## What we learned
We learned an abundance of things from where we started. A few of the highlights are as follows:
Machine learning/CNN basic terminology,
Image processing in machine learning,
How neural networks work, 
Importance of a balanced dataset,
How images get processed in machine learning,
Types of machine learning models,
How to use pre-trained machine learning models,
Testing an existing model,

## What's next for our project
We would like to build a better front-end for this project, allowing easier access for others to utilize it. Further, we see that the algorithm can be optimized for training and find a way to only periodically re-train the model using the newly added user class to save memory and computational power. Further, instead of storing each individual’s image, some other form of labels should be stored in a light-weight text file to enable quick processing and less memory usage.

We would love to see this technology being implemented in the Indian railway as well as bus systems and people reaping benefits in terms of speed and efficiency.

## References

1. Tim Esler's facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

2. Head Pose Image Database [http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html] by
N. Gourier, D. Hall, J. L. Crowley 
Estimating Face Orientation from Robust Detection of Salient Facial Features
Proceedings of Pointing 2004, ICPR, International Workshop on Visual Observation of Deictic Gestures, Cambridge, UK


