# Face-Mask-detection
Open run.py to start Detection<br>
This Repo is Demonstration of how we can use some Computer Vision with Deep Learning to make a Face Mask Detection which can be used on Entry Gates of Companies or Public Transport or Simply used to Provide some services only to those wearing mask like Ticket Vending Machine, ATM, Etc. 

# Face Detection
Using OpenCv Haar Cascade Face detection to detect a face then the image is cropped and fed into the Classifier

# Model
Since we are classifying video feed in 3 Categories:<br>
    1. Wear Mask<br>
    2. Wear Mask Properly<br>
    3. Thank you for wearing Mask<br><br>
Firstly we used a CNN model made from Scratch on out custom made dataset. Since Dataset is very less so Model was finding it dificult to do feature extraction Succesfully.<br>
So we used a Pretrained model like (VGG16 , ResNet, etc) which gave us satisfactory Result

# Note
[Since Dataset was not available so dataset includes my faces wearing mask in different Position and Lightining Conditions and may need a little Callibration ( an automatic cllibration wont work unless we have good amount of data ) ]

