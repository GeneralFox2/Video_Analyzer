# Video Analyzer And Cleaner

This repository consists of two programs:    

analyze_video.py - This application can be used to watch a video, frame by frame. The application detects if the frame is safe for work or not (NSFW), and detects faces. Per each face detected, the appication will predict the person's age, gender, and will try to identify them using an image library. The actor will be identified as a minor if their age is less than 18.  

clean_video.py - Creates a copy of a video with any frames detected as NSFW removed from the copy.  The application creates a copy of the video, using the AVI format.  


### Built For
The idea for this application came from my review of a system known as MEMEX, which was used by the United States FBI to help catch those involved in human trafficking and to free those children that were affected by the crime. In my research I noticed that a free, open-source application that could be used to review video's for NSFW content and detect minors was not available to both the general public and law enforcement agencies, and I wanted to correct that. This application reviews the video frame by frame so as to help ensure that even a minor appearance of a missing person (abducted child) is detected.  

My intention was to create a powerful application that had an insignificant memory and disk footprint, as it could then be run on a grouping of low-cost single board computers, such as the Raspberry PI. This would allow the detection of NSFW video content and missing persons at an extremely low cost. The program is slow, as it runs uncompiled in Python - however it was written to be easily ported to another faster language.  

The application is easy to adjust per needs.  For instance, per my configuration the application runs on a single CPU (no GPU) - but by adjusting Tensorflow the application could be easily made to work across multiple CPU (or GPU) cores.  Triggers are easily spotted in the code, and could be adjusted to output information to a file, or send a message, or show an on-screen advisory, etc., as needed.  The image library code could be substituted to enable the ability to connect with a database.  It could be adapted to review a video stream for 'live' feeds, or review the contents of a hard drive and for videos and inspect them, etc., etc..  


### Future Development / Improvements
I plan to upgrade the face detection engine at a later time, to improve the ability of the application in detecting faces at various angles. As well, when provided with a limited number of actor images, the program is more apt to incorrectly identify people - which I believe can be remedied by increasing the resolution of the face detection system. I've also noticed that the application has difficulty detecting faces when they are not upright - I feel that application needs to be able to detect and identify faces when people are in other positions than upright.  I also recognize a need to improve the ability of the application in detecting people who are in limited light situations. I also plan to update all of the modules referenced in the application.  


### Command Line (Usage)

Analyze_Video.py - run this program using: python3 analyze_video.py {videoname.extension} {picture folder}  
Replace {videoname.extenstion} with the name of the video to be reviewed.  
Replace {picture folder} with the name of the picture repository you created.  
  
Clean_Video.py - run this program using: python3 clean_video.py {videoname.extension}  
Replace {videoname.extenstion} with the name of the video to be reviewed.  


### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


### Acknowledgments

Thanks go out to the following developers whose vision provided me with the ability to create this application!
* Yusuke Uchida - Age / Gender Estimation: https://github.com/yu4u/age-gender-estimation  
* Assama Afzal - NSFW Prediction: https://github.com/usama093/tensorflow-open_nsfw  
* Habrman - Face Recognition: https://github.com/habrman/FaceRecognition  


## Environment

The applications used to construct this application referenced older modules (such as Tensorflow, numpy, etc.) - I did not update them. Instead, I simply re-created the original development environment manually. To ensure backward compatibility I used an Ubuntu 16.04 LTS 64-bit environment. I did not use a virtual development environment. I used Python 3.


### Built With

This repository contains an implementation of Yahoo's Open NSFW Classifier rewritten in tensorflow. The original caffe weights have been extracted using Caffe to TensorFlow. It also contains a Keras implementation of a CNN network for age and gender estimation, using the IMDB-WIKI dataset. And it performs face recognition using tensorflow and opencv using deep neural networks.  
  
The applications were built using the following modules installed via pip3:  
numpy (1.13.3), six (1.13.0), setuptools (42.0.2), mock (3.0.5), future (0.17.1), Keras (2.2.3), Keras-Applications (1.0.8), Keras-Preprocessing (1.1.0), enum34 (1.1.6), wheel (0.33.6), autograd (1.3), scikit-image (0.14.1), scikit-learn (0.15.0), scipy (1.1.0), matplotlib (3.0.0), PyWavelets (1.1.1), ipython (7.10.1), comb (0.9.10), tensorflow (1.12.3)  
  
Other modules installed by apt:  
git, python3-pip, pinta, OpenCV


### Installation Procedure

Interdependancies require a specific installation of the modules, so I suggest you follow the commands listed at the bottom of this ReadMe.md (Installation Procedure - Steps) to install the modules. Once complete, return to this section and continue.  
  
Clone this to your hard drive.  Create a folder, "temp_files" in the Video_Analyzer folder. Download the models found in the "model" and "pretrained_models" folders.  

Create a folder in the Video_Analyzer folder that you will use to store the images of those people you wish the application to identify in the videos. After you create it, then for each person you will need to create a sub-folder using their name as the folder name. You then need to place images of that person in the folder, naming each picture using the following naming convention "foldernameXX", replacing XX with sequential numbers starting at 00. Headshots work best, taken from different angles, and having different facial expressions. Each image should only have one face present.  Using the folder name "ids" and an actor "Eve", the folder would look like this:  

ids  
| -> Eve  
      |->Eve00.jpg  
      |->Eve01.jpg  
etc..  
  
You will need to download the weights and pre-trained models as described in the "pretrained_models" folder, and the "model" folder.  


### Installation Procedure - Steps
*Install GIT*  
sudo apt update  
sudo apt install git  
*Install pip*  
sudo apt update  
sudo apt install python3-pip  
*Install numpy*  
pip3 install -U --user numpy=1.10.1  
  
*Install six, setuptools, mock, future, keras, keras_applications, keras_preprocessing, enum34, wheel, autograd*  
pip3 install -U --user six setuptools mock future==0.17.1  
pip3 install -U --user keras_applications --no-deps  
pip3 install -U --user keras_preprocessing --do-deps  
pip3 install -U --user enum34 wheel  
python3 -m pip install autograd --no-deps  
pip3 install -U --user keras==2.23  
  
*Download and install Bazel 0.15.0*  
*Find your version in the Asset list here: https://github.com/bazelbuild/bazel/releases?after=0.18.1*  
chmod +x bazel-0.15.0-installer-linux-x86_64.sh  
./bazel-0.15.0-installer-linux-x86_64.sh --user  
  
*Build Tensorflow r1.12.0 installation media*  
git clone https://github.com/tensorflow/tensorflow.git  
cd tensorflow  
git checkout r1.12  
bazel shutdown  
./configure  {NOTE: python found in: /usr/bin/python3.5}  
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package  
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg  
*Built package is found in /tmp folder "tensorflow_pkg" - substitute the correct name below*  
pip3 install /tmp/tensorflow_pkg/tensorflow-1.12.3-cp35-cp35m-linux_x86_64.whl  
  
*Install scikit-image, scipy, matplotlib, PyWavelets, scikit-learn, ipython*  
pip3 install -U --user scikit-image==0.15.0 --no-deps  
pip3 install -U --user scipy==1.1.0 --no-deps  
pip3 install -U --user matplotlib==3.0 --no-deps  
pip3 install -U --user PyWavelets --no-deps  
pip3 install -U --user scikit-learn=0.15.0 --no-deps  
pip3 install -U --user ipython --no-deps  
  
*Now install OpenCV for Python3 - follow the detailed steps outlined here: https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/   Note that I did not install into a virtual environment.*  
  
*Interdependancies installed a newer version of numpy - downgrade to 1.13.3*  
*Use the "pip list" command to determine the version of numpy installed on your system - mine was 1.17.4*  
pip list  
pip3 uninstall numpy=1.17.4  
pip3 install -U --user numpy==1.13.3  
  
*Install pinta, freeze, comb*  
sudo add-apt-repository ppa:moonlight-team/pinta  
sudo apt-get update  
sudo apt-get install pinta  
pip3 install comb  

*Return to the Installation Procedure above.*
