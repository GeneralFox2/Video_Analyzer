# Video Analyzer And Cleaner

This repository consists of two programs:    

analyze_video.py - This application can be used to watch a video, frame by frame. The application detects if the frame is safe for work or not (NSFW), and detects faces. Per each face detected, the appication will predict the person's age, gender, and will try to identify them using an image library.  

clean_video.py - Creates a copy of a video with any frames detected as NSFW removed from the copy.


## Environment

I did not update these applications to work with newer versions of the supporting modules (Tensorflow, etc.), so I did my best to re-create the necessary environment manually. I built an Ubuntu 16.04 LTS 64-bit environment. I did not build a virtual development environment. Python 3 was used.


## Built With

This repository contains an implementation of Yahoo's Open NSFW Classifier rewritten in tensorflow. The original caffe weights have been extracted using Caffe to TensorFlow. You can find them at pretrained_models/open_nsfw-weights.npy.  
  
The applications were built using the following modules installed via pip3:
numpy (1.13.3), six (1.13.0), setuptools (42.0.2), mock (3.0.5), future (0.17.1), Keras (2.2.3), Keras-Applications (1.0.8), Keras-Preprocessing (1.1.0), enum34 (1.1.6), wheel (0.33.6), autograd (1.3), scikit-image (0.14.1), scikit-learn (0.15.0), scipy (1.1.0), matplotlib (3.0.0), PyWavelets (1.1.1), ipython (7.10.1), comb (0.9.10), tensorflow (1.12.3)  
  
Other modules installed by apt:  
git, python3-pip, pinta, OpenCV


## Installation Procedure

Interdependancies require a specific installation of the modules, so I suggest you follow the commands listed at the bottom of this ReadMe.md (Installation Procedure - Steps) to install the modules.  
  
Clone this to your hard drive.  Create a folder, "temp_files" in the Video_Analyzer folder.  
  
Create a folder called "ids" in the Video_Analyzer folder. This is the folder where you will store the images of those people you wish the application to identify in videos. For each person, create a sub-folder using their name as the folder name. Place images of that person in the folder, using the following naming convention "foldernameXX", replacing XX with sequential numbers starting at 00. Headshots work best, taken from different angles, and having different facial expressions. Each image should only have one face present.  
  
You will need to download the weights and pre-trained models as described in the "pretrained_models" folder. Do the same with the "model" folder.  


### Usage

Analyze_Video.py - run this program using: python3 analyze_video.py {videoname.extension}  
  
Clean_Video.py - run this program using: python3 clean_video.py {videoname.extension}  


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Yusuke Uchida - Age / Gender Estimation: https://github.com/yu4u/age-gender-estimation  
* Assama Afzal - NSFW Prediction: https://github.com/usama093/tensorflow-open_nsfw  
* Habrman - Face Recognition: https://github.com/habrman/FaceRecognition  


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
*Determine your version of numpy - mine was 1.17.4*  
pip list 
pip3 uninstall numpy=1.17.4  
pip3 install -U --user numpy==1.13.3  
  
*Install pinta, freeze, comb*  
sudo add-apt-repository ppa:moonlight-team/pinta  
sudo apt-get update  
sudo apt-get install pinta  
pip3 install comb  
