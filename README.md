# Video Analyzer And Cleaner

This repository consists of two programs:    

analyze_video.py - This application can be used to watch a video, frame by frame. The application detects if the frame is safe for work or not (NSFW), and detects faces. Per each face detected, the appication will predict the person's age, gender, and will try to identify them using an image library. The actor will be identified as a minor if their age is less than 18.  

clean_video.py - Creates a copy of a video with any frames detected as NSFW removed from the copy.  The application creates a copy of the video, using the AVI format.  


### Built For
The idea for this application came from my review of a system known as MEMEX, which was used by the United States FBI to help catch those involved in human trafficking and to free the victims. In my research I noticed that a free, open-source application that could be used to review video's for NSFW content and detect minors was not available to both the general public and law enforcement agencies, and I wanted to correct that. 

This application was just part of the total solution.  It reviews videos frame by frame and can detect NSFW content, minors, and can identify individuals (provided an image has been given of that person).  The application runs on a single CPU (no GPU) - but could be scaled to be used across multiple GPU cores to speed the detection rate by adjusting Tensorflow.  It could be ported to a faster language to speed the detection rate, or be run on a grouping of low-cost single board computers, such as the Raspberry PI.  The image library code could be substituted to enable the ability to connect with a live database.  The program could be adapted to review a video stream for 'live' feeds, or review the contents of a hard drive and for videos and inspect them.

The second part of my solution involved the automated reviewing of Internet videos, looking for human trafficking victims and minors.  The intention was to couple this application with this one.  I was unable to get this part to work properly, but in my research found another solution (PPCensor) which is overall superior to my own efforts: https://www.researchgate.net/publication/342309042_PPCensor_Architecture_for_real-time_pornography_detection_in_video_streaming  Their application runs as a web service (proxy), can detect NSFW content in real-time while streaming, and can censor out NSFW content.  This group has other works, which if combined, trump my own efforts - so I will not be working on these applications further.


### Future Development / Improvements
None at this time.


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

I used Phton 3.  The applications used to construct this application referenced older modules (such as Tensorflow, numpy, etc.) - I did not update them. Instead, I simply re-created the original development environment manually. To ensure backward compatibility I used an Ubuntu 16.04 LTS 64-bit environment. I did not use a virtual development environment.


### Built With

This repository contains an implementation of Yahoo's Open NSFW Classifier rewritten in tensorflow. The original caffe weights have been extracted using Caffe to TensorFlow. It also contains a Keras implementation of a CNN network for age and gender estimation, using the IMDB-WIKI dataset. And it performs face recognition using tensorflow and opencv using deep neural networks.  
  
The applications were built using the following modules installed via pip3:  
numpy (1.13.3), six (1.13.0), setuptools (42.0.2), mock (3.0.5), future (0.17.1), Keras (2.2.3), Keras-Applications (1.0.8), Keras-Preprocessing (1.1.0), enum34 (1.1.6), wheel (0.33.6), autograd (1.3), scikit-image (0.14.1), scikit-learn (0.15.0), scipy (1.1.0), matplotlib (3.0.0), PyWavelets (1.1.1), ipython (7.10.1), comb (0.9.10), tensorflow (1.12.3)  
  
Other modules installed via apt:  
git, python3-pip, pinta, OpenCV


### Installation Procedure

Interdependancies require the modules be installed in a specific order, so I suggest you follow the commands listed at the bottom of this ReadMe (Installation Procedure - Steps). Once complete, return to this section and continue.  
  
Clone this repository to your hard drive. Create a sub-folder called, "temp_files", in the Video_Analyzer folder. The models can be downloaded from my website - refer to the ReadMe files in the "model" and "pretrained_models" folders.  

For the application to identify faces, you need to create a folder called "ids". Then create one sub-folder for each of the actors in the "ids" folder and ensure that the sub-folder is named per the actor. Then put pictures of the actors in each of the respective sub-folders. For example, I created the folder "ids", then a sub-folder, "Eve". I then put pictures of Eve in that sub-folder.  


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
