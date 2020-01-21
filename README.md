# Video Analyzer And Cleaner

This repository consists of two programs:   analyze_video.py   and  clean_video.py

analyze_video.py - This application can be used to watch a video, frame by frame.  The application detects if the frame is safe for work or not (NSFW), and detects faces.  Per each face detected, the appication will predict the person's age, gender, and will try to identify them using an image library.

clean_video.py - Creates a copy of a video with any frames detected as NSFW removed from the copy.


## Environment

I did not update these applications to work with newer versions of the supporting modules (Tensorflow, etc.), so I did my best to re-create the necessary environment manually.  I built an Ubuntu 16.04 LTS 64-bit environment.  I did not build a virtual development environment.  Python 3 was used.


## Modules

The applications were built using the following modules:


absl-py (0.9.0), apturl (0.5.2), astor (0.8.0), autograd (1.3), blinker (1.3), Brlapi (0.6.4), comb (0.9.10), cryptography (1.2.3), cycler (0.10.0), defer (1.0.6), dlib (19.19.0), enum34 (1.1.6), feedparser (5.1.3), future (0.17.1), gast (0.3.2), grpcio (1.26.0), guacamole (0.9.2), h5py (2.10.0), html5lib (0.999), httplib2 (0.9.1), idna (2.0), ipython (7.10.1), Jinja2 (2.8), Keras (2.2.3), Keras-Applications (1.0.8), Keras-Preprocessing (1.1.0), kiwisolver (1.1.0), language-selector (0.1), louis (2.6.4), lxml (3.5.0), Mako (1.0.3), Markdown (3.1.1), MarkupSafe (0.23), matplotlib (3.0.0), mock (3.0.5), numpy (1.13.3), oauthlib (1.0.3), onboard (1.2.0), padme (1.1.1), pexpect (4.0.1), Pillow (3.1.2), plainbox (0.25), protobuf (3.11.1), psutil (5.6.7), ptyprocess (0.5), pyasn1 (0.1.9), pycups (1.9.73), pycurl (7.43.0), Pygments (2.1), pygobject (3.20.0), PyJWT (1.3.0), pyparsing (2.4.5), python-apt (1.1.0b1+ubuntu0.16.4.5), python-dateutil (2.8.1), python-debian (0.1.27), python-systemd (231), PyWavelets (1.1.1), pyxdg (0.25), PyYAML (5.2), reportlab (3.3.0), requests (2.9.1), scikit-image (0.14.1), scikit-learn (0.15.0), scipy (1.1.0), sessioninstaller (0.0.0), setuptools (42.0.2), six (1.13.0), system-service (0.3), tensorboard (1.12.2), tensorflow (1.12.3), termcolor (1.1.0), ufw (0.35), unity-scope-calculator (0.1), Werkzeug (0.16.0), wheel (0.33.6), xdiagnose (3.8.4.1), xkit (0.0.0), XlsxWriter (0.7.3)


## Installation Procedure

Clone this to your hard drive.  Interdependancies require a specific installation of the modules, so I suggest you follow the commands listed at the bottom of this ReadMe.md (Installation Procedure - Steps) to install the modules.

Create a you will need to download the weights and pre-trained models as described in the pretrained_models folder. 



### Usage

Analyze_Video.py - run this program using: python3 analyze_video.py {videoname.extension}

Clean_Video.py - run this program using: python3 clean_video.py {videoname.extension}


## Built With

This repository contains an implementation of Yahoo's Open NSFW Classifier rewritten in tensorflow. The original caffe weights have been extracted using Caffe to TensorFlow. You can find them at pretrained_models/open_nsfw-weights.npy.

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

Using two projects as a code base, I massaged them into this one new application.  I will update this later.

**Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

I will update this later.  

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

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
