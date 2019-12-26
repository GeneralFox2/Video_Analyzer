# Video Analyzer And Cleaner

Two programs:

Analyze_Video.py - Watch a video, frame by frame, as it detects whether the frame contains content that is NSFW and any visible faces.  Per each face detected, it will predict the person's age and gender.

Clean_Video.py - Creates a copy of a video with any frames detected as NSFW removed from the copy.

This repository contains an implementation of Yahoo's Open NSFW Classifier rewritten in tensorflow. The original caffe weights have been extracted using Caffe to TensorFlow. You can find them at pretrained_models/open_nsfw-weights.npy.

## Getting Started

You will need to download the weights and pre-trained models as described in the pretrained_models folder.  Links to be provided later.


### Prerequisites

I will provide a full listing of all of the prerequisites later.


### Usage

Analyze_Video.py - run this program using: python3 analyze_video.py {arguments}
Current arguments: videoname.format

Clean_Video.py - run this program using: python3 clean_video.py {arguments}
Current arguments: videoname.format


## Built With

I will update this later.

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

