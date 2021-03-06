# medical-plotting-tools
A collection of personal plotting code for all kinds of projects

### Important notes
Still very unclean, just a very long Jupyter Notebook that uses the functions contained in `plotting.py` to export images for an experiment we ran (demo image below).

Import `plotting.py` to make use of the functions, they should be pretty straightforward.

Will rebuild this repo with proper functions you can actually use

#### How-To
Tested on Python >= 3.7
To run:
* clone this repo
* install requirements: `pip install -r requirements.txt`

#### Features
Can plot overlays for segmentations such as this one:

![Image of Segmentation](https://raw.githubusercontent.com/christophbrgr/medical-plotting-tools/master/img/demo_t1.png)

Other features:
* automatic slice selection (e.g. the center of mass of the tumor)
* fully customizable color specification for each label
* padding of images to final shape
* saving images with or without overlay
* and much more..

Written during my time at IBBM Group Munich, Germany and CIBL at Dana-Farber Cancer Institute in Boston, MA. The responsibility for the contents of this repository however remain solely with me as this is only a utilities repository.
