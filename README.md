# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

This project is a part of the Sensor Fusion nanodagree program.

The project provides source codes for applying different types of the keypoint detectors and the descriptor extractors.

The summary results are provided in the file 'summary.pdf' accordingly to raw data in 'src/summary.txt' and raw data's processing file is provided in 'preproc.xlsx'.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Run instructions

For gathering statistics run: `../src/run_stat.sh`.

For getting statistics of the number of detected keypoints run:`./2D_feature_tracking MP7`

For getting statistics of the matched keypoints use:`./2D_feature_tracking MP8 DESCRIPTOR_NAME` (you should select one of the: BRISK, BRIEF, ORB, FREAK, AKAZE, SIF)

For getting statistics of the matching time and keypoints detection time use: `./2D_feature_tracking MP9 DESCRIPTOR_NAME` (you should chose one of the: BRISK, BRIEF, ORB, FREAK, AKAZE, SIF)
