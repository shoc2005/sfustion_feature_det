/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
  
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results
  
  	std::map<std::string, int> mapKeyPointDetector;
  	mapKeyPointDetector.insert(std::make_pair("SHITOMASI", 0));
  	mapKeyPointDetector.insert(std::make_pair("HARRIS", 1));
  	mapKeyPointDetector.insert(std::make_pair("FAST", 2));
  	mapKeyPointDetector.insert(std::make_pair("BRISK", 3));
  	mapKeyPointDetector.insert(std::make_pair("ORB", 4)); 
	mapKeyPointDetector.insert(std::make_pair("AKAZE", 5));
  	mapKeyPointDetector.insert(std::make_pair("SIFT", 6));
  
  	/* use the commandline's parameters */
  	string cm_detector;
  	string cm_descriptor;
  	string TASKID = "MP8"; // MP7, MP8, MP9
  	bool cm_used = false;
  	ofstream repf;
  	if (argc >1 && argc <= 3)
    {
      cm_used = true;
      repf.open("summary.txt", std::ofstream::app);
      
      if (((string)argv[1]).compare("MP7") == 0) // count only number of keypoints for all keypoint detectors.
      {
        repf << "\nMP7: " <<  endl;
        cm_descriptor = "BRISK";
        repf << "| ImageID | Keypoints | Detector |" << endl;
        TASKID = "MP7";
      } else if (((string)argv[1]).compare("MP8") == 0) // count matched keypoints for all combinations of keypoint detector and descriptors
      { 
        cm_descriptor = (string)argv[2];
        repf << "\nMP8, Descriptor: " << cm_descriptor << endl;
        repf << "| ImageID | Matched Points | Keypoint Detector |" << endl;
        TASKID = "MP8";
      } else // task MP9 // measure keypoint detectors and descriptors processing time
      {
        cm_descriptor = (string)argv[2];
        repf << "\nMP9, Descriptor with timing:" << cm_descriptor << endl;
        repf << "| ImageID | Keypoint Detector | KeyPoints Time | Desription Time |" << endl;
        TASKID = "MP9";
      }
      
      
    } else // use the default (SHITOMASI) keypoint detector by removing all remaining detectors from the map
    {
      while (mapKeyPointDetector.size() != 1)
      {
        for (std::map<std::string, int>::iterator it_detector=mapKeyPointDetector.begin(); it_detector!=mapKeyPointDetector.end(); ++it_detector) 
        {
          if (it_detector->first.compare("SHITOMASI") != 0) // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
          {
            mapKeyPointDetector.erase(it_detector);
          }
        }
      }
    }
    /* LOOP OVER ALL KEYPOINT DETECTORS */
  
  	for (std::map<std::string, int>::iterator it_detector=mapKeyPointDetector.begin(); it_detector!=mapKeyPointDetector.end(); ++it_detector)
    {
	dataBuffer.clear();
	 /* MAIN LOOP OVER ALL IMAGES */
      for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
      {
          /* LOAD IMAGE INTO BUFFER */

          // assemble filenames for current index
          ostringstream imgNumber;
          imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
          string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

          // load image from file and convert to grayscale
          cv::Mat img, imgGray;
          img = cv::imread(imgFullFilename);
          cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

          //// STUDENT ASSIGNMENT
          //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

          // push image into data frame buffer
          DataFrame frame;
          frame.cameraImg = imgGray;
          if (dataBuffer.size() >= dataBufferSize)
          {
            dataBuffer.erase(dataBuffer.begin());
          }
          dataBuffer.push_back(frame);

          //// EOF STUDENT ASSIGNMENT
          cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

          /* DETECT IMAGE KEYPOINTS */

          // extract 2D keypoints from current image
          vector<cv::KeyPoint> keypoints; // create empty feature list for current image
          string detectorType = it_detector->first;

          //// STUDENT ASSIGNMENT
          //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
          //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
          float t_detector =0.0;

          switch (mapKeyPointDetector.find(detectorType)->second)
          {
              case 0:
                  detKeypointsShiTomasi(keypoints, imgGray, t_detector, bVis);
                  break;
              case 1:
                  detKeypointsHarris(keypoints, imgGray, t_detector, bVis);
                  break;
              case 2:
                  detKeypointsFast(keypoints, imgGray, t_detector, bVis);
                  break;
              case 3:
                  detKeypointsBrisk(keypoints, imgGray, t_detector, bVis);
                  break;
              case 4:
                  detKeypointsOrb(keypoints, imgGray, t_detector, bVis);
                  break;
              case 5:
                  detKeypointsAkaze(keypoints, imgGray, t_detector, bVis);
                  break;
              case 6:
                  detKeypointsSift(keypoints, imgGray, t_detector, bVis);
                  break;         
              default:
                  cout << "Detector is not determined!" << endl;
              	  return 0;
                  break;

          }
          
          //// EOF STUDENT ASSIGNMENT

          //// STUDENT ASSIGNMENT
          //// TASK MP.3 -> only keep keypoints on the preceding vehicle

          // only keep keypoints on the preceding vehicle
          bool bFocusOnVehicle = true;
          cv::Rect vehicleRect(535, 180, 180, 150);

          vector<cv::KeyPoint> filtered;
          if (bFocusOnVehicle)
          {
              for (auto& keypoint: keypoints)
              {
                  if (vehicleRect.contains((&keypoint)->pt)
                      //((&keypoint)->pt.x >= vehicleRect.x) && ((&keypoint)->pt.x <= (vehicleRect.x + vehicleRect.width)) &&
                      //((&keypoint)->pt.y >= vehicleRect.y) && ((&keypoint)->pt.y <= (vehicleRect.y + vehicleRect.height))
                      )
                  {
                    filtered.push_back(keypoint);
                  }
              }
              keypoints = filtered;
          }



          //// EOF STUDENT ASSIGNMENT

          // optional : limit number of keypoints (helpful for debugging and learning)
          bool bLimitKpts = false;
          if (bLimitKpts)
          {
              int maxKeypoints = 50;

              if (detectorType.compare("SHITOMASI") == 0)
              { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                  keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
              }
              cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
              cout << " NOTE: Keypoints have been limited!" << endl;
          }

          // push keypoints and descriptor for current frame to end of data buffer
          (dataBuffer.end() - 1)->keypoints = keypoints;
          cout << "#2 : DETECT KEYPOINTS done" << endl;
        
          if (TASKID.compare("MP7") == 0 && cm_used)
              {
               	repf << "| " << imgIndex << " | " << keypoints.size() << " | " << it_detector->first << " |" << endl;
              }

          /* EXTRACT KEYPOINT DESCRIPTORS */

          //// STUDENT ASSIGNMENT
          //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
          //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

          cv::Mat descriptors;
          string descriptorType = cm_used ? cm_descriptor : "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
          float t_descriptor=0.0;
        
          try
          {
          descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, t_descriptor);
          } catch(const std::exception& e)
          {
             if (TASKID.compare("MP8") == 0 && cm_used)
              {
                if (imgIndex != 0)
                {
               		repf << "|" << imgIndex << " | " << "Uncomp." << " | " << it_detector->first << endl;
                }
              }
            
            if (TASKID.compare("MP9") == 0 && cm_used)
              {
                if (imgIndex != 0)
                {
               		repf << "|" << imgIndex << " | " << it_detector->first << " | - | - |" << endl;
                }
              }
             cout << "Can`t make descriptors" << endl;
             continue;
          }
          if (TASKID.compare("MP9") == 0 && cm_used)
              {
                repf << "| " << imgIndex << " | " << it_detector->first << " | " << t_detector << " | " << t_descriptor << " |" << endl;
              }
        
          //// EOF STUDENT ASSIGNMENT

          // push descriptors for current frame to end of data buffer
          (dataBuffer.end() - 1)->descriptors = descriptors;

          cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

          if (dataBuffer.size() > 1) // wait until at least two images have been processed
          {

              /* MATCH KEYPOINT DESCRIPTORS */

              vector<cv::DMatch> matches;
              string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
              string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
              string selectorType;
              if (cm_used && (TASKID.compare("MP8") == 0 || TASKID.compare("MP9") == 0))
              {
              	selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
              } else
              {
                selectorType = "SEL_NN";
              }

              //// STUDENT ASSIGNMENT
              //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
              //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

              matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                               (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                               matches, descriptorType, matcherType, selectorType);

              //// EOF STUDENT ASSIGNMENT

              // store matches in current data frame
              (dataBuffer.end() - 1)->kptMatches = matches;

              cout << "\t #4 : MATCH KEYPOINT DESCRIPTORS done: " <<  matches.size() << endl;
              if (TASKID.compare("MP8") == 0 && cm_used)
              {
                repf << "|" << imgIndex << " | " << matches.size() << " | " << it_detector->first << " |" << endl;
              }

              

              // visualize matches between current and previous image
              //bVis = true;
              if (bVis)
              {
                  cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                  cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                  (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                  matches, matchImg,
                                  cv::Scalar::all(-1), cv::Scalar::all(-1),
                                  vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                  string windowName = "Matching keypoints between two camera images";
                  cv::namedWindow(windowName, 7);
                  cv::imshow(windowName, matchImg);
                  cout << "Press key to continue to next image or ESC to quit"  << endl;
                  int key_pressed = cv::waitKey(0) & 255;
                  if (key_pressed == 27)
                  {
                      //cv::destroyAllWindows();
                      break;
                  }
              }
              //bVis = false;
          }

      } // eof loop over all images
      
    } // eof loop over all detectors
  	if (cm_used) repf.close();

    return 0;
}
