#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
		if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
      	matcher =  cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

      	int k = 2;
      	std::vector<vector<cv::DMatch>> mt;
      
      	double t = (double)cv::getTickCount();
      	
      	matcher->cv::DescriptorMatcher::knnMatch(descSource, descRef, mt, k);
      	
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
      	
      	const float threshold = 0.8;
      	for (int i = 0; i < mt.size(); ++i)
        {
          if (mt[i][0].distance < mt[i][1].distance * threshold) {matches.push_back(mt[i][0]);}
        }
      	
      	return;
    }
  	//matcher->match(descSource, descRef, matches);
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, float &time_ms)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
  	std::map<std::string, int> extractor_type;
  	extractor_type.insert(std::make_pair("BRISK", 0));
  	extractor_type.insert(std::make_pair("BRIEF", 1));
  	extractor_type.insert(std::make_pair("ORB", 2));
  	extractor_type.insert(std::make_pair("FREAK", 3));
  	extractor_type.insert(std::make_pair("AKAZE", 4)); 
	extractor_type.insert(std::make_pair("SIFT", 5));
  
  	switch (extractor_type.find(descriptorType)->second)
    {
      case 0: // BRISK
        {
          int threshold = 30;        // FAST/AGAST detection threshold score.
          int octaves = 3;           // detection octaves (use 0 to do single scale)
          float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
          extractor = cv::BRISK::create(threshold, octaves, patternScale);
          break;
        }
      case 1: // BRIEF
        {
          int  bytes = 32;
          bool use_orientation = false;
          extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
          break;
        }
      case 2: // ORB
        {
          int nfeatures = 500;
          float scaleFactor = 1.2f; 
          int nlevels = 8;
          int edgeThreshold = 31;
          int firstLevel = 0;
          int WTA_K = 2;
          cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
          int patchSize = 31;
          int fastThreshold = 20;
          extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
          break;
        }
      case 3: // FREAK
        {
          bool orientationNormalized = true;
          bool scaleNormalized = true;
          float patternScale = 22.0f;
          int nOctaves = 4;
          extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
          break;
        }
      case 4: // AKAZE
        {
          cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
          int  descriptor_size = 0;
          int  descriptor_channels = 3;
          float  threshold = 0.001f;
          int  nOctaves = 4;
          int  nOctaveLayers = 4;
          cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
          extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
          break;
        }
      case 5: // SIFT
        {
          int  nfeatures = 0;
          int  nOctaveLayers = 3;
          double  contrastThreshold = 0.04;
          double  edgeThreshold = 10;
          double  sigma = 1.6;
          extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
          break;
        }
      default:
        cout << "The extractor is not determined!" << endl;
        break;
    }
  
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  	
  	time_ms = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  	time_ms = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	int blockSize = 4;
  	int ksize = 3;
  	double k = 0.04; // harris free parameter
  	int borderType = cv::BorderTypes::BORDER_DEFAULT;
  	cv::Mat outHarris;
  	int thresh = 100;
  
  	/* apply Harris keypoints detector */
  	double t = (double)cv::getTickCount();
  	cv::cornerHarris(img, outHarris, blockSize, ksize, k, borderType);
  	cv::normalize( outHarris, outHarris, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
  
  	/* get keypoints */
  	for (int i = 0; i < outHarris.rows; ++i)
    {
      	for (int j = 0; j < outHarris.cols; ++j)
        {
          	if (outHarris.at<float>(i,j) >= thresh)
            {
              	cv::KeyPoint newKeyPoint;
              	newKeyPoint.pt = cv::Point2f((float)j, (float)i);
        		newKeyPoint.size = blockSize;
        		keypoints.push_back(newKeyPoint);
            }
        }
    }
  
  	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  	time_ms = 1000 * t / 1.0;
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        
      
    }
  	
  
}

void detKeypointsFast(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	int threshold = 100;
  	bool nonmaxSuppression = false;
  	//int type = cv::FastFeatureDetector::TYPE_9_16;
  	//int type = cv::FastFeatureDetector::TYPE_7_12;
  	//int type = cv::FastFeatureDetector::TYPE_5_8;
  
    /* apply FAST keypoints detector */
  	double t = (double)cv::getTickCount();
  	cv::FAST(img, keypoints, threshold, nonmaxSuppression);
  
  	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  	time_ms = 1000 * t / 1.0;
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  	// visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Fast Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
    }
}

void detKeypointsBrisk(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	cv::Ptr<cv::BRISK> detector;
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
    detector = cv::BRISK::create(threshold, octaves, patternScale);
  
  	/* apply BRISK keypoints detector */
  	double t = (double)cv::getTickCount();
  	detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  	time_ms = 1000 * t / 1.0;
    cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
    }
  
}

void detKeypointsOrb(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	cv::Ptr<cv::ORB> detector;
  	int nfeatures = 500;
  	float scaleFactor = 1.2f; 
  	int nlevels = 8;
  	int edgeThreshold = 31;
  	int firstLevel = 0;
  	int WTA_K = 2;
  	cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
  	int patchSize = 31;
    int fastThreshold = 20;
  	
  	detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  	
  	/* apply ORB keypoints detector */
  	double t = (double)cv::getTickCount();
  	detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_ms = 1000 * t / 1.0;
    cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  	if (bVis)
      {
          cv::Mat visImage = img.clone();
          cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          string windowName = "ORB Detector Results";
          cv::namedWindow(windowName, 6);
          imshow(windowName, visImage);
      }
}

void detKeypointsAkaze(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	cv::Ptr<cv::AKAZE> detector;
    cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int  descriptor_size = 0;
    int  descriptor_channels = 3;
    float  threshold = 0.001f;
    int  nOctaves = 4;
    int  nOctaveLayers = 4;
    cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
  
  	detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
  
  /* apply AKAZE keypoints detector */
  	double t = (double)cv::getTickCount();
  	detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_ms = 1000 * t / 1.0;
    cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  	if (bVis)
      {
          cv::Mat visImage = img.clone();
          cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          string windowName = "AKAZE Detector Results";
          cv::namedWindow(windowName, 6);
          imshow(windowName, visImage);
      }
  	
}

void detKeypointsSift(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time_ms, bool bVis)
{
  	cv::Ptr<cv::xfeatures2d::SIFT> detector;
  	int  nfeatures = 0;
	int  nOctaveLayers = 3;
	double  contrastThreshold = 0.04;
	double  edgeThreshold = 10;
	double  sigma = 1.6;
    detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
 	
  	/* apply SIFT keypoints detector */
  	double t = (double)cv::getTickCount();
  	detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_ms = 1000 * t / 1.0;
    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  	if (bVis)
      {
          cv::Mat visImage = img.clone();
          cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          string windowName = "SIFT Detector Results";
          cv::namedWindow(windowName, 6);
          imshow(windowName, visImage);
      }
  	
}