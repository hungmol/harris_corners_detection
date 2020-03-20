#pragma once
// This program only use OpenCV datastructure for image I/O
// Mat data structure and show image. Nothing else

#include <iostream>
#include <exception>
#include <vector>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

#define K_HARRIS 0.04

class HarrisCorners
{
public:
    HarrisCorners(string imagePath)
    {
        this->imagePath = imagePath;
    }
    HarrisCorners(){};
    bool responseCalculation();
    bool computeCorners();
    bool isFileExist(string filePath);
    bool isPathExist(string fileDir);
    bool rgb2gray(Mat inputImage, Mat &outputImage);
    bool sobelFilter(Mat grayImage, Mat &gradImageX, Mat &gradImageY);

private:
    string imagePath;
    Mat image;
    Mat grayImage;
    Mat intensityGradX;
    Mat intensityGradY;
    Mat structureTensor;
    double score_R;
};
