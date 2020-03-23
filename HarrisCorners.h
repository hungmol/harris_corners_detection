#pragma once
// This program only use OpenCV datastructure for image I/O
// Mat data structure and show image. Nothing else

#include <iostream>
#include <exception>
#include <vector>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

#define K_HARRIS 0.04
#define WIN_W   5 // select the odd numbers
#define WIN_H	5

class HarrisCorners
{
public:
    HarrisCorners(string imagePath)
    {
        this->imagePath = imagePath;
    }
    HarrisCorners(){
	};
    bool calcHarrisRespone(const Mat M, float& R_score);
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
	vector<Point> cornesPoints;

	bool calcStructureTensor(const Mat Ix, const Mat Iy, Mat& M);
};
