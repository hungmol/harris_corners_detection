#include "HarrisCorners.h"

bool HarrisCorners::rgb2gray(Mat inputImage, Mat &outputImage)
{
    if (!inputImage.data)
    {
        cerr << "[ERROR] Invalid input image\n";
        return false;
    }
    double WR = 0.299;
    double WB = 0.587;
    double WG = 1 - (WR + WB);

    try
    {
        int width = inputImage.cols;
        int height = inputImage.rows;

        if (inputImage.channels() == 1)
        {
            outputImage = inputImage.clone();
        }
        else if (inputImage.channels() == 3)
        {
            outputImage = Mat(Size(inputImage.cols, inputImage.rows), CV_8UC1);
            for (uint y = 0; y < inputImage.rows; y++)
            {
                uchar *ptr = outputImage.data + y * inputImage.cols;
                for (uint x = 0; x < inputImage.cols; x++, ptr++)
                {
                    uchar r = inputImage.data[3 * (width * y + x) + 2];
                    uchar g = inputImage.data[3 * (width * y + x) + 1];
                    uchar b = inputImage.data[3 * (width * y + x) + 0];
                    *ptr = (uchar)(r * WR + g * WG + b * WB);
                }
            }
        }

        return true;
    }
    catch (cv::Exception &err)
    {
        cerr << "[ERROR] " << err.what() << endl;
    }
    return false;
}

bool HarrisCorners::sobelFilter(Mat grayImage, Mat &gradImageX, Mat &gradImageY)
{
    if (!grayImage.data)
    {
        cerr << "[ERROR] Invalid input image\n";
        return false;
    }
    // To-do: Implement sobel in seperable x and y
    // Sobel kernel
	float Gx[9] = {1.0, 0.0, -1.0,
				   2.0, 0.0, -2.0,
				   1.0, 0.0, -1.0};
	float Gy[9] = {1.0, 2.0, 1.0,
				   0.0, 0.0, 0.0,
				   -1.0, -2.0, -1.0};

    int width = grayImage.cols;
    int height = grayImage.rows;
    gradImageX = Mat(Size(width, height), CV_32FC1);
    gradImageY = Mat(Size(width, height), CV_32FC1);

    int padOffset = 2;
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            if (i > padOffset && i < height - padOffset && j > padOffset && j < width - padOffset)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        // flipped kernel
                        int xn = j + k - 1;
                        int yn = i + l - 1;
                        int index = xn + yn * width;
                        gradImageX.data[i * width + j] += Gx[3 * k + l] * grayImage.data[index];
                        gradImageY.data[i * width + j] += Gy[3 * k + l] * grayImage.data[index];
                    }
                }
            }
        }
    }
    return true;
}

bool HarrisCorners::computeCorners()
{
    // Step 1 - Load and convert image color (3 channels) -> gray
    if (isFileExist(this->imagePath))
    {
        this->image = imread(this->imagePath, CV_LOAD_IMAGE_COLOR);
        if (!this->image.data)
        {
            cerr << "[ERROR] Could not load the image\n";
            return false;
        }
        if (this->image.channels() == 3)
        {
            rgb2gray(this->image, this->grayImage);
        }
    }
    int width = this->grayImage.cols;
    int height = this->grayImage.rows;
    cv::imshow("test", this->grayImage);

    // Step 2: Harris response calculation - the image gradient intensity - image derivatives
    // Using Sobel filter float version
    sobelFilter(this->grayImage, this->intensityGradX, this->intensityGradY);

    // Step 3: Compute the structure tensor - M matrix
    this->structureTensor = Mat::zeros(Size(2, 2), CV_32FC1);
    
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            this->structureTensor.at<float>(0, 0) += pow(this->intensityGradX.data[x + y*width], 2); 
            this->structureTensor.at<float>(0, 1) += this->intensityGradX.data[x + y*width] * this->intensityGradY.data[x + y*width];
            this->structureTensor.at<float>(1, 0) += this->structureTensor.at<float>(0, 1);
            this->structureTensor.at<float>(1, 1) += pow(this->intensityGradY.data[x + y*width], 2); 
        }
    }
    cout << this->structureTensor << endl;
    

}

bool HarrisCorners::responseCalculation(){
    
}

bool HarrisCorners::isFileExist(string filePath)
{
    try
    {
        fs::path pathObj(filePath);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            return true;
        }
    }
    catch (exception &err)
    {
        cerr << "[ERROR] " << err.what() << endl;
    }
    return false;
}

bool HarrisCorners::isPathExist(string fileDir)
{
    try
    {
        fs::path pathObj(fileDir);
        if (fs::exists(pathObj) && fs::is_directory(pathObj))
        {
            return true;
        }
    }
    catch (exception &err)
    {
        cerr << "[ERROR] " << err.what() << endl;
    }
    return false;
}

int main()
{
    cout << "hello world!";
    HarrisCorners cornersDetection("test.jpg");
    cornersDetection.computeCorners();
    return 0;
}