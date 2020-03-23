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
	try {
		int width = grayImage.cols;
		int height = grayImage.rows;
		gradImageX = Mat(Size(width, height), CV_16S);
		gradImageY = Mat(Size(width, height), CV_16S);
		double value_gx = 0.0;
		double value_gy = 0.0;

		int padOffset = 0;
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				value_gx = 0.0;
				value_gy = 0.0;
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
							value_gx += Gx[3 * k + l] * grayImage.data[index];
							value_gy += Gy[3 * k + l] * grayImage.data[index];
						}
					}
				}
				gradImageX.data[i * width + j] =(int)value_gx;
				gradImageY.data[i * width + j] = (int)value_gy;
			}
		}
		return true;
	}
	catch (cv::Exception& err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool HarrisCorners::computeCorners()
{
    // Step 1 - Load and convert image color (3 channels) -> gray
    if (isFileExist(this->imagePath))
    {
		//if (CV_MAJOR_VERSION < 4) {
		//	this->image = imread(this->imagePath, CV_LOAD_IMAGE_COLOR);
		//}
		//else {
			this->image = imread(this->imagePath, IMREAD_COLOR);
		//}
        
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

    // Step 2: Harris response calculation - the image gradient intensity - image derivatives
    // Using Sobel filter float version
	//if (!sobelFilter(this->grayImage, this->intensityGradX, this->intensityGradY))
	//	return false;


    // Step 3: Compute the structure tensor - M matrix
	try {
		Sobel(this->grayImage, this->intensityGradX, CV_16S, 1, 0, 3);
		Sobel(this->grayImage, this->intensityGradY, CV_16S, 0, 1, 3);

		// Structure tensor
		Mat IxIx = Mat::zeros(this->grayImage.size(), CV_16S);
		Mat IyIy = Mat::zeros(this->grayImage.size(), CV_16S);
		Mat IxIy = Mat::zeros(this->grayImage.size(), CV_16S);

		//for (int y = 0; y < height; y++) {
		//	for (int x = 0; x < width; x++) {
		//		IxIx.data[x + y * width] = pow(this->intensityGradX.data[x + y * width], 2);
		//		IyIy.data[x + y * width] = pow(this->intensityGradY.data[x + y * width], 2);
		//		IxIy.data[x + y * width] = this->intensityGradX.data[x + y * width] * this->intensityGradY.data[x + y * width];
		//	}
		//}
		IxIx = this->intensityGradX.mul(this->intensityGradX);
		IyIy = this->intensityGradY.mul(this->intensityGradY);
		IxIy = this->intensityGradX.mul(this->intensityGradY);

		// Using gaussian window function 
		Mat G_IxIx, G_IyIy, G_IxIy;
		GaussianBlur(IxIx, G_IxIx, Size(5, 5), 1, 1);
		GaussianBlur(IyIy, G_IyIy, Size(5, 5), 1, 1);
		GaussianBlur(IxIy, G_IxIy, Size(5, 5), 1, 1);

		// Harris Respone Calculation - get the interest windows
		Mat R = Mat::zeros(IxIx.size(), IxIx.type());

		for (int y = 2; y < height - 2; y++) {
			for (int x = 2; x < width - 2; x++) {
				float det_M = G_IxIx.at<int>(y, x) * G_IyIy.at<int>(y, x) - pow(G_IxIy.at<int>(y, x), 2);
				float tr_M = G_IxIx.at<int>(y, x) + G_IyIy.at<int>(y, x);
				float tmp = det_M - K_HARRIS * tr_M * tr_M;
				if (tmp > 0.0f) {
					cout << "\nR score = " << tmp 
						<< " - det M = " << det_M << " - tr_M = " << tr_M << endl;
					cout << "Ixx = " << G_IxIx.at<int>(y, x) << " - IyIy = " << G_IyIy.at<int>(y, x) << " - IxIy = " << G_IxIy.at<int>(y, x) <<endl;
					cout << "grad x = " << this->intensityGradX.at<int>(y, x) << " - grad y = " << this->intensityGradY.at<int>(y, x) << endl;
					R.at<int>(y, x) = tmp;
				}
				else {
					R.at<int>(y, x) = 0;
				}
			}
		}
		// Non-max supprestion with 3x3 filter
		int offset_x = (int)(WIN_W / 2);
		int offset_y = (int)(WIN_H / 2);
		for (int y = offset_y; y < height - offset_y; y += offset_y) {
			for (int x = offset_x; x < width - offset_x; x += offset_x) {
				//find max
				float max_val = 0;
				for (int j = y - offset_y; j < y + offset_y; j++) {
					for (int i = x - offset_x; i < x + offset_x; i++) {
						if (max_val < R.at<int>(j, i)) {
							max_val = R.at<int>(j, i);
						}
					}
				}

				// Remove non-max
				for (int j = y - offset_y; j < y + offset_y; j++) {
					for (int i = x - offset_x; i < x + offset_x; i++) {
						if (R.at<int>(j, i) < max_val) {
							R.at<int>(j, i) = 0.0;
						}
						else if (R.at<int>(j, i) >= max_val){
							//add to vector corners points
							this->cornesPoints.push_back(Point(i, j));
						}
					}
				}
			}
		}
		for (int i = 0; i < this->cornesPoints.size(); i++) {
			circle(this->image, this->cornesPoints[i], 3, Scalar(0, 255, 0), 2);
			imshow("test", this->image);
			waitKey(30);
		}
		cout << "Number of points = " << this->cornesPoints.size() << endl;
		
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return true;
}

bool HarrisCorners::calcStructureTensor(const Mat Ix, const Mat Iy, Mat &M) {
	if (!Ix.data || !Iy.data) {
		cerr << "[ERROR] Invalid input\n";
		return false;
	}
	int width = Ix.cols;
	int height = Ix.rows;
	M = Mat(Size(2, 2), CV_32F);
	try {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				M.at<float>(0, 0) += pow(Ix.data[x + y * width], 2);
				M.at<float>(0, 1) += Ix.data[x + y * width] * Iy.data[x + y * width];
				M.at<float>(1, 0) += M.at<float>(0, 1);
				M.at<float>(1, 1) += pow(Iy.data[x + y * width], 2);
			}
		}
		return true;
	}
	catch (cv::Exception& err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool HarrisCorners::calcHarrisRespone(const Mat M, float &R_score){
	float tr_M = M.at<float>(0, 0) + M.at<float>(1, 1);
	float det_M = M.at<float>(0, 0) * M.at<float>(1, 1) - pow(M.at<float>(0, 1), 2);
	R_score = det_M - K_HARRIS * tr_M * tr_M;
	return true;
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
    HarrisCorners cornersDetection("test.png");
    cornersDetection.computeCorners();
    return 0;
}