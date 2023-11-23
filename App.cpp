#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

	cv::Mat image;
	image = cv::imread("images/IMG_20231123_233211.jpg", cv::IMREAD_COLOR);

	if (image.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::resize(image, image, cv::Size(), 0.15, 0.15);

	cv::imshow("img", image);

	cv::waitKey(0);

	return 0;
}