#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

	// Load etalon images
	cv::Mat etalonRock;
	etalonRock = cv::imread("images/Rock.jpg", cv::IMREAD_COLOR);

	if (etalonRock.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::resize(etalonRock, etalonRock, cv::Size(), 0.15, 0.15);

	// Resize images to fit screen
	cv::Mat grayEtalonRock;
	cv::cvtColor(etalonRock, grayEtalonRock, cv::COLOR_BGR2GRAY);

	cv::Mat threshEtalonRock;

	cv::adaptiveThreshold(grayEtalonRock, threshEtalonRock, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);

	cv::bitwise_not(threshEtalonRock, threshEtalonRock);

	cv::GaussianBlur(threshEtalonRock, threshEtalonRock, cv::Size(3, 3), 0, 0);

	cv::imshow("gray", grayEtalonRock);
	cv::imshow("thresh", threshEtalonRock);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(threshEtalonRock, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);



	cv::Mat drawing = cv::Mat::zeros(threshEtalonRock.size(), CV_8UC3);

	int largest_area = 0;
	int largest_contour_index = 0;

	for (size_t i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);

		if (area > largest_area) {
			largest_area = area;
			largest_contour_index = i;
		}
	}

	cv::Scalar color = cv::Scalar(0, 0, 255);
	cv::drawContours(drawing, contours, largest_contour_index, color, 1, cv::LINE_8);

	imshow("Contours", drawing);

	cv::waitKey(0);

	return 0;
}