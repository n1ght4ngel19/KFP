#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

	// Load etalon images
	cv::Mat etalon, current;
	etalon = cv::imread("images/Paper.jpg", cv::IMREAD_COLOR);
	current = cv::imread("images/Rock.jpg", cv::IMREAD_COLOR);
	//current = cv::imread("images/Scissors.jpg", cv::IMREAD_COLOR);


	if (etalon.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Resize images to fit screen
	cv::resize(etalon, etalon, cv::Size(), 0.15, 0.15);
	cv::resize(current, current, cv::Size(), 0.15, 0.15);

	cv::Mat grayEtalon, grayCurrent;
	cv::cvtColor(etalon, grayEtalon, cv::COLOR_BGR2GRAY);
	cv::cvtColor(current, grayCurrent, cv::COLOR_BGR2GRAY);

	cv::Mat threshEtalon, threshCurrent;

	cv::adaptiveThreshold(grayEtalon, threshEtalon, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);
	cv::adaptiveThreshold(grayCurrent, threshCurrent, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);

	cv::bitwise_not(threshEtalon, threshEtalon);
	cv::bitwise_not(threshCurrent, threshCurrent);

	cv::stackBlur(threshEtalon, threshEtalon, cv::Size(3, 3));
	cv::stackBlur(threshCurrent, threshCurrent, cv::Size(3, 3));

	//cv::imshow("grayEtalon", grayEtalon);
	//cv::imshow("threshEtalon", threshEtalon);

	//cv::imshow("grayCurrent", grayCurrent);
	//cv::imshow("threshCurrent", threshCurrent);

	std::vector<std::vector<cv::Point>> etalonContours, currentContours;
	cv::findContours(threshEtalon, etalonContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(threshCurrent, currentContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat etalonDrawing = cv::Mat::zeros(threshEtalon.size(), CV_8UC3);
	cv::Mat currentDrawing = cv::Mat::zeros(threshCurrent.size(), CV_8UC3);

#pragma region Etalon Image Contour
	int largestAreaEtalon = 0;
	int largestContourIdxEtalon = 0;

	for (size_t i = 0; i < etalonContours.size(); i++) {
		double area = cv::contourArea(etalonContours[i]);

		if (area > largestAreaEtalon) {
			largestAreaEtalon = area;
			largestContourIdxEtalon = i;
		}
	}
#pragma endregion

#pragma region Current Image Contour
	int largestAreaCurrent = 0;
	int largestContourIdxCurrent = 0;

	for (size_t i = 0; i < currentContours.size(); i++) {
		double area = cv::contourArea(currentContours[i]);

		if (area > largestAreaCurrent) {
			largestAreaCurrent = area;
			largestContourIdxCurrent = i;
		}
	}
#pragma endregion

	// Make sure the contour is thick enough
	// If line width is too small, parallel contours will ruin the match
	int lineWidth = 8;

	cv::Scalar color = cv::Scalar(0, 0, 255);
	cv::drawContours(etalonDrawing, etalonContours, largestContourIdxEtalon, color, lineWidth, cv::LINE_8);
	cv::cvtColor(etalonDrawing, etalonDrawing, cv::COLOR_BGR2GRAY);

	cv::Scalar color2 = cv::Scalar(0, 255, 0);
	cv::drawContours(currentDrawing, currentContours, largestContourIdxCurrent, color2, lineWidth, cv::LINE_8);
	cv::cvtColor(currentDrawing, currentDrawing, cv::COLOR_BGR2GRAY);

	// Calculate match
	double match = cv::matchShapes(etalonDrawing, currentDrawing, cv::CONTOURS_MATCH_I2, 0);
	std::cout << "Match: " << match << std::endl;

	// Decide and print the result
	std::string result = match < .15 ? "It's a match!" : "They aren't the same!";
	std::cout << result << std::endl;

	imshow("etalonDrawing", etalonDrawing);
	imshow("currentDrawing", currentDrawing);

	cv::waitKey(0);

	return 0;
}