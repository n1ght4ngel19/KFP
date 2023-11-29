#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat colorRead(std::string directory, std::string fileName) {
	std::string path = "images/" + directory + "/" + fileName + ".jpg";

	std::cout << "Reading " << path << std::endl;

	cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

	cv::waitKey(0);

	if (image.empty()) {
		std::cout << "Could not open or find the image (Reading)" << std::endl;

		return cv::Mat();
	}

	return image;
}

int findLargestContour(std::vector<std::vector<cv::Point>> contours) {
	int largestArea = 0;
	int largestContourIdx = 0;

	for (size_t i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);

		if (area > largestArea) {
			largestArea = area;
			largestContourIdx = i;
		}
	}

	return largestContourIdx;
}

cv::Mat makeDrawing(cv::Mat image, std::string drawingName, bool doResize, cv::Scalar color) {
	if (doResize) {
		cv::resize(image, image, cv::Size(), 0.15, 0.15);
	}

	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat threshImage;
	cv::adaptiveThreshold(grayImage, threshImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);

	cv::bitwise_not(threshImage, threshImage);

	cv::stackBlur(threshImage, threshImage, cv::Size(3, 3));

	//cv::imshow(drawingName + " gray", grayImage);
	//cv::imshow(drawingName + " thresh", threshImage);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(threshImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat drawing = cv::Mat::zeros(threshImage.size(), CV_8UC1);

	int largestContourIdx = findLargestContour(contours);

	cv::drawContours(drawing, contours, largestContourIdx, color, 8, cv::LINE_8);

	cv::imshow(drawingName, drawing);

	return drawing;
}

bool isMatch(cv::Mat drawing1, cv::Mat drawing2, float threshold) {
	float match = cv::matchShapes(drawing1, drawing2, cv::CONTOURS_MATCH_I2, 0);
	std::cout << "Match (0-1): " << match << std::endl;

	return match < threshold;
}

int main() {
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

	std::string basePath = "images/base/";
	std::string gesturesPath = "images/gestures/";

	// Base images for comparison
	cv::Mat rock, paper, scissors;

	std::cout << "Reading " << basePath << "paper.jpg" << std::endl;

	cv::Mat basePaper = cv::imread(basePath + "paper.jpg", cv::IMREAD_COLOR);
	cv::Mat baseRock = cv::imread(basePath + "rock.jpg", cv::IMREAD_COLOR);
	cv::Mat baseScissors = cv::imread(basePath + "scissors.jpg", cv::IMREAD_COLOR);

	// Checking for reading error
	if (basePaper.empty()) {
		std::cout << "Could not open or find the image paper" << std::endl;
	}
	if (baseRock.empty()) {
		std::cout << "Could not open or find the image rock" << std::endl;
	}
	if (baseScissors.empty()) {
		std::cout << "Could not open or find the image scissors" << std::endl;
	}

	cv::resize(basePaper, basePaper, cv::Size(), 0.15, 0.15);
	cv::resize(baseRock, baseRock, cv::Size(), 0.15, 0.15);
	cv::resize(baseScissors, baseScissors, cv::Size(), 0.15, 0.15);

	// Converting to grayscale
	cv::Mat grayPaper, grayRock, grayScissors;

	cv::cvtColor(basePaper, grayPaper, cv::COLOR_BGR2GRAY);
	cv::cvtColor(baseRock, grayRock, cv::COLOR_BGR2GRAY);
	cv::cvtColor(baseScissors, grayScissors, cv::COLOR_BGR2GRAY);

	// Thresholding images
	cv::Mat threshPaper, threshRock, threshScissors;

	cv::adaptiveThreshold(grayPaper, threshPaper, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);
	cv::adaptiveThreshold(grayRock, threshRock, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);
	cv::adaptiveThreshold(grayScissors, threshScissors, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);

	// Final touches setting up images for contouring
	cv::bitwise_not(threshPaper, threshPaper);
	cv::bitwise_not(threshRock, threshRock);
	cv::bitwise_not(threshScissors, threshScissors);

	cv::stackBlur(threshPaper, threshPaper, cv::Size(3, 3));
	cv::stackBlur(threshRock, threshRock, cv::Size(3, 3));
	cv::stackBlur(threshScissors, threshScissors, cv::Size(3, 3));

	// Creating contours
	std::vector<std::vector<cv::Point>> paperContours, rockContours, scissorsCotours;

	cv::findContours(threshPaper, paperContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(threshRock, rockContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(threshScissors, scissorsCotours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat paperDrawing = cv::Mat::zeros(threshPaper.size(), CV_8UC1);
	cv::Mat rockDrawing = cv::Mat::zeros(threshRock.size(), CV_8UC1);
	cv::Mat scissorsDrawing = cv::Mat::zeros(threshScissors.size(), CV_8UC1);

	int largPaperCIndex = findLargestContour(paperContours);
	int largRockCIndex = findLargestContour(rockContours);
	int largScissorsCIndex = findLargestContour(scissorsCotours);

	cv::drawContours(paperDrawing, paperContours, largPaperCIndex, cv::Scalar(255, 0, 0), 8, cv::LINE_8);
	cv::drawContours(rockDrawing, rockContours, largRockCIndex, cv::Scalar(255, 0, 0), 8, cv::LINE_8);
	cv::drawContours(scissorsDrawing, scissorsCotours, largScissorsCIndex, cv::Scalar(255, 0, 0), 8, cv::LINE_8);

	cv::imshow("paperDrawing", paperDrawing);
	cv::imshow("rockDrawing", rockDrawing);
	cv::imshow("scissorsDrawing", scissorsDrawing);

	cv::waitKey(0);



	//double match1 = cv::matchShapes(paperDrawing, rockDrawing, cv::CONTOURS_MATCH_I2, 0);
	//std::cout << "Match (0-1): " << match1 << std::endl;

	//std::string result1 = match1 < .15 ? "It's a match!" : "They aren't the same!";
	//std::cout << result1 << std::endl;

	//std::string result2 = match2 < .15 ? "It's a match!" : "They aren't the same!";
	//std::cout << result2 << std::endl;

	cv::waitKey(0);

	std::cout << "Bye OpenCV" << std::endl;

	return 0;
}