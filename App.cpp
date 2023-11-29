// A projektem végül nem kõ- papír-olló játék, hanem a játék kézjeleit
// használja fel egy képfelismerõ program egyképfelismerõ funkció ellá-
// tásához. Ideális esetben felismeri a kõ papír, és olló jeleket,
// ezeken felül kiszûri az ide nem illõ elemeket. Észlelt hibák (false
// positives) esetén további próbálkozásokkal tanul azokból, a futása végén
// pedig kiértékeli az eredményeket.

// A program legjobban homogén háttérrel rendelkezõ képek esetén mûködik,
// ahol a kézjelek jól elkülönülnek a háttértõl, egyéb esetekben általában
// megbízhatatlan megoldásokat adhat.

// A fõ módszer a jelek meghatározására a kontúrok keresése, majd a leg-
// nagyobb kontúr megtalálása, és a kontúr alapján a kézjel meghatározása.
// Ez feltételezi, hogy a kézjel a legnagyobb kontúr lesz, ami nem mindig
// igaz, ezért is fontos, hogy a háttér homogén legyen.

// A program mûködését egyebek mellett nagyobb mennyiségû példaképek
// felhasználásával lehetne javítani, ami már a kezdeti eredményeket is
// jelentõsen javítaná.

#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// The thickness of the line for drawing contours
const int LINE_THICKNESS = 40;
// The similarity threshold for a match to be accepted as possibly valid
// This is adjusted when the program detects false positives
double ACCEPTANCE_THRESHOLD = 0.1;
// The maximum number of retries for re-checking false positives
int MAX_RETRIES = 5;
// Whether to print out intermediate results (like match values)
bool RUN_VERBOSE = true;

// The possible gestures
enum Gesture {
	NONE,
	PAPER,
	ROCK,
	SCISSORS
};

// Finds the largest contour in a vector of contours
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

// Returns the match value of two images
// Values range from 0 to 1, with 0 representing the closest resemblance, and 1 the opposite
double matchingOf(cv::Mat drawing1, cv::Mat drawing2) {
	// matchShapes() is independent of the scale, rotation, and starting point of the contour,
	// so we don't need to adjust the images for these factors
	float match = cv::matchShapes(drawing1, drawing2, cv::CONTOURS_MATCH_I3, 0);

	return match;
}

// Processes an image for contouring and returns the contour drawing
cv::Mat makeDrawing(cv::Mat image, std::string drawingName, bool doResize, cv::Scalar color) {
	// Resize large image to fit screen
	cv::resize(image, image, cv::Size(), 0.15, 0.15);

	// Converting to grayscale
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	// Thresholding image
	cv::Mat threshImage;
	cv::adaptiveThreshold(grayImage, threshImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 11);

	// Final touches setting up image for contouring
	cv::bitwise_not(threshImage, threshImage);
	cv::stackBlur(threshImage, threshImage, cv::Size(3, 3));

	// Creating contours and finding the largest
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(threshImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	int largestContourIdx = findLargestContour(contours);

	cv::Mat drawing = cv::Mat::zeros(threshImage.size(), CV_8UC1);

	cv::drawContours(drawing, contours, largestContourIdx, color, LINE_THICKNESS, cv::LINE_8);

	if (drawingName != "") {
		cv::imshow(drawingName, drawing);
	}

	return drawing;
}

int main(int argc, char* argv[]) {
	//RUN_VERBOSE = argc > 1 && argv[1] == "-v";

	std::string basePath = "images/base/";
	std::string gesturesPath = "images/gestures/";

	// Base images for later comparison
	cv::Mat rock, paper, scissors;

	cv::Mat basePaper = cv::imread(basePath + "paper.jpg", cv::IMREAD_COLOR);
	cv::Mat baseRock = cv::imread(basePath + "rock.jpg", cv::IMREAD_COLOR);
	cv::Mat baseScissors = cv::imread(basePath + "scissors.jpg", cv::IMREAD_COLOR);


	if (RUN_VERBOSE) {
		// Checking for reading errors
		if (basePaper.empty()) {
			std::cout << "Could not open or find the image paper" << std::endl;
		}
		if (baseRock.empty()) {
			std::cout << "Could not open or find the image rock" << std::endl;
		}
		if (baseScissors.empty()) {
			std::cout << "Could not open or find the image scissors" << std::endl;
		}
	}

	// Creating contour drawings of base images (initial references)
	cv::Mat paperDrawing = makeDrawing(basePaper, "", false, cv::Scalar(255, 0, 0));
	cv::Mat rockDrawing = makeDrawing(baseRock, "", false, cv::Scalar(255, 0, 0));
	cv::Mat scissorsDrawing = makeDrawing(baseScissors, "", false, cv::Scalar(255, 0, 0));

	if (RUN_VERBOSE) {
		cv::imshow("paperDrawing", paperDrawing);
		cv::imshow("rockDrawing", rockDrawing);
		cv::imshow("scissorsDrawing", scissorsDrawing);

		cv::waitKey(0);
	}

	// Trackers for the number of images for each gesture
	int maxNones = 0;
	int maxPapers = 0;
	int maxRocks = 0;
	int maxScissors = 0;

	// Trackers for success rates
	int initialNoneSuccessRate = 0;
	int initialPaperSuccessRate = 0;
	int initialRockSuccessRate = 0;
	int initialScissorsSuccessRate = 0;

	// Vectors for storing real matches to avoid reading them again from disk
	std::vector<cv::Mat> realNonesVector;
	std::vector<cv::Mat> realPapersVector;
	std::vector<cv::Mat> realRocksVector;
	std::vector<cv::Mat> realScissorsVector;

	std::vector<std::string> falseIds;

	// Trackers for false positives
	int falseNone = 0;
	int falsePaper = 0;
	int falseRock = 0;
	int falseScissors = 0;

	int readingErrors = 0;

	// Checking for path error
	if (!fs::exists(gesturesPath) && fs::is_directory(gesturesPath)) {
		std::cout << "Could not open or find the directory gestures" << std::endl;

		return -1;
	}

	// Looping through all images in the gestures directory
	for (const auto& entry : fs::directory_iterator(gesturesPath)) {
		std::string fileName = entry.path().filename().string();
		std::string directory = entry.path().parent_path().filename().string();

		Gesture properMatch = fileName.substr(0, fileName.find("_")) == "paper"
			? Gesture::PAPER
			: fileName.substr(0, fileName.find("_")) == "rock"
			? Gesture::ROCK
			: fileName.substr(0, fileName.find("_")) == "scissors"
			? Gesture::SCISSORS
			: Gesture::NONE;


		if (properMatch == Gesture::PAPER) {
			maxPapers++;
		}
		else if (properMatch == Gesture::ROCK) {
			maxRocks++;
		}
		else if (properMatch == Gesture::SCISSORS) {
			maxScissors++;
		}
		else {
			maxNones++;
		}

		cv::Mat gesture = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

		// Checking for reading errors
		if (gesture.empty()) {
			std::cout << "Could not open or find the image (Reading)" << std::endl;
			readingErrors++;

			continue;
		}

		cv::Mat gestureDrawing = makeDrawing(gesture, "", true, cv::Scalar(255, 0, 0));

		// Determining matches to base images and the closest match among them
		double matchPaper = matchingOf(gestureDrawing, paperDrawing);
		double matchRock = matchingOf(gestureDrawing, rockDrawing);
		double matchScissors = matchingOf(gestureDrawing, scissorsDrawing);
		double smallestMatch = std::min(matchPaper, std::min(matchRock, matchScissors));

		// Checking for matches
		if (smallestMatch > ACCEPTANCE_THRESHOLD) {
			if (RUN_VERBOSE) {
				std::cout << "Match! " << entry.path().filename() << " -> " << "none (" << smallestMatch << ")" << std::endl;
			}

			if (properMatch == Gesture::NONE) {
				initialNoneSuccessRate++;
				realNonesVector.push_back(gestureDrawing);
			}
			else {
				falseNone++;
				falseIds.push_back(fileName);
			}
		}
		else if (smallestMatch == matchPaper) {
			if (RUN_VERBOSE) {
				std::cout << "Match! " << entry.path().filename() << " -> " << "paper (" << smallestMatch << ")" << std::endl;
			}

			if (properMatch == Gesture::PAPER) {
				initialPaperSuccessRate++;
				realPapersVector.push_back(gestureDrawing);
			}
			else {
				falsePaper++;
				falseIds.push_back(fileName);
			}
		}
		else if (smallestMatch == matchRock) {
			if (RUN_VERBOSE) {
				std::cout << "Match! " << entry.path().filename() << " -> " << "rock (" << smallestMatch << ")" << std::endl;
			}

			if (properMatch == Gesture::ROCK) {
				initialRockSuccessRate++;
				realRocksVector.push_back(gestureDrawing);
			}
			else {
				falseRock++;
				falseIds.push_back(fileName);
			}
		}
		else if (smallestMatch == matchScissors) {
			if (RUN_VERBOSE) {
				std::cout << "Match! " << entry.path().filename() << " -> " << "scissors (" << smallestMatch << ")" << std::endl;
			}

			if (properMatch == Gesture::SCISSORS) {
				initialScissorsSuccessRate++;
				realScissorsVector.push_back(gestureDrawing);
			}
			else {
				falseScissors++;
				falseIds.push_back(fileName);
			}
		}

		std::cout << std::endl;
	}

	std::cout << "--------------------------------------------------" << std::endl;

	// Trackers for adjusted success rates
	int endNoneSuccessRate = initialNoneSuccessRate;
	int endPaperSuccessRate = initialPaperSuccessRate;
	int endRockSuccessRate = initialRockSuccessRate;
	int endScissorsSuccessRate = initialScissorsSuccessRate;

	// If there are false positives, check them again until none remains or max retries are reached
	// Compare them to proven real matches
	while (!falseIds.empty()) {
		if (MAX_RETRIES == 0) {
			std::cout << std::endl << "Max retries reached, exiting!" << std::endl;

			break;
		}

		// Looping through all false positives
		for (int i = 0; i < falseIds.size(); i++) {
			std::string falseId = falseIds[i];

			cv::Mat falseGesture = cv::imread(gesturesPath + falseId, cv::IMREAD_COLOR);

			Gesture properMatch = falseId.substr(0, falseId.find("_")) == "paper"
				? Gesture::PAPER
				: falseId.substr(0, falseId.find("_")) == "rock"
				? Gesture::ROCK
				: falseId.substr(0, falseId.find("_")) == "scissors"
				? Gesture::SCISSORS
				: Gesture::NONE;


			if (RUN_VERBOSE) {
				// Checking for reading error
				if (falseGesture.empty()) {
					std::cout << "Could not open or find the image for reading" << std::endl;
					readingErrors++;

					continue;
				}
			}

			cv::Mat falseGestureDrawing = makeDrawing(falseGesture, "", true, cv::Scalar(255, 0, 0));

			// Determining matches to base images and the closest match among them
			if (properMatch == Gesture::PAPER) {
				double papersMatch = 1;

				for (int i = 0; i < realPapersVector.size(); i++) {
					double match = matchingOf(falseGestureDrawing, realPapersVector[i]);

					papersMatch = match < papersMatch ? match : papersMatch;
				}

				if (papersMatch <= ACCEPTANCE_THRESHOLD) {
					std::cout << "Match! " << falseId << " -> " << "paper" << std::endl;

					endPaperSuccessRate++;
					realPapersVector.push_back(falseGestureDrawing);
					falseIds.erase(falseIds.begin() + i);
				}
			}

			if (properMatch == Gesture::ROCK) {
				double rocksMatch = 1;

				for (int i = 0; i < realRocksVector.size(); i++) {
					double match = matchingOf(falseGestureDrawing, realRocksVector[i]);

					rocksMatch = match < rocksMatch ? match : rocksMatch;
				}

				if (rocksMatch <= ACCEPTANCE_THRESHOLD) {
					std::cout << "Match! " << falseId << " -> " << "rock" << std::endl;

					endRockSuccessRate++;
					realRocksVector.push_back(falseGestureDrawing);
					falseIds.erase(falseIds.begin() + i);
				}
			}

			if (properMatch == Gesture::SCISSORS) {
				double scissorsMatch = 1;

				for (int i = 0; i < realScissorsVector.size(); i++) {
					double match = matchingOf(falseGestureDrawing, realScissorsVector[i]);

					scissorsMatch = match < scissorsMatch ? match : scissorsMatch;
				}

				if (scissorsMatch <= ACCEPTANCE_THRESHOLD) {
					std::cout << "Match! " << falseId << " -> " << "scissors" << std::endl;

					endScissorsSuccessRate++;
					realScissorsVector.push_back(falseGestureDrawing);
					falseIds.erase(falseIds.begin() + i);
				}
			}

			if (properMatch == Gesture::NONE) {
				double nonesMatch = 1;

				for (int i = 0; i < realNonesVector.size(); i++) {
					double match = matchingOf(falseGestureDrawing, realNonesVector[i]);

					nonesMatch = match < nonesMatch ? match : nonesMatch;
				}

				if (nonesMatch <= ACCEPTANCE_THRESHOLD) {
					std::cout << "Match! " << falseId << " -> " << "none" << std::endl;

					endNoneSuccessRate++;
					realNonesVector.push_back(falseGestureDrawing);
					falseIds.erase(falseIds.begin() + i);
				}
			}
		}

		std::cout << "Remaining false positives: " << falseIds.size() << std::endl;

		MAX_RETRIES--;
	}

	// Summary of results
	std::cout << std::endl << "--------------------------------------------------" << std::endl;
	std::cout << "Initial none success rate: " << initialNoneSuccessRate << "/" << maxNones << std::endl;
	std::cout << "Initial paper success rate: " << initialPaperSuccessRate << "/" << maxPapers << std::endl;
	std::cout << "Initial rock success rate: " << initialRockSuccessRate << "/" << maxRocks << std::endl;
	std::cout << "Initial scissors success rate: " << initialScissorsSuccessRate << "/" << maxScissors << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "End none success rate: " << initialNoneSuccessRate << "/" << maxNones << std::endl;
	std::cout << "End paper success rate: " << initialPaperSuccessRate << "/" << maxPapers << std::endl;
	std::cout << "End rock success rate: " << initialRockSuccessRate << "/" << maxRocks << std::endl;
	std::cout << "End scissors success rate: " << initialScissorsSuccessRate << "/" << maxScissors << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "False positives for none: " << falseNone << std::endl;
	std::cout << "False positives for paper: " << falsePaper << std::endl;
	std::cout << "False positives for rock: " << falseRock << std::endl;
	std::cout << "False positives for scissors: " << falseScissors << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	// Initial success rates
	std::cout << std::endl << "--------------------------------------------------" << std::endl;
	std::cout << "Initial success rate: "
		<< (initialNoneSuccessRate + initialPaperSuccessRate + initialRockSuccessRate + initialScissorsSuccessRate)
		<< "/"
		<< (maxNones + maxPapers + maxRocks + maxScissors) << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	// Adjusted success rates
	std::cout << std::endl << "--------------------------------------------------" << std::endl;
	std::cout << "End success rate: "
		<< (endNoneSuccessRate + endPaperSuccessRate + endRockSuccessRate + endScissorsSuccessRate)
		<< "/"
		<< (maxNones + maxPapers + maxRocks + maxScissors) << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	cv::waitKey(0);

	return 0;
}
