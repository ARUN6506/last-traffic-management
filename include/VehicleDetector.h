#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include<string>

struct Detection {
	cv::Rect bbox;
	int classId;
	float confidence;
};
class VehicleDetector
{
public:
	VehicleDetector(const std::string& modelPath, const std::string& classesPath, float confThreshold = 0.5,
		float nmsThreshold = 0.4);
	std::vector<Detection> detect(const cv::Mat& frame);
	int getClassId(const std::string& className) const;

private:
	cv::dnn::Net net_;
	std::vector<std::string> classNames_;
	float confThreshold_, nmsThreshold_;
};

