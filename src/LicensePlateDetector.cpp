#include "LicensePlateDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

LicensePlateDetector::LicensePlateDetector(const std::string& tessdataPath) {
    if (tess.Init(tessdataPath.c_str(), "eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Could not initialize tesseract.\n";
        exit(1);
    }
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
}

std::string LicensePlateDetector::detectPlate(const cv::Mat& frame) {
    cv::Mat gray, blurred, edged;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::bilateralFilter(gray, blurred, 11, 17, 17);
    cv::Canny(blurred, edged, 30, 200);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::string bestText;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        float aspectRatio = (float)rect.width / rect.height;

        if (aspectRatio > 2 && aspectRatio < 6 && rect.area() > 2000) {
            cv::Mat plateROI = gray(rect);
            cv::threshold(plateROI, plateROI, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

            tess.SetImage((uchar*)plateROI.data, plateROI.cols, plateROI.rows, 1, plateROI.step);
            char* text = tess.GetUTF8Text();
            if (text) {
                std::string candidate(text);
                if (candidate.length() > bestText.length()) {
                    bestText = candidate;
                }
                delete[] text;
            }
        }
    }

    return bestText.empty() ? "No Plate Detected" : bestText;
}
