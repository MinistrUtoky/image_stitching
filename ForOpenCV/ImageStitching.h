#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#include <map>
#include <filesystem>

static std::mt19937 rng;
static std::uniform_real_distribution<float> udist(0.f, 1.f);
static std::normal_distribution<float> ndist(0.f, 5.f);

static void init_random()
{
    rng.seed(static_cast<long unsigned int>(time(0)));
}

static float random_number()
{
    return udist(rng);
}

static float random_noise()
{
    return ndist(rng);
}


class ImageStitching
{
public:
    static cv::Mat calculateNormalizationMatrix(const std::vector<cv::Point2d>& points);
    static std::vector<cv::Point2d> normalizePoints(const std::vector<cv::Point2d>& points, const cv::Mat& T);
    static cv::Mat calculateHomography(std::vector<cv::Point2d> points1, std::vector<cv::Point2d> points2);
    static void readFeatures(const std::string& fileName,
        std::vector<cv::Point2d>& points1,
        std::vector<cv::Point2d>& points2);
    static std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> detectAndMatchFeatures(std::string imageAddress1,
        std::string imageAddress2, int numberOfFeatures=0);
    static cv::Mat RANSAC_filter(std::vector<cv::Point2d>& image1Features, std::vector<cv::Point2d>& image2Features, cv::Mat H, double threshold = 1e7) {
        double d;
        std::vector<cv::Point2d> points1, points2;
        for (int i = 0; i < image1Features.size(); i++) {
            cv::Mat point1_homogeneous = (cv::Mat_<double>(3, 1) << image1Features[i].x, image1Features[i].y, 1.0);
            cv::Mat point2_homogeneous = (cv::Mat_<double>(3, 1) << image2Features[i].x, image2Features[i].y, 1.0);
            cv::Mat point1_proj_mat = H * point1_homogeneous;
            cv::Mat point2_proj_mat = H.inv() * point2_homogeneous;
            cv::Point2d point1_proj = cv::Point2d(point1_proj_mat.at<double>(0, 0),
                point2_proj_mat.at<double>(1, 0));
            cv::Point2d point2_proj = cv::Point2d(point2_proj_mat.at<double>(0, 0),
                point2_proj_mat.at<double>(1, 0));
            cv::Point2d delta1 = point1_proj - image2Features[i];
            cv::Point2d delta2 = point2_proj - image1Features[i];
            d = delta2.x * delta2.x + delta2.y * delta2.y + delta1.x * delta1.x + delta1.y * delta1.y;
            if (d <= threshold) {
                points1.push_back(image1Features[i]);
                points2.push_back(image2Features[i]);
                //std::cout << d << std::endl;
            }
        }
        if (points1.size() > 3) {
            image1Features = points1;
            image2Features = points2;
            H = ImageStitching::calculateHomography(points1, points2);
        }
        return H;
    }


    static cv::Mat misalignedFilter(int imageWidth, std::vector<cv::Point2d>& image1Features, std::vector<cv::Point2d>& image2Features, cv::Mat H, double maxMisalignment) {
        for (int i = 0; i < image2Features.size(); i++) {
            image2Features[i].x += imageWidth;
        }
        std::vector<int> bestParallels; 
        for (int i = 0; i < image1Features.size(); i++) {
            float a1 = image1Features[i].y - image2Features[i].y,
                  b1 = image2Features[i].x - image1Features[i].x, 
                  c1 = image1Features[i].x * image2Features[i].y - image2Features[i].x * image1Features[i].y;
            std::vector<int> parallels;
            parallels.push_back(i);
            for (int j = 0; j < image2Features.size(); j++) {
                if (i != j) {
                    float a2 = image1Features[j].y - image2Features[j].y,
                        b2 = image2Features[j].x - image1Features[j].x,
                        c2 = image1Features[j].x * image2Features[j].y - image2Features[j].x * image1Features[j].y;
                    std::cout << abs(a1 * b2 - b1 * a2) << std::endl;
                    if (abs(a1 * b2 - b1 * a2) < maxMisalignment) {
                        parallels.push_back(j);
                    }
                }
            }
            //std::cout << parallels.size() << std::endl;
            if (bestParallels.size() < parallels.size()) {
                bestParallels = parallels;
            }
            //std::cout << image1Features[i] << " " << image2Features[i] << std::endl;
        }
        if (bestParallels.size() > 3) {
            std::vector<cv::Point2d> points1, points2;
            for (int i = 0; i < bestParallels.size(); i++) {                
                points1.push_back(image1Features.at(bestParallels.at(i)));
                points2.push_back(image2Features.at(bestParallels.at(i)));
                points2[i].x -= imageWidth;
                //std::cout << image1Features.at(bestParallels.at(i)) << " " << image2Features.at(bestParallels.at(i)) << std::endl;
            }
            image1Features = points1;
            image2Features = points2;
            H = calculateHomography(points1, points2);
        }

        return H;
    }
};