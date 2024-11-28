#include "ImageStitching.h"

cv::Mat getR(cv::Mat H, std::vector<cv::Point2d> image1Features,
                 std::vector<cv::Point2d> image2Features) {
    int N = image1Features.size();
    cv::Mat r = cv::Mat(2 * N, 1, CV_64F);
    double X, Y, w, x_proj, y_proj;
    for (int i = 0; i < N; ++i) {
        X = image1Features[i].x;
        Y = image1Features[i].y;
        w = H.at<double>(2, 0) * X + H.at<double>(2, 1) * Y + H.at<double>(2, 2);
        x_proj = (H.at<double>(0, 0) * X + H.at<double>(0, 1) * Y + H.at<double>(0, 2)) / w;
        y_proj = (H.at<double>(1, 0) * X + H.at<double>(1, 1) * Y + H.at<double>(1, 2)) / w;
        r.at<double>(i) = x_proj - image2Features[i].x; 
        r.at<double>(N + i) = y_proj - image2Features[i].y; 
    }
    return r;
}

cv::Mat getJ(cv::Mat xs, const double* h) {
    int N = xs.rows / 2;
    cv::Mat X = xs.rowRange(0, N);
    cv::Mat Y = xs.rowRange(N, xs.rows);

    cv::Mat s_x = h[0] * X + h[1] * Y + h[2];
    cv::Mat s_y = h[3] * X + h[4] * Y + h[5];
    cv::Mat w = h[6] * X + h[7] * Y + h[8];
    cv::Mat w_sq = w.mul(w);

    cv::Mat J = cv::Mat::zeros(2 * N, 9, CV_64F);
    J.rowRange(0, N).col(0) = X / w;
    J.rowRange(0, N).col(1) = Y / w;
    J.rowRange(0, N).col(2) = 1.0 / w;
    J.rowRange(0, N).col(6) = (-s_x.mul(X)) / w_sq;
    J.rowRange(0, N).col(7) = (-s_x.mul(Y)) / w_sq;
    J.rowRange(0, N).col(8) = -s_x / w_sq;
    J.rowRange(N, 2 * N).col(3) = X / w;
    J.rowRange(N, 2 * N).col(4) = Y / w;
    J.rowRange(N, 2 * N).col(5) = 1.0 / w;
    J.rowRange(N, 2 * N).col(6) = (-s_y.mul(X)) / w_sq;
    J.rowRange(N, 2 * N).col(7) = (-s_y.mul(Y)) / w_sq;
    J.rowRange(N, 2 * N).col(8) = -s_y / w_sq;
    return J;
}

cv::Mat refineHomography(cv::Mat H, std::vector<cv::Point2d> image1Features, 
                         std::vector<cv::Point2d> image2Features, int maxIters, float goodEnoughDeltaNorm=1e-6) {
    int N = image1Features.size();
    cv::Mat X(N, 1, CV_64F), Y(N, 1, CV_64F), 
            x(N, 1, CV_64F), y(N, 1, CV_64F);
    cv::Mat xs = cv::Mat::zeros(N * 2, 1, CV_64F); 
    for (int i = 0; i < N; i++) {
        X.at<double>(i, 0) = image1Features[i].x;
        Y.at<double>(i, 0) = image1Features[i].y;
        x.at<double>(i, 0) = image2Features[i].x;
        y.at<double>(i, 0) = image2Features[i].y;
    }
    X.copyTo(xs.rowRange(0, N));
    Y.copyTo(xs.rowRange(N, 2 * N));
    cv::Mat h(1, 9, CV_64F); h.data = H.data;
    cv::Mat H_refined(3, 3, CV_64F); H.copyTo(H_refined);
    double step = 0.001;
    cv::Mat E = cv::Mat::eye(9, 9, CV_64F);
    cv::Mat J(2 * N, 9, CV_64F);
    for (int i = 0; i < maxIters; ++i) {
        cv::Mat r = getR(H_refined, image1Features, image2Features);
        double hh[9]{ h.at<double>(0), h.at<double>(1), h.at<double>(2),
                      h.at<double>(3), h.at<double>(4), h.at<double>(5),
                      h.at<double>(6), h.at<double>(7), h.at<double>(8) };
        cv::Mat J = getJ(xs, hh);
        cv::Mat delta = ((J.t() * J + step * E).inv() * J.t() * r).t(); 

        h -= delta;

        H_refined.data = h.data;
        H_refined /= H_refined.at<double>(8);
        if (cv::norm(delta) < goodEnoughDeltaNorm) {
            break;
        }
    }

    return H_refined;
}

cv::Mat stitch(cv::Mat img1, cv::Mat img2, cv::Mat H_normalized, bool isLeft) {
    cv::Mat stitched_img;
    if (isLeft) {
        cv::Mat img1_transformed;
        cv::warpPerspective(img1, img1_transformed, H_normalized, cv::Size(img2.cols + img1.cols, img2.rows));
        stitched_img = img1_transformed.clone();
        img2.copyTo(stitched_img(cv::Rect(0, 0, img2.cols, img2.rows)));
    }
    else {
        cv::Mat img2_transformed;
        cv::warpPerspective(img2, img2_transformed, H_normalized.inv(), cv::Size(img1.cols + img2.cols, img1.rows));
        stitched_img = img2_transformed.clone();
        img1.copyTo(stitched_img(cv::Rect(0, 0, img1.cols, img1.rows)));
    }
    return stitched_img;
}


cv::Mat stitchByMatches(std::string image1Name, std::string image2Name,
    std::vector<cv::Point2d> image1Features, std::vector<cv::Point2d> image2Features,
    cv::Mat& H_collected, bool& isLeft) {
    cv::Mat stitched_img;
    cv::Mat img1 = cv::imread(image1Name);
    cv::Mat img2 = cv::imread(image2Name);
    
    std::cout << cv::findHomography(image1Features, image2Features) << std::endl; 

    ImageStitching is = ImageStitching();
    cv::Mat T1 = is.calculateNormalizationMatrix(image1Features);
    cv::Mat T2 = is.calculateNormalizationMatrix(image2Features);
    std::vector<cv::Point2d> normalized_points1 = is.normalizePoints(image1Features, T1);
    std::vector<cv::Point2d> normalized_points2 = is.normalizePoints(image2Features, T2);

    cv::Mat H_normalized = is.calculateHomography(image1Features, image2Features);
    
    std::cout << H_normalized << std::endl;
    // refinement
    H_normalized = ImageStitching::RANSAC_filter(image1Features, image2Features, H_normalized, 1e7); // for 1 better 2*1e6, for 3 better 1e7

    std::cout << H_normalized << std::endl;

    H_normalized = ImageStitching::misalignedFilter(img1.cols, image1Features, image2Features, H_normalized, 8*1e3);

    std::cout << H_normalized << std::endl;
    
    refineHomography(H_normalized, image1Features, image2Features, 50, 1e-6).copyTo(H_normalized);
    //H_normalized = cv::findHomography(image1Features, image2Features); for comparison
    std::cout << H_normalized << std::endl; 

    /*H_normalized.convertTo(H_normalized, CV_64FC1);
    T2.convertTo(H_normalized, CV_64FC1);
    T1.convertTo(H_normalized, CV_64FC1);
    cv::Mat H = T2.inv() * H_normalized * T1;
    std::cout << H << std::endl;*/

    double minStitchDistance = img2.cols + img1.cols;
    double averageX = 0;
    for (int i = 0; i < image1Features.size(); i++) {
        cv::Point2d p1 = image1Features.at(i);
        averageX += p1.x;
    }
    averageX /= image1Features.size();
    isLeft = averageX < img1.cols / 2;
    stitched_img = stitch(img1, img2, H_normalized, isLeft);

    if (img1.empty() || img2.empty()) {
        std::cout << "Error loading images!" << std::endl;
        return stitched_img;
    }

    H_normalized.convertTo(H_normalized, CV_64F);

    if (!cv::imwrite("stitched_image.png", stitched_img)) {
        std::cout << "Failed to save stitched image." << std::endl;
    }
    H_collected = H_normalized;
    return stitched_img;
}


void main() {
    std::vector<cv::Point2d> points1; std::vector<cv::Point2d> points2;
    //ImageStitching::readFeatures("features.txt", points1, points2); useless by nowl

    std::string path = "T:\\images\\5\\*.jpeg";

    std::vector<std::string> imageNames;
    cv::glob(path, imageNames, false);

    std::vector<cv::Mat> Hs;
    bool isLeft;
    for (int i = 0; i < imageNames.size() - 1; i++) {
        std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> points = ImageStitching::detectAndMatchFeatures(imageNames.at(i), imageNames.at(i + 1));
        points1 = points.first;
        points2 = points.second;
        cv::Mat H;
        cv::Mat img = stitchByMatches(imageNames.at(i), imageNames.at(i + 1), points1, points2, H, isLeft);
        cv::Mat displayImage;
        cv::resize(img, displayImage, cv::Size((int)(img.cols / img.rows * 600), 600));
        cv::imshow("Stitched", displayImage);
        cv::imwrite("pair" + std::to_string(i) + ".png", img);
        cv::waitKey(0);
        Hs.push_back(H);
    }

    cv::Mat intermediateImage = cv::imread(imageNames.at(0)); 
    cv::Mat H_i = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat hostImage(cv::Size(intermediateImage.cols * imageNames.size(), intermediateImage.rows), intermediateImage.type());
    cv::Mat displayImage;

    for (int i = 0; i < Hs.size(); i++) {
        cv::Mat img = intermediateImage.clone();
        if (isLeft) {
            cv::Mat img1 = cv::imread(imageNames.at(i + 1));
            H_i = Hs.at(i);
            H_i = H_i / H_i.at<double>(8);
            std::cout << H_i << std::endl;
            cv::Mat img1_transformed;
            cv::warpPerspective(intermediateImage, img1_transformed, H_i, cv::Size(intermediateImage.cols + img1.cols, intermediateImage.rows));
            intermediateImage = img1_transformed.clone();
            cv::Mat mask;
            cv::inRange(img1, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), mask);
            img1.copyTo(intermediateImage(cv::Rect(0, 0, img1.cols, img1.rows)), 255 - mask);
        }
        else {
            cv::Mat img2 = cv::imread(imageNames.at(i + 1));
            H_i = H_i * Hs.at(i).inv();
            H_i = H_i / H_i.at<double>(8);
            std::cout << H_i << std::endl;
            cv::Mat img2_transformed;
            cv::warpPerspective(img2, img2_transformed, H_i, cv::Size(intermediateImage.cols + img2.cols, intermediateImage.rows));
            intermediateImage = img2_transformed.clone();
            cv::Mat mask;
            cv::inRange(img, cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), mask);
            img.copyTo(intermediateImage(cv::Rect(0, 0, img.cols, img.rows)), 255-mask);
        }
        intermediateImage.copyTo(hostImage(cv::Rect(0, 0, intermediateImage.cols, intermediateImage.rows)));
        cv::resize(hostImage, displayImage, cv::Size(1920, (double)(hostImage.rows) / hostImage.cols * 1920));
        cv::imshow("Stitched", displayImage);
        cv::waitKey(0);
    }
    cv::imwrite("the_stitched_image.png", hostImage);

    return;
}
