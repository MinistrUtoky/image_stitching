#include "ImageStitching.h"


cv::Mat ImageStitching::calculateNormalizationMatrix(const std::vector<cv::Point2d>& points) {
    cv::Point2d mean(0, 0); //mean
    cv::Point2d st_dev(0, 0); //standard deviation
    int n = points.size();

    for (cv::Point2d p : points) { mean += p; }
    mean *= 1.0 / n;

    for (cv::Point2d point : points) {
        st_dev.x += pow(point.x - mean.x, 2);
        st_dev.y += pow(point.y - mean.y, 2);
    }
    st_dev.x = sqrt(st_dev.x / n); st_dev.y = sqrt(st_dev.y / n);
    double std_avg = (st_dev.x + st_dev.y) / 2.0f;
    double scale = sqrt(2) / std_avg;
    cv::Point2d offset = -scale * mean;

    cv::Mat T = cv::Mat::eye(3, 3, CV_64F); // normalization matrix
    T.at<double>(0, 0) = scale;
    T.at<double>(1, 1) = scale;
    T.at<double>(0, 2) = offset.x;
    T.at<double>(1, 2) = offset.y;
    return T;
}

std::vector<cv::Point2d> ImageStitching::normalizePoints(const std::vector<cv::Point2d>& points, const cv::Mat& T) {
    std::vector<cv::Point2d> normalized_points;
    for (cv::Point2d point : points) {
        cv::Mat point_homogeneous = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
        cv::Mat normalized_point = T * point_homogeneous;
        normalized_points.push_back(cv::Point2d(normalized_point.at<double>(0, 0) / normalized_point.at<double>(2, 0),
            normalized_point.at<double>(1, 0) / normalized_point.at<double>(2, 0)));
    }
    return normalized_points;
}

cv::Mat ImageStitching::calculateHomography(std::vector<cv::Point2d> points1, std::vector<cv::Point2d> points2) {
    double* aData = new double[(points1.size() - 1) * 18 + 17];

    for (int i = 0; i < points1.size(); i++) {
        cv::Point p1 = points1.at(i), p2 = points2.at(i);
        //A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        aData[18 * i] = p1.x; aData[18 * i + 1] = p1.y; aData[18 * i + 2] = 1;
        aData[18 * i + 3] = 0; aData[18 * i + 4] = 0; aData[18 * i + 5] = 0;
        aData[18 * i + 6] = -p1.x * p2.x; aData[18 * i + 7] = -p1.y * p2.x; aData[18 * i + 8] = -p2.x;
        //A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        aData[(18 * i + 9)] = 0; aData[(18 * i + 9) + 1] = 0; aData[(18 * i + 9) + 2] = 0;
        aData[(18 * i + 9) + 3] = p1.x; aData[(18 * i + 9) + 4] = p1.y; aData[(18 * i + 9) + 5] = 1;
        aData[(18 * i + 9) + 6] = -p1.x * p2.y; aData[(18 * i + 9) + 7] = -p1.y * p2.y; aData[(18 * i + 9) + 8] = -p2.y;
    }

    cv::Mat A(points1.size() * 2, 9, CV_64F, aData, cv::Mat::AUTO_STEP);

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt); //params explained here (singlar = eigen)

    double minEigVal = w.at<double>(0, 0); int minEigIdx = 0;
    for (int i = 0; i < w.rows; i++) {
        if (w.at<double>(i, 0) < minEigVal)
        {
            minEigVal = w.at<double>(i, 0);
            minEigIdx = i;
        }
    }

    cv::Mat vtMinRow = vt.row(minEigIdx);
    cv::Mat H(3, 3, CV_64F);
    H.at<double>(0, 0) = vtMinRow.at<double>(0) / vtMinRow.at<double>(8);
    H.at<double>(0, 1) = vtMinRow.at<double>(1) / vtMinRow.at<double>(8);
    H.at<double>(0, 2) = vtMinRow.at<double>(2) / vtMinRow.at<double>(8);
    H.at<double>(1, 0) = vtMinRow.at<double>(3) / vtMinRow.at<double>(8);
    H.at<double>(1, 1) = vtMinRow.at<double>(4) / vtMinRow.at<double>(8);
    H.at<double>(1, 2) = vtMinRow.at<double>(5) / vtMinRow.at<double>(8);
    H.at<double>(2, 0) = vtMinRow.at<double>(6) / vtMinRow.at<double>(8);
    H.at<double>(2, 1) = vtMinRow.at<double>(7) / vtMinRow.at<double>(8);
    H.at<double>(2, 2) = 1; 

    return H;
}

void ImageStitching::readFeatures(const std::string& fileName,
    std::vector<cv::Point2d>& points1,
    std::vector<cv::Point2d>& points2) {
    std::ifstream file(fileName);
    std::string line;

    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << fileName << std::endl;
        return;
    }

    std::getline(file, line);
    std::stringstream x1s(line);

    std::getline(file, line);
    std::stringstream y1s(line);

    std::getline(file, line);
    std::stringstream x2s(line);

    std::getline(file, line);
    std::stringstream y2s(line);

    double x1, y1, x2, y2;
    for (int i = 0; i < 8; i++) {
        x1s >> x1;
        y1s >> y1;
        x2s >> x2;
        y2s >> y2;
        points1.push_back(cv::Point2d(x1, y1));
        points2.push_back(cv::Point2d(x2, y2));
    }

    file.close();
}

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ImageStitching::detectAndMatchFeatures(std::string imageAddress1,
    std::string imageAddress2, int numberOfFeatures) {
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ps;

    cv::Mat img1 = cv::imread(imageAddress1, 0),
        img2 = cv::imread(imageAddress2, 0);

    if (img1.empty() || img2.empty()) {
        std::cout << "Error loading images!" << std::endl;
        return ps;
    }

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(numberOfFeatures);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat dsc1, dsc2;

    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;

    sift->detectAndCompute(img1, cv::noArray(), keypoints1, dsc1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, dsc2);

    matcher.match(dsc1, dsc2, matches);

    std::vector<cv::DMatch> matches2;
    // distance between match points
    float d = 0, min_d = 1e10, max_d = 0;
    for (auto match : matches) {
        d = match.distance;
        if (d < min_d) min_d = d;
        if (d > max_d) max_d = d;
    }
    // simple filtering
    for (auto match : matches) {
        if (match.distance <= 2*min_d) {
           matches2.push_back(match);
        }
    }

    cv::Mat display;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches2, display);
    cv::imwrite("matches.png", display);

    std::vector<cv::Point2d> ps1; std::vector<cv::Point2d> ps2;
    for (auto match : matches2) {
        cv::Point2d p1 = cv::Point2d((int)keypoints1[match.queryIdx].pt.x, (int)keypoints1[match.queryIdx].pt.y);
        cv::Point2d p2 = cv::Point2d((int)keypoints2[match.trainIdx].pt.x, (int)keypoints2[match.trainIdx].pt.y);
        ps1.push_back(p1);
        ps2.push_back(p2);
    }
    ps.first = ps1; ps.second = ps2;

    return ps;
}
