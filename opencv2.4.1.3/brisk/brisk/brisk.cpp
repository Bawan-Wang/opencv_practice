#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <Windows.h>

using namespace cv;
using namespace std;

void PrintMat(Mat A)
{
	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.cols; j++)
			cout << A.at<double>(i, j) << " ";
		cout << endl;
	}
	cout << endl;
}


int main()
{
	Mat book_color = imread("book.jpg");
	Mat books_color = imread("books.jpg");
	Mat book = imread("book.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat books = imread("books.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//detect kp
	BRISK brisk_detector;
	vector<KeyPoint> book_kp, books_kp;
	book_kp.reserve(5000);
	books_kp.reserve(5000);
	double start = GetTickCount();
	brisk_detector.detect(book, book_kp);
	brisk_detector.detect(books, books_kp);


	//extract BRISK descriptor
	Mat book_desc, books_desc;
	brisk_detector.compute(book, book_kp, book_desc);
	brisk_detector.compute(books, books_kp, books_desc);
	cout << "size of descriptor of book: " << book_kp.size() << endl;
	cout << "size of descriptor of books: " << books_kp.size() << endl;

	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matches.reserve(5000);
	matcher.match(book_desc, books_desc, matches);
	double end = GetTickCount();
	cout << "elapsed time：" << (end - start) << "ms" << endl;

	//find good matched points
	vector<DMatch> good_matches;
	double minDist = 1000, good_matches_th;
	for (int i = 0; i < book_desc.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
	}
	good_matches_th = max(2 * minDist, 0.02);
	for (int i = 0; i < book_desc.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < good_matches_th)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_match;
	drawMatches(book_color, book_kp, books_color, books_kp, good_matches, img_match);
	cout << "number of matched points: " << good_matches.size() << endl;

	vector<Point2f> book_match_kp;
	vector<Point2f> books_match_kp;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		book_match_kp.push_back(book_kp[good_matches[i].queryIdx].pt);
		books_match_kp.push_back(books_kp[good_matches[i].trainIdx].pt);

	}
	//Generate transformation matrix(Homography matrix)
	Mat H = findHomography(book_match_kp, books_match_kp, RANSAC);
	PrintMat(H);

	vector<Point2f>book_corner(4);
	vector<Point2f>book_corner_in_scene(4);
	book_corner[0] = Point(0, 0);
	book_corner[1] = Point(book.cols, 0);
	book_corner[2] = Point(book.cols, book.rows);
	book_corner[3] = Point(0, book.rows);

	//Corners transform
	perspectiveTransform(book_corner, book_corner_in_scene, H);
	Mat img_result = img_match.clone();
	line(img_result, book_corner_in_scene[0] + Point2f(book.cols, 0), book_corner_in_scene[1] + Point2f(book.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(img_result, book_corner_in_scene[1] + Point2f(book.cols, 0), book_corner_in_scene[2] + Point2f(book.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(img_result, book_corner_in_scene[2] + Point2f(book.cols, 0), book_corner_in_scene[3] + Point2f(book.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(img_result, book_corner_in_scene[3] + Point2f(book.cols, 0), book_corner_in_scene[0] + Point2f(book.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	namedWindow("match demo", 0);
	resizeWindow("match demo", img_result.cols / 4, img_result.rows / 4);
	imshow("match demo", img_result);

	waitKey(0);
	cvDestroyAllWindows();

	img_result.release();
	H.release();
	img_match.release();
	vector<DMatch>().swap(good_matches);
	vector<DMatch>().swap(matches);
	book_desc.release();
	books_desc.release();
	vector<KeyPoint>().swap(book_kp);
	vector<KeyPoint>().swap(books_kp);
	return 0;
}
