#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<vector>
#include<math.h>
#include<time.h>  
#include "imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

int getFrequencyDomain(Mat &img)
{
	// get optimal size for fft
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	// resize original image, with 0s in new space
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	// get two image to save real and imaginary parts
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;

	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);

	// get sum of square, do log and normalize
	magnitude(planes[0], planes[1], planes[0]);
	Mat result_img = planes[0];
	result_img += Scalar::all(1);
	log(result_img, result_img);
	result_img = result_img(Rect(0, 0, result_img.cols & -2, result_img.rows & -2)); // trim the spectrum
	Mat result_ori = result_img.clone();
	normalize(result_ori, result_ori, -1, 1, CV_MINMAX);
	imshow("BeforeRearrange ", result_ori);

	// rearrange original image
	int cx = result_img.cols / 2;
	int cy = result_img.rows / 2;
	Mat tmp;
	Mat q0(result_img, Rect(0, 0, cx, cy));
	Mat q1(result_img, Rect(cx, 0, cx, cy));
	Mat q2(result_img, Rect(0, cy, cx, cy));
	Mat q3(result_img, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(result_img, result_img, -1, 1, CV_MINMAX);
	imshow("FrequencyDomain", result_img);
	waitKey();

	return 0;
}

int frequencyFilter(Mat &img)
{
	// get frequency image by dft
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	::copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat FrequencyImg;
	::merge(planes, 2, FrequencyImg);
	::dft(FrequencyImg, FrequencyImg); 

	Mat iDFT[] = { Mat::zeros(padded.size(),CV_32F),Mat::zeros(padded.size(),CV_32F) };
	::split(FrequencyImg, iDFT);
	Mat iDFT_cos;
	::magnitude(iDFT[0], iDFT[1], iDFT_cos);

	// show frequency graph, move to center
	int cx = iDFT_cos.cols / 2;
	int cy = iDFT_cos.rows / 2;
	Mat tmp;
	Mat q0(iDFT_cos, Rect(0, 0, cx, cy));
	Mat q1(iDFT_cos, Rect(cx, 0, cx, cy));
	Mat q2(iDFT_cos, Rect(0, cy, cx, cy));
	Mat q3(iDFT_cos, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	log(iDFT_cos, iDFT_cos);
	::normalize(iDFT_cos, iDFT_cos, -1, 1, CV_MINMAX);
	::imshow("iDFT_before", iDFT_cos);

	// remove unexpected points
	iDFT[0].at<float>(0, 9) = iDFT[0].at<float>(cx, cy);
	iDFT[0].at<float>(0, 10) = iDFT[0].at<float>(cx, cy);
	iDFT[0].at<float>(0, 170) = iDFT[0].at<float>(cx, cy);
	iDFT[0].at<float>(0, 171) = iDFT[0].at<float>(cx, cy);
	iDFT[1].at<float>(0, 9) = iDFT[0].at<float>(cx, cy);
	iDFT[1].at<float>(0, 10) = iDFT[0].at<float>(cx, cy);
	iDFT[1].at<float>(0, 170) = iDFT[0].at<float>(cx, cy);
	iDFT[1].at<float>(0, 171) = iDFT[0].at<float>(cx, cy);

	// show frequency graph after remove unexpected points
	::magnitude(iDFT[0], iDFT[1], iDFT_cos);
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	log(iDFT_cos, iDFT_cos);
	::normalize(iDFT_cos, iDFT_cos, -1, 1, CV_MINMAX);
	::imshow("iDFT_after", iDFT_cos);

	// inverse transform from frequency to space
	::merge(iDFT, 2, FrequencyImg);
	Mat iPartDft[] = { Mat::zeros(padded.size(),CV_32F),Mat::zeros(padded.size(),CV_32F) };
	Mat outputImg;
	::idft(FrequencyImg, outputImg);
	::split(outputImg, iPartDft);
	::magnitude(iPartDft[0], iPartDft[1], iPartDft[0]);
	::normalize(iPartDft[0], iPartDft[0], 0, 1, CV_MINMAX);
	::imshow("output", iPartDft[0]);
	::imshow("img", img);

	::waitKey();
	return 0;
}

int simpleGetSsd(Mat &imgI, Mat &imgT, int u, int v)
{
	if ((imgI.cols - v) < imgT.step || (imgI.rows - u) < imgT.rows) {
		cout << "Wrong index! " << u << " " << v << endl;
		return -1;
	}
	int count = 0;
	for (int i = 0; i < imgT.rows; i++) 
		for (int j = 0; j < imgT.cols; j++)
				count += (int)pow(((int)imgT.at<uchar>(i, j) - (int)imgI.at<uchar>(i + u, j + v)), 2);
	return count;
}

int simpleTemplateMatching(Mat &imgI, Mat &imgT, int &u, int &v)
{
	Mat splitImgT[3];
	split(imgT, splitImgT);
	Mat splitImgI[3];
	split(imgI, splitImgI);
	int count = simpleGetSsd(splitImgI[0], splitImgT[0], 0, 0);
	count += simpleGetSsd(splitImgI[1], splitImgT[1], 0, 0);
	count += simpleGetSsd(splitImgI[2], splitImgT[2], 0, 0);
	for (int i = 0; i < imgI.rows - imgT.rows + 1; i++) {
		cout << i << "/" << imgI.rows - imgT.rows + 1 << endl;
		for (int j = 0; j < imgI.cols - imgT.cols + 1; j++)
		{
			int temp = simpleGetSsd(splitImgI[0], splitImgT[0], i, j);
			temp += simpleGetSsd(splitImgI[1], splitImgT[1], i, j);
			temp += simpleGetSsd(splitImgI[2], splitImgT[2], i, j);
			if (temp < count)
			{
				count = temp;
				u = i; v = j;
			}
		}
	}
	cout << " u is " << u << "  v is " << v << "  count is " << count << endl;
	return 0;
}

int getIntegralGraph(Mat &imgI, long long int** integralGraph) {
	long long int integral_row = 0;
	for (int j = 0; j < imgI.cols; j++) {
		integral_row += (int)imgI.at<uchar>(0, j) * (int)imgI.at<uchar>(0, j);
		integralGraph[0][j] = integral_row;
	}
	for (int i = 1; i < imgI.rows; i++) {
		integral_row = 0;
		for (int j = 0; j < imgI.cols; j++) {
			integral_row += (int)imgI.at<uchar>(i, j) * (int)imgI.at<uchar>(i, j);
			integralGraph[i][j] = integral_row + integralGraph[i - 1][j];
		}
	}
	return 0;
}

void getConvolve(Mat A, Mat B, Mat& C)
{ 
	C.create(abs(A.rows - B.rows) + 1, abs(A.cols - B.cols) + 1, A.type());
	Size dftSize;
	dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
	Mat tempA(dftSize, A.type(), Scalar::all(0));
	Mat tempB(dftSize, B.type(), Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB respectively, with 0s in other space
	Mat roiA(tempA, Rect(0, 0, A.cols, A.rows));
	A.copyTo(roiA);
	Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
	B.copyTo(roiB);

	dft(tempA, tempA, 0, A.rows);
	dft(tempB, tempB, 0, B.rows);
	//mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT);
	mulSpectrums(tempA, tempB, tempA, DFT_REAL_OUTPUT);  
	idft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

	// copy the result back to C.  
	//tempA(Rect(tempA.cols - C.cols, tempA.rows - C.rows, tempA.cols, tempA.rows)).copyTo(C);
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
}

long long int fastGetSsd(Mat &imgI, Mat &imgT, int u, int v, long long int TSqure, long long int** integralGraph, Mat imgConvolve)
{
	if ((imgI.cols - v) < imgT.cols || (imgI.rows - u) < imgT.rows) {
		cout << "Wrong index!" << endl;
		return -1;
	}
	long long int count;
	if (u != 0) {
		if (v != 0) {
			count = TSqure + integralGraph[u - 1][v - 1] + integralGraph[u + imgT.rows - 1][v + imgT.cols - 1] - 
				integralGraph[u - 1][v + imgT.cols - 1] - integralGraph[u + imgT.rows - 1][v - 1];
		}
		else {
			count = TSqure + integralGraph[u + imgT.rows - 1][v + imgT.cols - 1] - integralGraph[u - 1][v + imgT.cols - 1];
		}
	}
	else {
		if (v != 0) {
			count = TSqure + integralGraph[u + imgT.rows - 1][v + imgT.cols - 1] - integralGraph[u + imgT.rows - 1][v - 1];
		}
		else {
			count = TSqure + integralGraph[u + imgT.rows - 1][v + imgT.cols - 1];
		}
	}
	count -= 2 * (long long int)imgConvolve.at<float>(u, v);
	return count;
}

int fastTemplateMatching(Mat &imgI, Mat &imgT, int &u, int &v)
{
	// get TSqure
	long long int TSqure[3];
	TSqure[0] = 0; TSqure[1] = 0; TSqure[2] = 0;
	Mat splitImgT[3];
	split(imgT, splitImgT);
	for (int i = 0; i < imgT.rows; i++) {
		for (int j = 0; j < imgT.cols; j++) {
			TSqure[0] += (int)splitImgT[0].at<uchar>(i, j) * (int)splitImgT[0].at<uchar>(i, j);
			TSqure[1] += (int)splitImgT[1].at<uchar>(i, j) * (int)splitImgT[1].at<uchar>(i, j);
			TSqure[2] += (int)splitImgT[2].at<uchar>(i, j) * (int)splitImgT[2].at<uchar>(i, j);
		}
	}
	
	// get integral graph of pixel squre
	long long int*** integralGraph = (long long int***)malloc(sizeof(long long int**)*3);
	for (int h = 0; h < 3; h++) {
		integralGraph[h] = (long long int**)malloc(sizeof(long long int*)*imgI.rows);
		for (int i = 0; i < imgI.rows; i++)
			integralGraph[h][i] = (long long int*)malloc(sizeof(long long int)*imgI.cols);
	}
	Mat splitImgI[3];
	split(imgI, splitImgI);
	getIntegralGraph(splitImgI[2], integralGraph[2]);
	getIntegralGraph(splitImgI[1], integralGraph[1]);
	getIntegralGraph(splitImgI[0], integralGraph[0]);
	
	// get convolve graph
	Mat imgConvolve[3];
	Mat floatI = Mat_<float>(splitImgI[0]);// change image type into double 
	Mat floatT = Mat_<float>(splitImgT[0]);
	//getConvolve(floatI, floatT, imgConvolve[0]);
	filter2D(floatI, imgConvolve[0], floatI.depth(), floatT, Point(1, 1));
	floatI = Mat_<float>(splitImgI[1]);
	floatT = Mat_<float>(splitImgT[1]);
	//getConvolve(floatI, floatT, imgConvolve[1]);
	filter2D(floatI, imgConvolve[1], floatI.depth(), floatT, Point(0, 0));
	floatI = Mat_<float>(splitImgI[2]);
	floatT = Mat_<float>(splitImgT[2]);
	//getConvolve(floatI, floatT, imgConvolve[2]);
	filter2D(floatI, imgConvolve[2], floatI.depth(), floatT, Point(0, 0));
	
	// fing minimum ssd
	long long int count = fastGetSsd(imgI, imgT, 0, 0, TSqure[0], integralGraph[0], imgConvolve[0]);
	count += fastGetSsd(imgI, imgT, 0, 0, TSqure[1], integralGraph[1], imgConvolve[1]);
	count += fastGetSsd(imgI, imgT, 0, 0, TSqure[2], integralGraph[2], imgConvolve[2]);
	for (int i = 0; i < imgI.rows - imgT.rows + 1; i++)
		for (int j = 0; j < imgI.cols - imgT.cols + 1; j++)
		{
			long long int temp = fastGetSsd(imgI, imgT, i, j, TSqure[0], integralGraph[0], imgConvolve[0]);
			temp += fastGetSsd(imgI, imgT, i, j, TSqure[1], integralGraph[1], imgConvolve[1]);
			temp += fastGetSsd(imgI, imgT, i, j, TSqure[2], integralGraph[2], imgConvolve[2]);
			if (temp < count)
			{
				count = temp;
				u = i; v = j;
			}
		}
	cout << " u is " << u << "  v is " << v << "  count is " << count << endl;
	return 0;
}

int main()
{
	//Mat img1 = imread("Lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("img1", img1);
	//getFrequencyDomain(img1);
	
	//Mat img2 = imread("img2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//frequencyFilter(img2);

	Mat imgI = imread("I.jpg");
	Mat imgT = imread("T.jpg");
	int u = -1, v = -1;
	clock_t starttimg, endtimg;
	starttimg = clock();
	simpleTemplateMatching(imgI, imgT, u, v);
	endtimg = clock();
	cout << "Time: " << (double)(endtimg - starttimg) << endl;
	u = -1, v = -1;
	starttimg = clock();
	fastTemplateMatching(imgI, imgT, u, v);
	endtimg = clock();
	cout << "Time: " << (double)(endtimg - starttimg) << endl;

	return 0;
}