#include <iostream>
#include <opencv2/highgui.hpp>
#include<vector>
#include<math.h>
using namespace cv;
using namespace std;	

/*
Generate Gauss Model
G(x,y)=exp(-(x^2+y^2)/2*sigma^2)/(2*pi*sigma^2)
*/
const double pi = 3.1415;
vector<vector<double>> Gauss_template(double sigma, int size) {
	int xcore = size / 2, ycore = size / 2;
	vector<vector<double>> res;
	double base = 1.0 / 2 / pi / sigma / sigma;
	for (int x = 0; x < size; x++) {
		vector<double>v;
		for (int y = 0; y < size; y++) {
			double t1 = (pow(x - xcore, 2) + pow(y - ycore, 2)) / 2.0 / sigma / sigma;
			double temp = base*exp(-t1);
			v.push_back(temp);
		}
		res.push_back(v);
	}
	return res;
}

// Generate Gauss Model - single dimension
vector<double> Gauss_single_template(double sigma, int size) {
	int xcore = size / 2;
	vector<double> res;
	double base = 1.0 / 2 / pi / sigma / sigma;
	for (int x = 0; x < size; x++) {
		double t1 = pow(x - xcore, 2) / 2.0 / sigma / sigma;
		double temp = base*exp(-t1);
		res.push_back(temp);
	}
	return res;
}

//Gauss Filter
void Gauss_Filter(const Mat &input, Mat &output, double sigma) {
	output.create(input.size(), input.type());
	int size = abs(6.0 * sigma - 1.0) / 2 * 2 + 1;
	vector<vector<double>> gaussTem = Gauss_template(sigma, size);
	int rows = input.rows, cols = input.cols;
	int start = size / 2;
	int step = input.step;
	for (int m = 0; m <rows; m++) {
		for (int n = 0; n < cols; n++) {
			if (m >= start && m < rows - start && n >= start && n < cols - start)
			{
				double sum = 0;
				for (int i = -start + m; i <= start + m; i++) 
					for (int j = -start + n; j <= start + n; j++) 
						sum += input.data[i*step + j] * gaussTem[i - m + start][j - n + start];
				output.data[m*step + n] = uchar(sum);
			}
			else
				output.data[m*step + n] = input.data[m*step + n];
		}
	}
}

//Gauss Filter - accelerated by separability
void Gauss_Filter_accelerated(const Mat &input, Mat &output, double sigma) {
	output.create(input.size(), input.type());
	int size = abs(6.0 * sigma - 1.0) / 2 * 2 + 1;
	vector<double> gaussTem = Gauss_single_template(sigma, size);
	int rows = input.rows, cols = input.cols;
	int start = size / 2;
	int step = input.step;
	for (int m = 0; m <rows; m++) {
		for (int n = 0; n < cols; n++) {
			if (m >= start && m < rows - start && n >= start && n < cols - start)
			{
				double sum = 0;
				for (int i = -start + n; i <= start + n; i++)
					sum += input.data[m*step + i] * gaussTem[i - n + start];
				output.data[m*step + n] = uchar(sum);
			}
			else
				output.data[m*step + n] = input.data[m*step + n];
		}
	}
	for (int m = 0; m <rows; m++) {
		for (int n = 0; n < cols; n++) {
			if (m >= start && m < rows - start && n >= start && n < cols - start)
			{
				double sum = 0;
				for (int i = -start + m; i <= start + m; i++)
					sum += input.data[i*step + n] * gaussTem[i - m + start];
				output.data[m*step + n] = uchar(sum);
			}
		}
	}
}

// mean filter
void Mean_Filter(const Mat& input, Mat& output, int window_size) {
	output.create(input.size(), input.type());
	int rows = input.rows, cols = input.cols;
	int start = window_size / 2;
	for (int m = 0; m <rows; m++) {
		for (int n = 0; n < cols; n++) {
			if (m >= start && m < rows - start && n >= start && n < cols - start)
			{
				int sum = 0;
				for (int i = -start + m; i <= start + m; i++) {
					for (int j = -start + n; j <= start + n; j++) {
						sum += input.at<uchar>(i, j);
					}
				}
				output.at<uchar>(m, n) = uchar(sum / window_size / window_size);
			}
			else
			{
				output.at<uchar>(m, n) = input.at<uchar>(m, n);
			}
		}
	}
}

// get integral graph
vector<vector<int>> Get_IntegralGraph(const Mat &input) {
	vector<vector<int>> res;
	int rows = input.rows;
	int cols = input.cols;
	for (int x = 0; x < rows; x++) {
		vector<int>v;
		int sum = 0;
		for (int y = 0; y < cols; y++) {
			if (x == 0)
			{
				sum += input.at<uchar>(x, y);
				v.push_back(sum);
			}
			else 
			{
				sum += input.at<uchar>(x, y);
				int temp = sum + res[x-1][y];
				v.push_back(temp);
			}
		}
		res.push_back(v);
	}
	return res;
}

// mean filter with integral graph
void Mean_Filter_IntegralGraph(const Mat& input, Mat& output, int window_size) {
	output.create(input.size(), input.type());
	vector<vector<int>> integralGraph = Get_IntegralGraph(input);
	int rows = input.rows, cols = input.cols;
	int start = window_size / 2;
	for (int m = 0; m <rows; m++) {
		for (int n = 0; n < cols; n++) {
			if (m >= start && m < rows - start && n >= start && n < cols - start)
			{
				int sum = integralGraph[m+start][n+start] - integralGraph[m + start][n - start] - 
					integralGraph[m - start][n + start] + integralGraph[m - start][n - start];
				output.at<uchar>(m, n) = uchar(sum / window_size / window_size);
			}
			else
			{
				output.at<uchar>(m, n) = input.at<uchar>(m, n);
			}
		}
	}
}

int main()
{
	Mat img = imread("Lena.jpg", 0);
	imshow("Lena", img);

	double sigma = 0.5;
	Mat img_gauss;
	Gauss_Filter_accelerated(img, img_gauss, sigma);
	imshow("Lena_gaussfilter", img_gauss);

	Mat img_mean;
	int window_size = 5;
	Mean_Filter_IntegralGraph(img, img_mean, window_size);
	imshow("Lena_meanfilter", img_mean);

	waitKey();

	return 0;
}