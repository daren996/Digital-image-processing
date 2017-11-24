#include <iostream>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

void getChannel(unsigned char *img_data, unsigned char *cimg_data, int rows, int cols, int img_step, int cimg_step, int c)
{
	unsigned char *p = img_data;
	for (int y = 0; y < rows; ++y, p += img_step)
	{
		unsigned char *pp = p;
		for (int x = 0; x < cols; ++x, pp += 4)
		{
			cimg_data[y * cimg_step + x] = *(pp + c - 1);
		}
	}
}

void getChannel(const Mat &img, Mat &cimg, int c)
{
	cimg.create(img.size(), CV_8UC1);
	getChannel(img.data, cimg.data, img.rows, img.cols, img.step, cimg.step, c);
}

void getMerge(const Mat &img, Mat & background, Mat &merge_img)
{
	merge_img.create(img.size(), CV_8UC4);
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols * 4; x += 4)
		{
			double alpha = (double)img.data[y * img.step + x + 3] / (double)255;
			merge_img.data[y * merge_img.step + x] = (1 - alpha) * background.data[y * background.step + x] +alpha * img.data[y * img.step + x];
			merge_img.data[y * merge_img.step + x + 1] = (1 - alpha) * background.data[y * background.step + x + 1] + alpha * img.data[y * img.step + x + 1];
			merge_img.data[y * merge_img.step + x + 2] = (1 - alpha) * background.data[y * background.step + x + 2] + alpha * img.data[y * img.step + x + 2];
			merge_img.data[y * merge_img.step + x + 3] = 255;
		}
}

int main()
{
	Mat img = imread("a.png", -1);
	imshow("a", img);

	Mat cimg;
	getChannel(img, cimg, 4); // Get alpha channel from image img. 
	imshow("cimg", cimg);

	Mat background = imread("background.png", -1);
	imshow("background", background);
	Mat merge_img;
	getMerge(img, background, merge_img); // Replace a new background for image img by Alpha Blends. 
	imshow("merge", merge_img);

	waitKey();

	return 0;
}