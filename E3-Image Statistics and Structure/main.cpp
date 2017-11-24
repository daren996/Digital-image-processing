#include<opencv2\highgui.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

/* Get the histogram of single channel image */
void calc_hist(uchar *data, int width, int height, int step, int H[256])
{
	memset(H, 0, sizeof(H[0]) * 256);
	uchar *row = data;
	for (int yi = 0; yi<height; ++yi, row += step)
		for (int xi = 0; xi<width; ++xi)
			H[row[xi]]++;
}

/* Get the histogram of an image with two channels */
void calc_hist(const Mat &input, int H1[256], int H2[256])
{
	memset(H1, 0, sizeof(H1[0]) * 256);
	memset(H2, 0, sizeof(H2[0]) * 256);
	uchar *row = input.data;
	for (int yi = 0; yi < input.rows; ++yi, row += input.step)
		for (int xi = 0; xi < input.cols * 3; xi += 2)
		{
			H1[row[xi]]++;
			H2[row[xi + 1]]++;
		}
}

/* Get the histogram of an image with three channels */
void calc_hist(const Mat &input, int H1[256], int H2[256], int H3[256])
{
	memset(H1, 0, sizeof(H1[0]) * 256);
	memset(H2, 0, sizeof(H2[0]) * 256);
	memset(H3, 0, sizeof(H3[0]) * 256);
	uchar *row = input.data;
	for (int yi = 0; yi < input.rows; ++yi, row += input.step)
		for (int xi = 0; xi < input.cols * 3; xi += 3)
		{
			H1[row[xi]]++;
			H2[row[xi + 1]]++;
			H3[row[xi + 2]]++;
		}
}

/* Histofram equalization for single channel image */
void histogram_equalization_one_channel(const Mat &input, Mat &output)
{
	int H[256];
	calc_hist(input.data, input.cols, input.rows, input.step, H);
	double S[256];
	double n = input.cols * input.rows;
	S[0] = (double)H[0] / n;
	for (int i = 1; i < 256; ++i)
		S[i] = S[i - 1] + (double)H[i] / n;
	int SS[256];
	for (int i = 0; i < 256; ++i)
		SS[i] = S[i] * 256;
	output.create(input.size(), input.type());
	uchar *p = input.data;
	uchar *pp = output.data;
	for (int yi = 0; yi < output.rows; ++yi, p += input.step, pp += output.step)
		for (int xi = 0; xi < output.cols; ++xi)
			pp[xi] = SS[p[xi]];
}

/* Histofram equalization for images with three channels */
void histogram_equalization_two_channel(const Mat &input, Mat &output)
{
	int H1[256]; int H2[256];
	calc_hist(input, H1, H2);
	double S1[256]; double S2[256];
	double n = input.cols * input.rows;
	S1[0] = (double)H1[0] / n; S2[0] = (double)H2[0] / n;
	for (int i = 1; i < 256; ++i)
	{
		S1[i] = S1[i - 1] + (double)H1[i] / n;
		S2[i] = S2[i - 1] + (double)H2[i] / n;
	}
	int SS1[256]; int SS2[256];
	for (int i = 0; i < 256; ++i)
	{
		SS1[i] = S1[i] * 256;
		SS2[i] = S2[i] * 256;
	}
	output.create(input.size(), input.type());
	uchar *p = input.data;
	uchar *pp = output.data;
	for (int yi = 0; yi < output.rows; ++yi, p += input.step, pp += output.step)
		for (int xi = 0; xi < output.cols * 2; xi += 2)
		{
			pp[xi] = SS1[p[xi]];
			pp[xi + 1] = SS2[p[xi + 1]];
		}
}

/* Histofram equalization for images with three channels */
void histogram_equalization_three_channel(const Mat &input, Mat &output)
{
	int H1[256]; int H2[256]; int H3[256];
	calc_hist(input, H1, H2, H3);
	double S1[256]; double S2[256]; double S3[256];
	double n = input.cols * input.rows;
	S1[0] = (double)H1[0] / n; S2[0] = (double)H2[0] / n;	S3[0] = (double)H3[0] / n;
	for (int i = 1; i < 256; ++i)
	{
		S1[i] = S1[i - 1] + (double)H1[i] / n;
		S2[i] = S2[i - 1] + (double)H2[i] / n;
		S3[i] = S3[i - 1] + (double)H3[i] / n;
	}
	int SS1[256]; int SS2[256]; int SS3[256];
	for (int i = 0; i < 256; ++i)
	{
		SS1[i] = S1[i] * 256; 
		SS2[i] = S2[i] * 256;
		SS3[i] = S3[i] * 256;
	}
	output.create(input.size(), input.type());
	uchar *p = input.data;
	uchar *pp = output.data;
	for (int yi = 0; yi < output.rows; ++yi, p += input.step, pp += output.step)
		for (int xi = 0; xi < output.cols * 3; xi += 3)
		{
			pp[xi] = SS1[p[xi]]; 
			pp[xi + 1] = SS2[p[xi + 1]]; 
			pp[xi + 2] = SS3[p[xi + 2]]; 
		}
}

/* Histofram equalization for images with four channels */
void histogram_equalization_four_channel(const Mat &input, Mat &output)
{
	int H1[256]; int H2[256]; int H3[256];
	calc_hist(input, H1, H2, H3);
	double S1[256]; double S2[256]; double S3[256];
	double n = input.cols * input.rows;
	S1[0] = (double)H1[0] / n; S2[0] = (double)H2[0] / n;	S3[0] = (double)H3[0] / n;
	for (int i = 1; i < 256; ++i)
	{
		S1[i] = S1[i - 1] + (double)H1[i] / n;
		S2[i] = S2[i - 1] + (double)H2[i] / n;
		S3[i] = S3[i - 1] + (double)H3[i] / n;
	}
	int SS1[256]; int SS2[256]; int SS3[256];
	for (int i = 0; i < 256; ++i)
	{
		SS1[i] = S1[i] * 256;
		SS2[i] = S2[i] * 256;
		SS3[i] = S3[i] * 256;
	}
	output.create(input.size(), input.type());
	uchar *p = input.data;
	uchar *pp = output.data;
	for (int yi = 0; yi < output.rows; ++yi, p += input.step, pp += output.step)
		for (int xi = 0; xi < output.cols * 4; xi += 4)
		{
			pp[xi] = SS1[p[xi]];
			pp[xi + 1] = SS2[p[xi + 1]];
			pp[xi + 2] = SS3[p[xi + 2]];
		}
}

void histogram_equalization(const Mat &input, Mat &output)
{
	int n = input.channels();
	if (n == 1)
		histogram_equalization_one_channel(input, output);
	if (n == 2)
		histogram_equalization_two_channel(input, output);
	if (n == 3)
		histogram_equalization_three_channel(input, output);
	if (n == 4)
		histogram_equalization_four_channel(input, output);
}

/* Some colors */
struct Colors
{
	int R;
	int G;
	int B;
	int c;
};
/*
ºìÉ« 255,0,0
×©ºì 156,102,31
»ÆÉ« 255,255,0
·Ûºì 255,192,203
ÂÌÉ« 0,255,0
À¶É« 0,0,255
×ÏÂÞÀ¼ 138,43,226
éÙ»Æ 255,128,0
µåÀ¶É« 8,46,84
*/

/* Quick Connect Area */
void get_connect_area(const Mat &input, Mat &output)
{
	output.create(input.size(), CV_8UC3);

	Colors* colors = new Colors[9];
	colors[0] = { 255,0,0,0 };
	colors[1] = { 156,102,31,0 };
	colors[2] = { 255,255,0,0 };
	colors[3] = { 255,192,203,0 };
	colors[4] = { 0,255,0,0 };
	colors[5] = { 0,0,255,0 };
	colors[6] = { 138,43,226,0 };
	colors[7] = { 255,128,0,0 };
	colors[8] = { 8,46,84,0 };

	uchar *p = input.data;
	int* index = (int *)malloc(input.cols * input.rows * sizeof(uint));
	for (int yi = 0; yi < input.rows; yi++)
	{
		for (int xi = 0; xi < input.cols; xi++)
		{
			if (yi >= 1 && xi >= 1)
			{
				int up = p[(yi - 1)*input.step + xi];
				int left = p[yi*input.step + xi - 1];
				int center = p[yi*input.step + xi];
				if (abs(up - center) >= 10 && abs(center - left) >= 10)
					index[yi*input.step + xi] = 0;
				if (abs(up - center) <= 10 && abs(center - left) >= 10)
				{
					int k = (yi - 1) * input.step + xi;
					while (index[k] != 0)
						k = index[k];
					index[yi*input.step + xi] = k;
				}
				if (abs(up - center) >= 10 && abs(center - left) <= 10)
				{
					int k = yi*input.step + xi - 1;
					while (index[k] != 0)
						k = index[k];
					index[yi*input.step + xi] = k;
				}
				if (abs(up - center) <= 10 && abs(center - left) <= 10)
				{
					int k = yi*input.step + xi - 1;
					while (index[k] != 0)
						k = index[k];
					index[yi*input.step + xi] = k;
					int l = (yi - 1)*input.step + xi;
					while (index[l] != 0)
						l = index[l];
					if (k != l)
						index[l] = k;
				}
			}
			else
			{
				if (yi == 0 && xi >= 1)
				{
					int left = p[yi*input.step + xi - 1];
					int center = p[yi*input.step + xi];
					if (abs(center - left) >= 10)
						index[yi*input.step + xi] = 0;
					if (abs(center - left) <= 10)
					{
						int k = yi*input.step + xi - 1;
						while (index[k] != 0)
							k = index[k];
						index[yi*input.step + xi] = k;
					}
				}
				else
				{
					if (yi >= 1 && xi == 0)
					{
						int up = p[(yi - 1)*input.step + xi];
						int center = p[yi*input.step + xi];
						if (abs(up - center) >= 10)
							index[yi*input.step + xi] = 0;
						if (abs(up - center) <= 10)
						{
							int k = (yi - 1)*input.step + xi;
							while (index[k] != 0)
								k = index[k];
							index[yi*input.step + xi] = k;
						}
					}
					else
						index[yi*input.step + xi] = 0;
				}
			}
		}
	}
	
	uchar *pp = output.data;
	int i = 0;
	for (int yi = 0; yi < output.rows; yi++)
		for (int xi = 0; xi < output.cols; xi++)
		{
			int k = index[yi*input.step + xi];
			while (index[k] != 0)
				k = index[k];
			int j = 0;
			for (j = 0; j < i; j++)
			{
				if (colors[j].c == k)
				{
					pp[yi*output.step + xi * 3] = colors[j].R;
					pp[yi*output.step + xi * 3 + 1] = colors[j].G;
					pp[yi*output.step + xi * 3 + 2] = colors[j].B;
					break;
				}
			}
			if (j == i)
			{
				i++;
				colors[i].c = k;
				pp[yi*output.step + xi * 3] = colors[i].R;
				pp[yi*output.step + xi * 3 + 1] = colors[i].G;
				pp[yi*output.step + xi * 3 + 2] = colors[i].B;
			}
		}
}

int main()
{

	// Give an imaeg with single channel which is too dark 
	Mat img_dark = imread("a_dark.png", 0);
	imshow("a_dark", img_dark);	
	Mat img_dark_equalization;
	histogram_equalization(img_dark, img_dark_equalization);
	imshow("a_dark_equalization", img_dark_equalization);

	// Give an imaeg with single channel which is too light 
	Mat img_light = imread("a_light.png", 0);
	imshow("a_light", img_light);
	Mat img_light_equalization;
	histogram_equalization(img_light, img_light_equalization);
	imshow("a_light_equalization", img_light_equalization);

	// Give an imaeg with three channels 
	Mat img_color = imread("b.png");
	imshow("b", img_color);
	Mat img_color_equalization;
	histogram_equalization(img_color, img_color_equalization);
	imshow("b_equalization", img_color_equalization);
	

	// Quick Connect Area 
	Mat img_cc = imread("cc_input.png", 0);
	imshow("cc", img_cc);
	Mat img_cc_connect_area;
	get_connect_area(img_cc, img_cc_connect_area);
	imshow("cc_connect_area", img_cc_connect_area);

	waitKey();

	return 0;
}
