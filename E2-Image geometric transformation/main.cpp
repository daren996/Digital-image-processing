#include<opencv2\highgui.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

/*Bilinear Interpolation*/
double bilinear(double a, double b, double c, double d, double dx, double dy)
{
	double h1 = a + dx * (b - a);
	double h2 = c + dx * (d - c);
	return (h1 + dy * (h2 - h1));
}

void Scale(const Mat &input, Mat &output, double sx, double sy)
{
	output.create(sy * input.rows, sx * input.cols, CV_8UC3);
	uchar *p = output.data;
	for (int i = 0; i < output.rows; ++i, p += output.step)
	{
		uchar *pp = p;
		for (int j = 0; j < output.cols; ++j, pp += 3)
		{
			double newx, newy, dx, dy;
			newx = j / sx;
			newy = i / sy;
			dx = newx - floor(newx);
			dy = newy - floor(newy);
			for (int k = 0; k < 3; ++k)
			{
				double a, b, c, d;
				a = input.data[(int)floor(newy) * input.step + (int)floor(newx) * 3 + k];
				b = input.data[(int)floor(newy) * input.step + (int)ceil(newx) * 3 + k];
				c = input.data[(int)ceil(newy) * input.step + (int)floor(newx) * 3 + k];
				d = input.data[(int)ceil(newy) * input.step + (int)ceil(newx) * 3 + k];
				*(pp + k) = bilinear(a, b, c, d, dx, dy);
			}
		}
	}
}

void Transform(const Mat &input, Mat &output)
{
	output.create(input.size(), CV_8UC3);
	uchar *p = output.data;
	double W, H;
	W = output.cols;
	H = output.rows;
	for (int i = 0; i < H; ++i, p += output.step)
	{
		uchar *pp = p;
		for (int j = 0; j < W; ++j, pp += 3)
		{
			double newx, newy, xprime, yprime, x, y, orix, oriy, r;
			newx = j; newy = i; 
			xprime = (newx - 0.5 * W) / (0.5 * W);
			yprime = (newy - 0.5 * H) / (0.5 * H);
			r = sqrt(xprime*xprime + yprime*yprime);
			if (r < 1)
			{
				double theta = pow((1 - r), 2);
				x = cos(theta) * xprime - sin(theta) * yprime;
				y = sin(theta) * xprime + cos(theta) * yprime;

			}
			else
			{
				x = xprime;
				y = yprime;
			}
			orix = 0.5 * W * x + 0.5 * W;
			oriy = 0.5 * H * y + 0.5 * H;
			double dx, dy;
			dx = orix - floor(orix);
			dy = oriy - floor(oriy);
			for (int k = 0; k < 3; ++k)
			{
				double a, b, c, d;
				a = input.data[(int)floor(oriy) * input.step + (int)floor(orix) * 3 + k];
				b = input.data[(int)floor(oriy) * input.step + (int)ceil(orix) * 3 + k];
				c = input.data[(int)ceil(oriy) * input.step + (int)floor(orix) * 3 + k];
				d = input.data[(int)ceil(oriy) * input.step + (int)ceil(orix) * 3 + k];
				*(pp + k) = bilinear(a, b, c, d, dx, dy);
			}
		}
	}
}

int main()
{
	Mat img = imread("a.jpg"); // The type of image a is CV_8UC3
	imshow("a", img);
	cout << img.size << endl;
	cout << img.cols << " " << img.rows << endl;

	Mat scale_img;
	Scale(img, scale_img, 1.2, 1.2); // Resize the original image
	imshow("Scale", scale_img);

	Mat transform_img;
	Transform(img, transform_img); // Transform the image
	imshow("Transform", transform_img);

	waitKey();

	return 0;
}