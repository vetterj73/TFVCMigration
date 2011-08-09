// ProjectiveAngleTest.cpp : main project file.

#include "stdafx.h"
#include "math.h"

using namespace System;

//void ImageToWorld(double T[3][3], 
//	double u, double v, 
//	double &x, double &y);

void ImageToWorld(double T[9], 
	double u, double v, 
	double &x, double &y);

int main(array<System::String ^> ^args)
{

	/*/ The transform
	double a[3][3] = { 
		{ 1.200000000480e-002,   3.220087228546e-014,   -1.554600000625e+001},
		{ 0,    1.206025102028e-002 ,    -1.171653386621e+001},
		{ 0,    -4.459318759302e-006,      1.004332228175e+000}} ; 

	for(int iy=0; iy<3; iy++)
		for(int ix=0; ix<3; ix++)
			a[iy][ix] = a[iy][ix]/a[2][2];
	
	double org[9] = {  
		8.33333333e+01,   4.79015523e-01,   1.29550000e+03,
		0.00000000e+00,   8.32762292e+01,   9.71500000e+02,
		0.00000000e+00,   3.69753395e-04,   1.00000000e+00};



	double a[9] = { 
		 1.200000000480e-002,   3.220087228546e-014,   -1.554600000625e+001,
		 0,    1.206025102028e-002 ,    -1.171653386621e+001,
		 0,    -4.459318759302e-006,      1.004332228175e+000}; 
		*/
	
	double a[9] = {
		1.09E-05, -3.74E-08, 0.155455,
		-2.62E-07,	1.19E-05, 0.0269252,
		-7.03E-06,	-1.07E-08, 1};


	for(int i=0; i<9; i++)
			a[i] = a[i]/a[8];

	double dPupilDistance = 270/1000.0;
	int rows = 1944;
	int cols = 2592;
//*	
	// Image center
	double xo, yo;
	ImageToWorld(a, (rows-1)/2.0, (cols-1)/2.0, xo, yo);

	double u1, v1, u2, v2;
	double x1, y1, x2, y2;
//** angle for x(row)	
	// (u, v) and (x, y)
	u1 = 0;
	v1 = (cols-1)/2.0;
	ImageToWorld(a, u1, v1, x1, y1);	
	u2 = rows-1;
	v2 = v1;
	ImageToWorld(a, u2, v2, x2, y2);

	// dxdu
	double temp;
	//temp = a[2][0]*u1+a[2][1]*v1+1;
	//double dxdu1 = a[0][0]/temp-a[2][0]*(a[0][0]*u1+a[0][1]*v1+a[0][2])/temp/temp;
	temp = a[6]*u1+a[7]*v1+1;
	//double dxdu1 = a[0]-a[6]*(a[0]*u1+a[1]*v1+a[2])/temp;
	//dxdu1 /= temp;
	double dxdu1 = (a[0]*a[7]-a[6]*a[1])*v1+a[0]-a[6]*a[2];
	dxdu1 = dxdu1/temp/temp;

	//temp = a[2][0]*u2+a[2][1]*v2+1;
	//double dxdu2 = a[0][0]/temp-a[2][0]*(a[0][0]*u2+a[0][1]*v2+a[0][2])/temp/temp;
	temp = a[6]*u2+a[7]*v2+1;
	//double dxdu2 = a[0]-a[6]*(a[0]*u2+a[1]*v2+a[2])/temp;
	//dxdu2 /= temp;
	double dxdu2 = (a[0]*a[7]-a[6]*a[1])*v2+a[0]-a[6]*a[2];
	dxdu2 = dxdu2/temp/temp;

	// sinTheta
	double gx = dxdu1/dxdu2;
	double sinThetaX = (1-gx)*dPupilDistance/((x1-xo)*gx-(x2-xo));
	double dThetaX = asin(sinThetaX)*180/3.14;

//** angle for y(column)
	// (u, v) and  (x, y)
	u1 = (rows-1)/2.0;
	v1 = 0;
	ImageToWorld(a, u1, v1, x1, y1);	
	u2 = u1;
	v2 = cols-1;
	ImageToWorld(a, u2, v2, x2, y2);

	// dydv
	//temp = a[2][0]*u1+a[2][1]*v1+1;
	//double dydv1 = a[1][1]/temp -a[2][1]*(a[1][0]*u1+a[1][1]*v1+a[1][2])/temp/temp;
	temp = a[6]*u1+a[7]*v1+1;
	double dydv1 = a[4]/temp -a[7]*(a[3]*u1+a[4]*v1+a[5])/temp/temp;

	//temp = a[2][0]*u2+a[2][1]*v2+1;
	//double dydv2 = a[1][1]/temp -a[2][1]*(a[1][0]*u2+a[1][1]*v2+a[1][2])/temp/temp;
	temp = a[6]*u2+a[7]*v2+1;
	double dydv2 = a[4]/temp -a[7]*(a[3]*u2+a[4]*v2+a[5])/temp/temp;
	//double dydv2 = (a[0]*a[7]-a[6]*a[1])*v2+a[0]-a[6]*a[2];
	//dydv2 = dydv2/temp/temp;

	double gy = dydv1/dydv2;
	double sinThetaY = (1-gy)*dPupilDistance/((y1-yo)*gy-(y2-yo));
	double dThetaY = asin(sinThetaY)*180/3.14;
/*/
	/*
	double cenx=(rows-1)/2.0, ceny=(cols-1)/2.0;
	//ImageToWorld(org, 0, 26.95, cenx, ceny);
	double x1, y1, x2, y2, x3, y3, x4, y4;
	ImageToWorld(a, cenx, ceny+1000, x1, y1);
	ImageToWorld(a, cenx-1000, ceny, x2, y2);
	ImageToWorld(a, cenx, ceny-1000, x3, y3);
	ImageToWorld(a, cenx+1000, ceny, x4, y4);
	double w = ((y4-y2)*(x2-x1)-(x4-x2)*(y2-y1))/((y4-y2)*(x3-x1)-(x4-x2)*(y3-y1));
	double x = x1+w*(x3-x1);
	double y = y1+w*(y3-y1);
	double d1 = sqrt((x1-x)*(x1-x)+(y1-y)*(y1-y));
	double d2 = sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y));
	double d3 = sqrt((x3-x)*(x3-x)+(y3-y)*(y3-y));
	double d4 = sqrt((x4-x)*(x4-x)+(y4-y)*(y4-y));

	double tanChi = (d2+d4)/(2*dPupilDistance);
	double dThetaX = atan((d1-d3)/(d1+d3)/tanChi) *180/3.14;
	double dvalue = (d4-d2)/(d4+d2)/tanChi;
	double dThetaY = atan((d4-d2)/(d4+d2)/tanChi) * 180/3.14;
	//*/
    Console::WriteLine(L"done thetax = "+dThetaX+";thetaY = "+dThetaY);
    return 0;
}
/*
void ImageToWorld(double T[3][3], 
	double u, double v, 
	double &x, double &y)
{
	x = u*T[0][0] + v*T[0][1] + T[0][2];
	y = u*T[1][0] + v*T[1][1] + T[1][2];
	double dTemp = u*T[2][0] + v*T[2][1] + T[2][2];

	x = x/dTemp;
	x = x/dTemp;
}*/
//*
void ImageToWorld(double T[9], 
	double u, double v, 
	double &x, double &y)
{
	x = u*T[0] + v*T[1] + T[2];
	y = u*T[3] + v*T[4] + T[5];
	double dTemp = u*T[6] + v*T[7] + T[8];

	x = x/dTemp;
	x = x/dTemp;
}//*/
