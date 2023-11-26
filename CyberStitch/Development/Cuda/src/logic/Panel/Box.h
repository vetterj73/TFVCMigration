/******************************************************************************
* Box.h
*
* The Box class provides common uses for bounding boxes defined by two points.
*
******************************************************************************/

#pragma once 

#include "Point.h"

#include <math.h>

class Box
{
public:

	Box(){ p1 = Point(); p2 = Point(); }
	Box(Point point1, Point point2) { p1=point1; p2=point2; }

	//
	//     -----p2
	//    |     |
	//    |  *  |
	//    |     |
	//   p1-----
	//
	// y+
	// |_ x+
	Point p1, p2;

	Point  Center()	{ return (p1+p2)/2.0; }
	double Height() { return fabs(p2.y-p1.y); }
	double Width()  { return fabs(p2.x-p1.x); }

	Point Max();
	Point Min();

	bool Square();

	void Rotate(double radians);

	bool operator==(double a);
	Box operator+(double a);
	Box operator-(double a);
	Box operator/(double a);
	Box operator*(double a);
};

