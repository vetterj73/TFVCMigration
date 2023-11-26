/******************************************************************************
* Box.h
*
* The Box class provides common uses for bounding boxes defined by two points.
*
******************************************************************************/

#include "Box.h"

#include <math.h>

#ifndef EPSILON
#define EPSILON 0.0000001
#endif

Point Box::Max()
{
	return Point(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
}

Point Box::Min()
{
	return Point(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
}

void Box::Rotate(double radians)
{
	if(fabs(radians)<EPSILON)
		return;

	//
	//   p[1]-----p[2]
	//    |       |
	//    |   *   |
	//    |       |
	//   p[0]-----p[3]
	//
	// y+
	// |_ x+

	Point center = Center();

	// To rotate the box, the center must be the origin
	Point p[4];
	p[0] = p1 - center;
	p[2] = p2 - center;
	
	p[1].x = p[0].x;
	p[1].y = p[2].y;
	p[3].x = p[2].x;
	p[3].y = p[0].y;

	//         y+
	//         |
	//    p[1]-|---p[2]
	//     |   |   |
	// x- -|---|---|---x+
	//     |   |   |
	//    p[0]-|---p[3]
	//         |
	//         y-

	// rotate about the origin
	for(int i=0; i<4; i++)
		p[i].rot(radians);

	// Find the max & min points, 
	Point maxPoint(-9999999.0, -9999999.0);
	Point minPoint(+9999999.0, +9999999.0);

	for(int i=0; i<4; i++)
	{
		if(p[i].x > maxPoint.x)
			maxPoint.x = p[i].x;
		if(p[i].x < minPoint.x)
			minPoint.x = p[i].x;

		if(p[i].y > maxPoint.y)
			maxPoint.y = p[i].y;
		if(p[i].y < minPoint.y)
			minPoint.y = p[i].y;
	}

	// set new bounding box and adjust back to original centroid
	p1 = minPoint + center;
	p2 = maxPoint + center;
}

bool Box::Square()
{
	if(fabs(Height()-Width())>EPSILON)
		return false;

	return true;
}

bool Box::operator==(double a)
{
	if(fabs(p1.x-a)>EPSILON)
		return false;
	else if(fabs(p1.y-a)>EPSILON)
		return false;
	else if(fabs(p2.x-a)>EPSILON)
		return false;
	else if(fabs(p2.y-a)>EPSILON)
		return false;

	return true;
}

Box Box::operator +(double a)
{
	Box b;
	b.p1.x=p1.x+a;
	b.p1.y=p1.y+a;
	b.p2.x=p2.x+a;
	b.p2.y=p2.y+a;

	return b;
}

Box Box::operator -(double a)
{
	Box b;
	b.p1.x=p1.x-a;
	b.p1.y=p1.y-a;
	b.p2.x=p2.x-a;
	b.p2.y=p2.y-a;

	return b;
}

Box Box::operator /(double a)
{
	Box b;
	b.p1.x=p1.x/a;
	b.p1.y=p1.y/a;
	b.p2.x=p2.x/a;
	b.p2.y=p2.y/a;

	return b;
}

Box Box::operator *(double a)
{
	Box b;
	b.p1.x=p1.x*a;
	b.p1.y=p1.y*a;
	b.p2.x=p2.x*a;
	b.p2.y=p2.y*a;

	return b;
}
