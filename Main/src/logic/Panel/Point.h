/******************************************************************************
* Point.h
*
* The Point class provides common uses for a x,y position.
*
******************************************************************************/

#pragma once
#include <math.h>
#include <list>
using std::list;

///
/// NearestInt
/// 
/// This function will return the nearest integer of a double.  It is 
/// convenient function when switching from CAD to Pixels.
/// Rudd has implemented this function as well.  It can be called through
/// his library as nint for return as double or inint for return as int.
static inline double NearestIntD(double number)	{ return floor(number+0.5); }
static inline int    NearestInt(double number)	{ return (int) NearestIntD(number); }


///
/// simple point struct - with math functions.
///
struct Point
{
	Point( double xin=0.0, double yin=0.0 ) : x(xin), y(yin) {}
	double x,y;

	void rot(double t)
	{
		static double th    = 0;
		static double sinth = 0;
		static double costh = 1;
		if( t!=th )
		{
			th=t;
			sinth=sin(t);
			costh=cos(t);
		}
		double tmpx = costh * x - sinth * y;
		double tmpy = sinth * x + costh * y;
		x = tmpx;
		y = tmpy;
	}

};

inline Point operator-( const Point& p )                   { return Point(-p.x, -p.y); }
inline Point operator+( const Point& p1, const Point& p2 ) { return Point(p1.x+p2.x,p1.y+p2.y); }
inline Point operator-( const Point& p1, const Point& p2 ) { return Point(p1.x-p2.x,p1.y-p2.y); }
inline Point operator*( const Point& p1, const Point& p2 ) { return Point(p1.x*p2.x,p1.y*p2.y); }
inline Point operator*( const Point& p,  double a )        { return Point(p.x*a,p.y*a); }
inline Point operator/( const Point& p,  double a )        { return Point(p.x/a,p.y/a); }
inline Point& operator+=( Point& p1, const Point& p2 )     { p1.x+=p2.x; p1.y+=p2.y; return p1; }
inline Point& operator-=( Point& p1, const Point& p2 )     { p1.x-=p2.x; p1.y-=p2.y; return p1; }
inline Point& operator*=( Point& p1, const Point& p2 )     { p1.x*=p2.x; p1.y*=p2.y; return p1; }
inline Point& operator*=( Point& p1, double a )            { p1.x*=a; p1.y*=a; return p1; }

typedef list<Point> PointList;
