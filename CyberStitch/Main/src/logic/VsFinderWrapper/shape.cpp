// Copyright (c) 2000,2001 CyberOptics Corp. 
//
// $Author: Lneyman $ 
// $Revision: 3 $
// $Header: /Ghidorah/Windows/Lib/Win32/vsFind/shape.cpp 3     12/28/04 12:37p Lneyman $
#include <math.h>
#include <limits.h>
#include "shape.h"

polygon::polygon(const rect &r)
{
	p.push(vector3(r.x1, r.y1));
	p.push(vector3(r.x1, r.y2));
	p.push(vector3(r.x2, r.y2));
	p.push(vector3(r.x2, r.y1));
}

void polygon::operator+=(const vector3 &v)
{
	for(listiter<vector3> i(p); i.on(); i++)
		*i=*i+v; 
}

void polygon::operator*=(const affine3 &a)
{
	for(listiter<vector3> i(p); i.on(); i++)
		*i=a(*i); 
}

void polygon::operator*=(const quad_trans &q)
{
	for(listiter<vector3> i(p); i.on(); i++)
		*i=q(*i); 
}

double polygon::perimeter() const
{
	double l=0;
	listiter<vector3> i(p);
	vector3 f=*i, c;

	do
	{
		c=*i++;
		l+=(*i-c).length();
	} while(i.on());

	l+=(f-c).length();

	return l;
}

double polygon::area() const
{
	double s=0;
	listiter<vector3> i(p);
	vector3 f=*i, c;

	for(c=*i++; i.on(); c=*i++)
		s+=c.x*i->y - c.y*i->x;

	s+=c.x*f.y - c.y*f.x;

	return fabs(s/2);
}

vector3 polygon::centroid() const
{
	double x=0, y=0, a=0, ai;
	listiter<vector3> i(p);
	vector3 f=*i, c;

	for(c=*i++; i.on(); c=*i++)
	{
		ai=c.x*i->y - c.y*i->x;
		a+=ai;
		x+=ai*(i->x + c.x);
		y+=ai*(i->y + c.y);
	}

	ai=c.x*f.y - c.y*f.x;
	a+=ai;
	x+=ai*(f.x + c.x);
	y+=ai*(f.y + c.y);

	return vector3(x, y, 0)/(3*a);
}

vector3 polygon::upperleft() const
{
	vector3 upperleft=p[0];

	for(int i=0;i<p.count();i++)
	{
		if(p[i].x <=upperleft.x)
			upperleft.x=p[i].x;
		if(p[i].y <=upperleft.y)
			upperleft.y=p[i].y;
	}
	return upperleft;
}

vector3 polygon::lowerright() const
{
	vector3 lowerright=p[0];

	for(int i=0;i<p.count();i++)
	{
		if(p[i].x >=lowerright.x)
			lowerright.x=p[i].x;
		if(p[i].y >=lowerright.y)
			lowerright.y=p[i].y;
	}
	return lowerright;
}

rect polygon::bound() const
{
	rect b(INT_MAX, INT_MAX, INT_MIN, INT_MIN);
	
	for(listiter<vector3> i(p); i.on(); i++)
		b=b|rect((int)i->x, (int)i->y, (int)i->x, (int)i->y);

	return b;
}


void ellipse::operator+=(const vector3 &v)
{
	c=c+v;
}

void ellipse::operator*=(const affine3 &a)
{
	c=a(c);
	r=a.A(r);
	r.x=fabs(r.x);
	r.y=fabs(r.y);
	r.z=fabs(r.z);
}

void ellipse::operator*=(const quad_trans &q)
{
	c=q(c);
	r=q.inertial()(r);
	r.x=fabs(r.x);
	r.y=fabs(r.y);
	r.z=fabs(r.z);
}

double ellipse::perimeter() const // only approximate!
{
	return 3.1415962 * sqrt(2*(r.x*r.x + r.y*r.y));
}

double ellipse::area() const
{
	return 3.1415962*r.x*r.y;
}

vector3 ellipse::centroid() const
{
	return c;
}

rect ellipse::bound() const
{
	return rect((int)(c.x-r.x), (int)(c.y-r.y), (int)(c.x+r.x), (int)(c.y+r.y));
}



void donut::operator+=(const vector3 &v)
{
	inner+=v;
	outer+=v;
}

void donut::operator*=(const affine3 &a)
{
	inner*=a;
	outer*=a;
}
