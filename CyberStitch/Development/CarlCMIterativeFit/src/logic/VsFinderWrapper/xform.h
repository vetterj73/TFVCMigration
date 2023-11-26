//
// File: xform.h
// Description: defines neat classes for coordinate transforms
// Created: June 24, 1999 
// Author: Adam Reinhardt 
// 
// Copyright (c) 1999,2000,2001,2002 CyberOptics Corp. 
//
// $Author: Adam $ 
// $Revision: 16 $
// $Header: /Products/SMT Systems/Ghidorah/Source/QNX/Realtime/Development/xform.h 16    3/22/02 3:57p Adam $
#ifndef __XFORM__
#define __XFORM__

#include <math.h>

class vector3
{
public:
	double x,y,z;

	vector3(double nx=0, double ny=0, double nz=0) 
		: x(nx), y(ny), z(nz) {}

	double length() const
		{ return sqrt(x*x+y*y+z*z); }
	vector3 normalize() const
		{ return operator/(length()); }

	vector3 operator*(double s) const
		{ return vector3(x*s,y*s,z*s); }
	vector3 operator/(double s) const
		{ return vector3(x/s,y/s,z/s); }
	vector3 operator+(const vector3 &v) const
		{ return vector3(x+v.x,y+v.y,z+v.z); }
	vector3 operator-(const vector3 &v) const
		{ return vector3(x-v.x,y-v.y,z-v.z); }
	vector3 operator-() const
		{ return operator*(-1); }
	double operator*(const vector3 &v) const
		{ return x*v.x+y*v.y+z*v.z; }
	vector3 operator^(const vector3 &v) const
		{ return vector3(y*v.z-z*v.y,
				 z*v.x-x*v.z,
				 x*v.y-y*v.x); }
};

inline vector3 operator*(double s, const vector3 &v)
{
	return vector3(s*v.x, s*v.y, s*v.z);
}

inline vector3 operator/(double s, const vector3 &v)
{
	return vector3(s/v.x, s/v.y, s/v.z);
}


class matrix3
{
public:
	double a[3][3];
	
	matrix3() { for(int i=0;i<9;i++) a[0][i]=!(i&3); }
	matrix3(double set){ for(int i=0;i<9;i++) a[0][i]=set; }
	
	matrix3 rotate(int axis, double angle) const;
	matrix3 scale(const vector3 &v) const;
	matrix3 invert() const;
	matrix3 transpose() const;
	double det() const;
	double det(int i, int j) const;

	matrix3 &operator*=(double s);
	matrix3 operator*(double s) const { return matrix3(*this)*=s; }
	matrix3 &operator/=(double s) { return operator*=(1/s); }
	matrix3 operator/(double s) const { return matrix3(*this)/=s; }
	matrix3 operator*(const matrix3 &m) const;
	matrix3 &operator*=(const matrix3 &m) { return *this=*this*m; }
	vector3 operator*(const vector3 &v) const;
	vector3 operator()(const vector3 &v) const { return *this * v; }
	matrix3 operator()(const matrix3 &m) const { return *this * m; }
	double& operator()(int m, int n) { return a[m][n]; }
	double operator()(int m, int n) const { return a[m][n]; }
};

class affine3
{
public:
	matrix3 A;
	vector3 b;

	affine3() {} // identity transformation
	affine3(const matrix3 &nA) : A(nA) {}
	affine3(const vector3 &nb) : b(nb) {}
	affine3(const matrix3 &nA, const vector3 &nb) :	A(nA), b(nb) {}

	affine3 rotate(int axis, double angle) const
		{ return affine3(matrix3().rotate(axis, angle))(*this); }
	affine3 scale(const vector3 &v) const
		{ return affine3(matrix3().scale(v))(*this); }
	affine3 offset(const vector3 &v) const
		{ return affine3(v)(*this); }

	affine3 invert() const
		{ return affine3(A.invert(), A.invert()*(-b) );}
	affine3 operator()(const affine3 &a) const 
		{ return affine3(A(a.A), A(a.b)+b); }
	vector3 operator()(const vector3 &x) const 
		{ return A(x)+b; }
};

class frame3
{
public:
	affine3 f, r; // forward and reverse transforms

	frame3() {} // identity transformations
	frame3(const affine3 &fwd, const affine3 &rev) : f(fwd), r(rev) {}

	frame3 rotate(int axis, double angle) const
		{ return frame3(affine3().rotate(axis,angle),
				affine3().rotate(axis,-angle))(*this); }
	frame3 scale(const vector3 &v) const
		{ return frame3(affine3().scale(v), 
				affine3().scale(1/v))(*this); }
	frame3 offset(const vector3 &v) const
		{ return frame3(affine3(v),affine3(-v))(*this); }
	frame3 invert() const { return frame3(r, f); }
	frame3 inertial() const { return frame3(affine3(f.A), affine3(r.A)); }

	// build a frame consisting of this frame atop another frame
	// b2c(a2b) = a2c
	frame3 operator()(const frame3 &base) const
		{ return frame3(f(base.f), base.r(r)); }

	// build a frame that goes from this to that, assuming the same base
	// a2c >> b2c = a2b
	frame3 operator>>(const frame3 &dest) const
		{ return frame3(dest.r(f), r(dest.f)); }

	// for use with hell when quad_frame is not used.
	frame3 linear_frame() const {return (*this);}
	vector3 operator()(const vector3 &v) const { return f(v); }
	vector3 invert(const vector3 &v) const { return r(v); }
	vector3 inertial(const vector3 &v) const { return f.A(v); }
	int quad() const {return 0;}
};

#include "hell.h"

#if 1

#define QUAD_AS_HELL
typedef quad_frame hell;

#else

#define FRAME3_AS_HELL
typedef frame3 hell;

#endif

#endif
