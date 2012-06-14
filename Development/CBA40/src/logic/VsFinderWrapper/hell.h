// 
// File: hell.h
// Description: Perform transformations with quadratic terms
//
// Created: February 18, 2002
// Author: Adam Reinhardt 
// 
// Copyright (c) 2002 CyberOptics Corp. 
//
// $Author: Adam $ 
// $Revision: 11 $
// $Header: /Products/SMT Systems/Ghidorah/Source/QNX/Realtime/Development/hell.h 11    4/02/02 2:49p Adam $
#ifndef __HELL_H__
#define __HELL_H__

// The math in this class is transcribed from 
// http://doc/Research/ruddpapers/quadform.pdf
//
// This paper discuss the case for x,y,z space.
//

/*
Dante Alighieri (dän´tâ ä´lê-gyè´rê, dàn´tê)
     1265-1321
     Italian poet whose masterpiece, The Divine Comedy (completed
     1321), details his visionary progress through Hell and
     Purgatory, escorted by Virgil, and through Heaven, guided by
     his lifelong idealized love Beatrice.
     - Dan´tean adjective & noun
     - Dantesque´ (dän-tèsk´, dàn-) adjective

     Excerpted from The American Heritage® Dictionary of the
     English Language, Third Edition  © 1996 by Houghton Mifflin
     Company. Electronic version licensed from INSO Corporation;
     further reproduction and distribution in accordance with the
     Copyright Law of the United States. All rights reserved.

Above the gate to the inferno was the inscription "Abandon all hope, ye
who enter here."

*/
#include <stdlib.h>

class matrix4;

class quadratic
{
public:
	double x2,xy,xz,x, y2,yz,y, z2,z,c;
	
	int linear() const
	{
		return !(x2||xy||xz||y2||yz||z2);
	}

	quadratic(double nx2=0, double nxy=0, double nxz=0, double nx=0,
		double ny2=0, double nyz=0, double ny=0,
			double nz2=0, double nz=0, double nc=0):
		x2(nx2), xy(nxy), xz(nxz), x(nx),
		       y2(ny2), yz(nyz), y(ny),
		              z2(nz2), z(nz),
		                     c(nc){}

	quadratic(const matrix4 &m);

	quadratic operator*(double s) const { return quadratic(*this)*=s; }
	quadratic operator*=(double s)
	{
		x2*=s; xy*=s; xz*=s; x*=s;
		       y2*=s; yz*=s; y*=s;
			          z2*=s; z*=s;
						     c*=s;
		return (*this);
	}
	double operator()(const vector3 &v) const
	{
		return x2*v.x*v.x + xy*v.x*v.y + xz*v.x*v.z + x*v.x
		                  + y2*v.y*v.y + yz*v.y*v.z + y*v.y
		                               + z2*v.z*v.z + z*v.z
		                                            + c;
	}

	quadratic operator+(const quadratic &v) const
		{ return quadratic(
				x2+v.x2,xy+v.xy,xz+v.xz,x+v.x,
					y2+v.y2, yz+v.yz, y+v.y,
						z2+v.z2, z+v.z, c+v.c); }
	quadratic operator-(const quadratic &v) const
		{ return quadratic(
				x2-v.x2,xy-v.xy,xz-v.xz,x-v.x,
					y2-v.y2, yz-v.yz, y-v.y,
						z2-v.z2, z-v.z, c-v.c); }


};

class matrix4
{
public:
	double a[4][4];
	
	matrix4() {for(int i=0;i<4;i++)for(int j=0;j<4;j++) a[i][j]=(i==j);}
	matrix4(double set){ for(int i=0;i<16;i++) a[0][i]=set; }
	matrix4(const quadratic &q)
	{
		for(int i=0;i<16;i++) a[0][i]=0;

		matrix4 &m=*this;
		m(0,0)=q.x2;
		m(0,1)=q.xy;
		m(0,2)=q.xz;
		m(0,3)=q.x;
		
		m(1,1)=q.y2;
		m(1,2)=q.yz;
		m(1,3)=q.y;
		
		m(2,2)=q.z2;
		m(2,3)=q.z;

		m(3,3)=q.c;
	}

	matrix4(const affine3 &A)
	{
		matrix4 &m=*this;

		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				m(i,j)=A.A(i,j);
		m(0,3)=A.b.x;
		m(1,3)=A.b.y;
		m(2,3)=A.b.z;
		m(3,0)=m(3,1)=m(3,2)=0;
		m(3,3)=1;
	}

	matrix4 transpose() const;

	matrix4 &operator*=(double s);
	matrix4 operator*(double s) const { return matrix4(*this)*=s; }
	matrix4 operator+(const matrix4 &n) const;
	matrix4 operator-(const matrix4 &n) const;
	matrix4 &operator/=(double s) { return operator*=(1/s); }
	matrix4 operator/(double s) const { return matrix4(*this)/=s; }
	matrix4 operator*(const matrix4 &m) const;
	matrix4 &operator*=(const matrix4 &m) { return *this=*this*m; }
	matrix4 operator()(const matrix4 &m) const { return *this * m; }
	double& operator()(int m, int n) { return a[m][n]; }
	double operator()(int m, int n) const { return a[m][n]; }
};

class quad_trans
{
public:
	quadratic Cx;
	quadratic Cy;
	quadratic Cz;

	int quad() const
	{
		return !(Cx.linear() && Cy.linear() && Cz.linear());
	}

	// create the unity transformation
	quad_trans()
	{
		Cx.x=Cy.y=Cz.z=1;
	}

	quad_trans(const affine3 &a)
	{
		Cx.x=a.A(0,0);
		Cx.y=a.A(0,1);
		Cx.z=a.A(0,2);
		Cx.c=a.b.x;

		Cy.x=a.A(1,0);
		Cy.y=a.A(1,1);
		Cy.z=a.A(1,2);
		Cy.c=a.b.y;

		Cz.x=a.A(2,0);
		Cz.y=a.A(2,1);
		Cz.z=a.A(2,2);
		Cz.c=a.b.z;
	}

	quad_trans rotate(int axis, double cycles) const
		{return quad_trans(affine3().rotate(axis, cycles));}
	quad_trans scale(const vector3 &s) const
		{return quad_trans(affine3().scale(s));}
	quad_trans offset(const vector3 &o) const
		{return quad_trans(affine3().offset(o));}

	vector3 operator()(const vector3 &v) const
	{
		vector3 n;

		n.x=Cx(v);
		n.y=Cy(v);
		n.z=Cz(v);
		return n;
	}

	quad_trans operator()(const affine3 &af) const;
	quad_trans operator()(const quad_trans &q) const;

	quad_trans operator-(const quad_trans &q)
	{
		quad_trans r;

		r.Cx=Cx-q.Cx;
		r.Cy=Cy-q.Cy;
		r.Cz=Cz-q.Cz;

		return r;
	}

	quad_trans invert() const;
	quad_trans inertial() const
	{
		quad_trans q=(*this);
		q.Cx.c=0;
		q.Cy.c=0;
		q.Cz.c=0;
		return q;
	}

	affine3 linear_frame() const;
};

/*
quad_frame handles forward and reverse transformations. It
handle concatenating affine and quadratic transformations
together.

Short coming can be specified in two items, only one quadratic 
can be applied in a stack and only small quadratic values can be 
tolerated for the invert function to work.

*/
class quad_frame
{
public:
	quad_trans f, r; // forward and reverse transforms

	quad_frame() {} // identity transformations
	quad_frame(const quad_trans &fwd, const quad_trans &rev) : f(fwd), r(rev) {}

	quad_frame(const frame3& fr): f(fr.f), r(fr.r) {}

	quad_frame rotate(int axis, double angle) const
		{ return quad_frame(quad_trans().rotate(axis,angle),
				quad_trans().rotate(axis,-angle))(*this); }
	quad_frame scale(const vector3 &v) const
		{ return quad_frame(quad_trans().scale(v), 
				quad_trans().scale(1/v))(*this); }
	quad_frame offset(const vector3 &v) const
		{ return quad_frame(quad_trans(v),quad_trans(-v))(*this); }
	quad_frame invert() const;
	quad_frame inertial() const { return quad_frame(f.inertial(), r.inertial()); }

	// build a frame consisting of this frame atop another frame
	// b2c(a2b) = a2c
	quad_frame operator()(const quad_frame &base) const;

	// build a frame that goes from this to that, assuming the same base
	// a2c >> b2c = a2b
	quad_frame operator>>(const quad_frame &dest) const;

	vector3 operator()(const vector3 &v) const { return f(v); }
	vector3 invert(const vector3 &v) const { return r(v); }
	vector3 inertial(const vector3 &v) const { return f.inertial()(v); }

	// return the transformation when zeroing out the quadratic terms of the transformation
	frame3 linear_frame() const;
};

void print_quad_frame(int verbose, const char *str, const quad_frame &qf);
void print_quad_trans(int verbose, const char *str, const quad_trans &q);
void print_quadratic(int verbose, const char *str, const quadratic &q);

#endif