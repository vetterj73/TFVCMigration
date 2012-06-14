// Copyright (c) 2000,2001 CyberOptics Corp. 
//
// $Author: Mhogan $ 
// $Revision: 3 $
// $Header: /Ghidorah/Windows/Lib/Win32/vsFind/shape.h 3     1/16/07 10:04a Mhogan $
#ifndef __SHAPE_H
#define __SHAPE_H

#include "xform.h"
#include "xlist.h"
#include "rect.h"

//#include <list>
//using std::list;

class shape
{
public:
	virtual ~shape() {};
	virtual shape *copy() const=0;

	typedef void spanproc(int y, int xl, int xr, void *data);

	virtual double perimeter() const=0;
	virtual double area() const=0; // as projected onto x-y plane
	virtual vector3 centroid() const=0;
	virtual rect bound() const=0;

	virtual void scan(const rect &b, spanproc *s, void *d=0) const=0;
	virtual void trace(const rect &b, spanproc *s, void *d=0) const;

	virtual void operator+=(const vector3 &v)=0; 
	virtual void operator*=(const affine3 &a)=0;
	virtual void operator*=(const quad_trans &q)=0;
	void operator-=(const vector3 &v) { operator+=(-v); }
	void operator*=(const quad_frame &f) { operator*=(f.f); }
	void operator*=(const frame3 &f) { operator*=(f.f); }
	void operator*=(double s) {operator*=(affine3().scale(vector3(s, s)));}
	void operator/=(double s) { operator*=(1/s); }
};

class polygon : public shape
{
public:
	polygon() {}
	~polygon() {}

	polygon(const rect &r);
	shape *copy() const { return new polygon(*this); }

	double perimeter() const;
	double area() const; // as projected onto x-y plane
	vector3 centroid() const;
	vector3 upperleft() const;  // returns the point of the bounding box closest to 0,0
	vector3 lowerright() const;  // returns the point of the bounding box farthest to 0,0
	rect bound() const;
	void scan(const rect &b, spanproc *s, void *d=0) const;
	void operator+=(const vector3 &v); 
	void operator*=(const affine3 &v);
	void operator*=(const frame3 &f) { operator*=(f.f); }
	void operator*=(const quad_trans &q);
	void operator*=(const quad_frame &q) {operator*=(q.f);}
	::xlist<vector3> &points() { return p; }
	const ::xlist<vector3> &points() const { return p; }

private:
	::xlist<vector3> p;
};

class ellipse : public shape
{
public:
	ellipse(vector3 C, vector3 R) : c(C), r(R) {};
	ellipse(vector3 C, double R) : c(C), r(vector3(R, R)) {};
	~ellipse() {}
	shape *copy() const { return new ellipse(*this); }

	double perimeter() const;
	double area() const; // as projected onto x-y plane
	vector3 centroid() const;
	rect bound() const;
	void scan(const rect &b, spanproc *s, void *d=0) const;
	void operator+=(const vector3 &v); 
	void operator*=(const affine3 &v);
	void operator*=(const frame3 &f) { operator*=(f.f); }
	void operator*=(const quad_trans &q);
	void operator*=(const quad_frame &q) {operator*=(q.f);}
	vector3 radius() const { return r; }

private:
	vector3 c, r;
};

class donut: public shape
{
public:
	donut(vector3 C, vector3 innerr, vector3 outerr) : inner(C, innerr), outer(C, outerr){};
	donut(vector3 C, double innerr, double outerr) : inner(C, innerr), outer(C, outerr){};
	~donut(){}
	shape *copy() const {return new donut(*this);}
	double perimeter() const {return outer.perimeter()+inner.perimeter();}
	double area() const {return outer.area()-inner.area();}
	vector3 centroid() const {return outer.centroid();}
	rect bound() const{return outer.bound();}
	void scan(const rect &b, spanproc *s, void *d=0) const;
	
	void operator+=(const vector3 &v); 
	void operator*=(const affine3 &v);
	void operator*=(const frame3 &f) { operator*=(f.f); }
	void operator*=(const quad_trans &q){inner*=q; outer*=q;}
	void operator*=(const quad_frame &q) {operator*=(q.f);}

	vector3 radius() const { return outer.radius(); }

private:
	ellipse inner;
	ellipse outer;
};
#endif
