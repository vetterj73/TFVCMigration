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
// $Revision: 12 $
// $Header: /Products/SMT Systems/Ghidorah/Source/QNX/Realtime/Development/hell.cpp 12    4/02/02 3:00p Adam $

#include "xform.h"
#include "hell.h"
#include <stdio.h>
#include <float.h>
#ifdef WIN32
#include <stdio.h>
#include <stdarg.h>
#define vsnprintf _vsnprintf
#else
#include <unix.h> // for vsnprintf
#endif


quadratic::quadratic(const matrix4 &m)
{
	x2=m(0,0);
	xy=m(0,1)+m(1,0);
	xz=m(0,2)+m(2,0);
	x=m(0,3)+m(3,0);

	y2=m(1,1);
	yz=m(1,2)+m(2,1);
	y=m(1,3)+m(3,1);

	z2=m(2,2);
	z=m(2,3)+m(3,2);

	c=m(3,3);
}

matrix4 matrix4::operator+(const matrix4 &n) const
{
	matrix4 p(0);
	const matrix4 &m=*this;
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			p(i,j)=m(i,j)+n(i,j);
	return p;
}

matrix4 matrix4::operator-(const matrix4 &n) const
{
	matrix4 p(0);
	const matrix4 &m=*this;
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			p(i,j)=m(i,j)-n(i,j);
	return p;
}

matrix4 &matrix4::operator*=(double s)
{
	for(int i=0;i<16;i++) a[0][i] *= s;
	
	return *this;
}

matrix4 matrix4::operator*(const matrix4 &n) const
{
	matrix4 p(0);
	const matrix4 &m=*this;
	
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			for(int k=0;k<4;k++)
				p(i,j)+=m(i,k)*n(k,j);

	return p;
}

static void swap(double &o, double &t)
{
	double temp=o;
	o=t;
	t=temp;
}

matrix4 matrix4::transpose() const
{
	matrix4 m((*this));

	for(int i=0;i<4;i++)
		for(int j=i+1;j<4;j++)
			swap(m(i,j), m(j,i));
	
	return m;
}

#include <stdio.h>
#include <stdarg.h>

#include "xform.h"
#include "hell.h"

#ifdef WIN32
void host_debug(int flag, const char *fmt, ...)
{
	if(flag)
	{
		char str[1024];
		va_list args;

		va_start(args, fmt);
		vsnprintf(str,1024,fmt,args);
		printf("%s\n",str);
		va_end(args);
	}
}

void host_debug(const char *fmt, ...)
{
	va_list args;
	char str[1024];

	va_start(args, fmt);

	vsnprintf(str,1024,fmt,args);
	printf("%s\n",str);

	va_end(args);
}

#else

#include "os.h"

#endif

quad_trans quad_trans::invert() const
{
	matrix3 m;
	vector3 o;

	m(0,0)=Cx.x;
	m(0,1)=Cx.y;
	m(0,2)=Cx.z;
	o.x=Cx.c;
	
	m(1,0)=Cy.x;
	m(1,1)=Cy.y;
	m(1,2)=Cy.z;
	o.y=Cy.c;
	
	m(2,0)=Cz.x;
	m(2,1)=Cz.y;
	m(2,2)=Cz.z;
	o.z=Cz.c;

	quad_trans ainv(affine3(m,o).invert());

	quad_trans q=quad_trans().scale(vector3(2,2,2))-(*this)(ainv);

	return ainv(q);
}

affine3 quad_trans::linear_frame() const
{
	affine3 a;

	a.A(0,0)=Cx.x;
	a.A(0,1)=Cx.y;
	a.A(0,2)=Cx.z;
	a.b.x=Cx.c;

	a.A(1,0)=Cy.x;
	a.A(1,1)=Cy.y;
	a.A(1,2)=Cy.z;
	a.b.y=Cy.c;

	a.A(2,0)=Cz.x;
	a.A(2,1)=Cz.y;
	a.A(2,2)=Cz.z;
	a.b.z=Cz.c;

	return a;
}

quad_trans quad_trans::operator()(const quad_trans &q) const
{
	if(quad() && q.quad())
	{
#ifdef DEBUG
		host_debug(1, "here i am");
		int a=0;
		while(!a)
			host_delay(1000);
		exit(-1);
#else
		return quad_trans();
#endif
	}

	quad_trans nq;

	if(q.quad())
	{
		nq.Cx=(q.Cx*Cx.x)+(q.Cy*Cx.y)+(q.Cz*Cx.z);
		nq.Cy=(q.Cx*Cy.x)+(q.Cy*Cy.y)+(q.Cz*Cy.z);
		nq.Cz=(q.Cx*Cz.x)+(q.Cy*Cz.y)+(q.Cz*Cz.z);
		nq.Cx.c+=Cx.c;
		nq.Cy.c+=Cy.c;
		nq.Cz.c+=Cz.c;
	}
	else
	{
		matrix4 a(0);

		a(0,0)=q.Cx.x;
		a(0,1)=q.Cx.y;
		a(0,2)=q.Cx.z;
		a(0,3)=q.Cx.c;

		a(1,0)=q.Cy.x;
		a(1,1)=q.Cy.y;
		a(1,2)=q.Cy.z;
		a(1,3)=q.Cy.c;

		a(2,0)=q.Cz.x;
		a(2,1)=q.Cz.y;
		a(2,2)=q.Cz.z;
		a(2,3)=q.Cz.c;

		a(3,3)=1;

		nq.Cx=a.transpose()*Cx*a;
		nq.Cy=a.transpose()*Cy*a;
		nq.Cz=a.transpose()*Cz*a;
	}

	return nq;
}

quad_trans quad_trans::operator()(const affine3 &af) const
{
	quad_trans q;
	matrix4 a=af;

	q.Cx=a.transpose()*Cx*a;
	q.Cy=a.transpose()*Cy*a;
	q.Cz=a.transpose()*Cz*a;

	return q;
}

quad_frame quad_frame::operator>>(const quad_frame &dest) const
	{ return quad_frame(dest.r(f), r(dest.f)); }

static char * add_float(char *str,const char *name, double d)
{
	if(fabs(d)<DBL_EPSILON) return str;

	str+=sprintf(str, "%s %g ", name, d);
	return str;
}

quad_frame quad_frame::invert() const { return quad_frame(r, f); }
quad_frame quad_frame::operator()(const quad_frame &base) const
	{ return quad_frame(f(base.f), base.r(r)); }

frame3 quad_frame::linear_frame() const {return frame3(f.linear_frame(), r.linear_frame()); }

void print_quadratic(int verbose, const char *str, const quadratic &q)
{
	host_debug(verbose, str);
	char string[1000];
	char *s=string;
	
	s=add_float(s, "x2", q.x2);
	s=add_float(s, "+ x", q.x);
	s=add_float(s, "+ xy", q.xy);
	s=add_float(s, "+ xz", q.xz);

	s=add_float(s, "+ y2", q.y2);
	s=add_float(s, "+ y", q.y);
	s=add_float(s, "+ yz", q.yz);

	s=add_float(s, "+ z2", q.z2);
	s=add_float(s, "+ z", q.z);

	s=add_float(s, "+", q.c);

	host_debug(verbose, "\t\t\t%s", string);
}

void print_quad_trans(int verbose, const char *str, const quad_trans &q)
{
	host_debug(verbose, "\t%s", str);
	print_quadratic(verbose, "\t\tx:", q.Cx);
	print_quadratic(verbose, "\t\ty:", q.Cy);
	print_quadratic(verbose, "\t\tz:", q.Cz);
}

void print_quad_frame(int verbose, const char *str, const quad_frame &qf)
{
	host_debug(verbose, str);
	print_quad_trans(verbose, "fwd:", qf.f);
	print_quad_trans(verbose, "rev:", qf.r);
}

#ifdef TEST_HELL

#include <stdlib.h>

double rand(double max_val){return rand()*max_val/RAND_MAX;}

void host_delay(int){}
int host_kill(unsigned int){return 0;}
unsigned int host_findname(const char *){return 0;}
char const * write_image(void const *,int,int,int,int,char const *,int,double,double,double,double,char const *){return 0;}

#ifndef PI
   #define PI        (3.141592653589793)
#endif
#ifndef TWOPI
   #define TWOPI     (2.*PI)
#endif

#include "list.h"

void test_quad(const list<quad_trans> &l, list<vector3> &points)
{
	host_debug("\n\n##############test quad ########################\n\n");
	quad_trans s;
	quad_trans sinv;
	quad_trans long_sinv;
	list<vector3> prime_points=points;

	for(listiter<quad_trans> q(l);q.on();q++)
	{
		s=(*q)(s);
		
		long_sinv=long_sinv(q->invert());

		for(listiter<vector3> prime_point(prime_points);prime_point.on();prime_point++)
			*prime_point=(*q)(*prime_point);
	}

	sinv=s.invert();

	print_quad_trans(1, "s", s);
	print_quad_trans(1, "sinv", sinv);
	print_quad_trans(1, "long_sinv", long_sinv);

	listiter<vector3> point(points);
	for(listiter<vector3> prime_point(prime_points);prime_point.on();prime_point++, point++)
	{
		vector3 pp=s(*point);
		vector3 back_to_p=sinv(*prime_point);

		host_debug("point,%g,%g,%g,back_to_p,%g,%g,%g",
			point->x,point->y,point->z,back_to_p.x,back_to_p.y,back_to_p.z);
		host_debug("pp,%g,%g,%g,prime_point,%g,%g,%g",
			pp.x,pp.y,pp.z,prime_point->x,prime_point->y,prime_point->z);
	}
}

int main(int argc,char *argv[])
{
	matrix3 m;
	m=m.invert();
	quad_trans q;
	q.Cy.xy=.00001/.16;//=.000000002;

	list<vector3> points;

	for(int i=0;i<100;i++)
		points.push(vector3(rand(1), rand(1), rand(1)));

	quad_trans f=quad_trans().offset(vector3(2,2,0));
	quad_trans s=quad_trans().scale(vector3(4,5,1));

	quad_trans m2e=quad_trans().scale(vector3(1/0.0000005,1/0.0000005,1/0.000000234375));

	list<quad_trans> l;

	quad_trans r=quad_trans().rotate(2,.25*TWOPI);

	l.push(f);
	l.push(s);
	l.push(r);
	//test_quad(l, points);

	l= list<quad_trans>();
	l.push(q);
	l.push(m2e);

	test_quad(l, points);

	l= list<quad_trans>();
	l.push(f);
	l.push(s);
	l.push(q);

	//test_quad(l, points);

	list<quad_trans> l2;

	l2.push(quad_trans().rotate(2,.5*TWOPI));
	l2.push(quad_trans().rotate(2,-.5*TWOPI));
	l2.push(f);
	l2.push(s);
	l2.push(q);
	l2.push(f);
	l2.push(quad_trans().rotate(2,.5));
	l2.push(quad_trans().rotate(1,.5));

	//test_quad(l2, points);

	host_debug("sizeof quad_trans %d", sizeof(quad_trans));
	host_debug("sizeof affine3 %d", sizeof(affine3));

	return 0;
}

#endif