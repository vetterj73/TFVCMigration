//
// File: xform.cpp
// Description: implements neat classes for coordinate transforms
// Created: June 24, 1999 
// Author: Adam Reinhardt 
// 
// Copyright (c) 1999,2000,2001,2002 CyberOptics Corp. 
//
// $Author: Adam $ 
// $Revision: 6 $
// $Header: /Products/SMT Systems/Ghidorah/Source/QNX/Realtime/Development/xform.cpp 6     3/04/02 10:49a Adam $

#include "xform.h"

// left and right lookups for det and rotate
static int l[3]={1,0,0}, r[3]={2,2,1};

matrix3 &matrix3::operator*=(double s)
{
	for(int i=0;i<9;i++) a[0][i] *= s;
	
	return *this;
}

matrix3 matrix3::operator*(const matrix3 &n) const
{
	matrix3 p(0);
	const matrix3 &m=*this;
	
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				p(i,j)+=m(i,k)*n(k,j);

	return p;
}

vector3 matrix3::operator*(const vector3 &x) const
{
	vector3 y;
	const matrix3 &a=*this;
	
	y.x = a(0,0)*x.x + a(0,1)*x.y + a(0,2)*x.z;
	y.y = a(1,0)*x.x + a(1,1)*x.y + a(1,2)*x.z;
	y.z = a(2,0)*x.x + a(2,1)*x.y + a(2,2)*x.z;

	return y;
}

double matrix3::det(int i, int j) const
{
	const matrix3 &m=*this;

	return m(l[i],l[j])*m(r[i],r[j]) - m(l[i],r[j])*m(r[i],l[j]);
}

double matrix3::det() const
{
	const matrix3 &m=*this;

	return m(0,0)*det(0,0) - m(0,1)*det(0,1) + m(0,2)*det(0,2);
}

static double mult[3][3]=
{
	{1,-1,1},
	{-1,1,-1},
	{1,-1,1},
};

matrix3 matrix3::invert() const
{
	double f=1/det();
	matrix3 m;
	
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			m(j,i)=det(i,j)*f*mult[i][j];

	return m;
}

void swap(double &o, double &t)
{
	double temp=o;
	o=t;
	t=temp;
}

matrix3 matrix3::transpose() const
{
	matrix3 m((*this));

	for(int i=0;i<3;i++)
		for(int j=i+1;j<3;j++)
			swap(m(i,j), m(j,i));
	
	return m;
}

matrix3 matrix3::rotate(int i, double a) const // axis, angle
{
	matrix3 m;
	double c=cos(a),s=sin(a);

	m(i,i)=1;
	m(l[i],l[i])=c;
	m(r[i],r[i])=c;
	m(l[i],r[i])=-s;
	m(r[i],l[i])=s;

	return m(*this);
}

matrix3 matrix3::scale(const vector3 &v) const
{
	matrix3 m;
	
	m(0,0)=v.x;
	m(1,1)=v.y;
	m(2,2)=v.z;
	
	return m(*this);
}
