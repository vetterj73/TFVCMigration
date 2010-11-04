#include "ImgTransform.h"
#include "Utilities.h"
#include "math.h"

// Constructors
ImgTransform::ImgTransform(void)
{
	_bHasInverse = false;

	_dT[0] = 1;
	_dT[1] = 0;
	_dT[2] = 0;

	_dT[3] = 0;
	_dT[4] = 1;
	_dT[5] = 0;

	_dT[6] = 0;
	_dT[7] = 0;
	_dT[8] = 1;
}


ImgTransform::ImgTransform(const double dT[9])
{
	SetMatrix(dT);
}

ImgTransform::ImgTransform(const double dT[3][3])
{
	SetMatrix(dT);
}

ImgTransform::ImgTransform( 
	double dScaleX, 
	double dScaleY,
	double dRotation, 
	double dTranslateX,
	double dTranslateY)
{
	_bHasInverse = false;

	double cosTheta = cos(dRotation);
	double sinTheta = sin(dRotation);

	_dT[0] = dScaleX * cosTheta;
	// Need double check
	_dT[1] = -(dScaleX+dScaleY)/2 *sinTheta;
	_dT[2] = dTranslateX;
	
	_dT[3] = -_dT[1];
	_dT[4] = dScaleY * cosTheta;
	_dT[5] = dTranslateY;

	_dT[6] = 0;
	_dT[7] = 0;
	_dT[8] = 1;
}


ImgTransform::ImgTransform(const ImgTransform& b)
{
	*this = b;
}

void ImgTransform::operator=(const ImgTransform& b)
{
	unsigned int i;
	for(i=0; i<9; i++)
		_dT[i] = b._dT[i];

	_bHasInverse = b._bHasInverse;

	if(_bHasInverse)
		for(i=0; i<9; i++)
			_dInvT[i] = b._dInvT[i];
}

// Get/set functions
void ImgTransform::GetMatrix(double dT[9]) const 
{
	unsigned int i;
	for(i=0; i<9; i++)
		dT[i] = _dT[i];
}

void ImgTransform::GetMatrix(double dT[3][3]) const 
{
	unsigned int ix, iy;
	for(iy=0; iy<3; iy++)
		for(ix=0; ix<3; ix++)
			dT[iy][ix] = _dT[3*iy+ix];
}

void ImgTransform::SetMatrix(const double dT[9])
{
	_bHasInverse = false;

	unsigned int i;
	for(i=0; i<9; i++)
		_dT[i] = dT[i];
}

void ImgTransform::SetMatrix(const double dT[3][3])
{
	_bHasInverse = false;

	unsigned int ix, iy;
	for(iy=0; iy<3; iy++)
		for(ix=0; ix<3; ix++)
			_dT[3*iy+ix] = dT[iy][ix];
}

void ImgTransform::GetInvertMatrix(double dInvT[9])
{
	if(!_bHasInverse)
		CalInverse();

	unsigned int i;
	for(i=0; i<9; i++)
		dInvT[i] = _dInvT[i];
}

void ImgTransform::GetInvertMatrix(double dInvT[3][3])
{
	if(!_bHasInverse)
		CalInverse();

	unsigned int ix, iy;
	for(iy=0; iy<3; iy++)
		for(ix=0; ix<3; ix++)
			dInvT[iy][ix] = _dInvT[3*iy+ix];
}

ImgTransform ImgTransform::Inverse()
{
	double dInvT[9];
	GetInvertMatrix(dInvT);
	ImgTransform t(dInvT);

	return(t);
}

// Map and inverse map
void ImgTransform::Map(double dx, double dy, double* pdu, double* pdv) const
{
	*pdu = dx*_dT[0] + dy*_dT[1] + _dT[2];
	*pdv = dx*_dT[3] + dy*_dT[4] + _dT[5];
	double dTemp = dx*_dT[6] + dy*_dT[7] + _dT[8];

	*pdu = *pdu/dTemp;
	*pdv = *pdv/dTemp;
}

void ImgTransform::InverseMap(double du, double dv, double* pdx, double* pdy) const
{
	*pdx = du*_dInvT[0] + dv*_dInvT[1] + _dInvT[2];
	*pdy = du*_dInvT[3] + dv*_dInvT[4] + _dInvT[5];
	double dTemp = du*_dInvT[6] + dv*_dInvT[7] + _dInvT[8];

	*pdx = *pdx/dTemp;
	*pdy = *pdx/dTemp;
}

void ImgTransform::CalInverse()
{
	inverse(_dT, _dInvT, 3, 3);

	_bHasInverse = true;
}

ImgTransform operator*(const ImgTransform& left, const ImgTransform& right)
{
	ImgTransform t;
	double out[9];
	double a[9], b[9];
	left.GetMatrix(a);
	right.GetMatrix(b);

	out[0] = a[0]*b[0]+a[1]*b[3]+a[2]*b[6];
	out[1] = a[0]*b[1]+a[1]*b[4]+a[2]*b[7];
	out[2] = a[0]*b[2]+a[1]*b[5]+a[2]*b[8];
											 
	out[3] = a[3]*b[0]+a[4]*b[3]+a[5]*b[6];
	out[4] = a[3]*b[1]+a[4]*b[4]+a[5]*b[7];
	out[5] = a[3]*b[2]+a[4]*b[5]+a[5]*b[8];
											 
	out[6] = a[6]*b[0]+a[7]*b[3]+a[8]*b[6];
	out[7] = a[6]*b[1]+a[7]*b[4]+a[8]*b[7];
	out[8] = a[6]*b[2]+a[7]*b[5]+a[8]*b[8];

	if(out[8]<0.01 && out[8]>-0.01)
	{
		out[8] = 0.01;
	}

	for(unsigned int i=0; i<9; i++)
		out[i] = out[i]/out[8];

	t.SetMatrix(out);

	return t;
}
