#pragma once
#include "STL.h"
#include "morph.h"
#include "rtypes.h"
#include "dot2d.h"
#include "lsqrpoly.h"

/*
	Mapping between the two coordinate frames (image pixel and CAD meters)
	is done using Eric's orthogonal basis functions (through 3rd order)
	this allows inclusion of camera distortion along with the normal projective 
	terms

	m[2][4][4] maps from image pixel space to CAD space
	m[0] maps in the image column direction ~= CAD Y direction
	m[1] maps in the image row direction ~= CAD X direction

	mInverse[2][4][4] maps from CAD to image
	
	Note that the basis functions are only defined over a finite range (say pixels in an image),
	to operate over a the real plane (the CAD space) we need to know the defined range
	(the min/max corners of the region on the board), this finite range is then mapped to units of pseudo pixels
	for use by htcorp().

	Pupil height is also taken into account as a senstivity of m and mInverse to z position,
	dmdz, dmdzInverse

	units:
	m			units of meters
	mInverse	units of pixels
	dmdz		meters XY / meter Z
	dmdzInverse	pixels / meter Z
	

*/
struct POINTPIX
{
	POINTPIX()
	{
		u = 0;
		v = 0;
	}
	POINTPIX(double du, double dv)
	{
		u = du;
		v = dv;
	}

   double u, v;
};    


// S and dSdZ for (col, row)<->(y,x) in world space
class TransformCamModel
{
public:

	/* constructors */	
	TransformCamModel();
	TransformCamModel(const TransformCamModel& orig);
			
	void Reset();

	POINT2D SPix2XY(POINTPIX uv);
	void SPix2XY(double u, double v, double* px, double* py);
	POINT2D dSPix2XY(POINTPIX uv);
	void dSPix2XY(double u, double v, double* px, double* py);
	
	float			S[2][MORPH_BASES][MORPH_BASES];
	float			dSdz[2][MORPH_BASES][MORPH_BASES];

	float			SInverse[2][MORPH_BASES][MORPH_BASES];
	float			dSdzInverse[2][MORPH_BASES][MORPH_BASES];
	double			uMin, uMax, vMin, vMax;
	double			xMin, xMax, yMin, yMax;
	void operator=(const TransformCamModel& b);


	void	CalculateInverse(); // THIS FUNCTION ISN'T NEEEDED
	void	CalcTransform(POINTPIX* uv, POINT2D* xy, unsigned int npts);

};


