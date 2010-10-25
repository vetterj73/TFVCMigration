#pragma once

/* 
	3*3 projective transform for 2D image
*/


class ImgTransform
{
public:

	// Constructors
	ImgTransform(void);

	ImgTransform(const double dT[9]);
	ImgTransform(const double dT[3][3]);

	ImgTransform(const ImgTransform& b);

	void operator=(const ImgTransform& b);

	// Get/set functions
	void GetMatrix(double dT[9]) const;
	void GetMatrix(double dT[3][3]) const;
	void SetMatrix(const double dT[9]);
	void SetMatrix(const double dT[3][3]);

	void GetInvertMatrix(double dInvT[9]);
	void GetInvertMatrix(double dInvT[3][3]);

private: 
	bool CalInverse();

	bool _bHasInverse;	// if the inverse transform Maxtric is valid

	double _dT[9];
	double _dInvT[9];
};

ImgTransform operator*(const ImgTransform& left, const ImgTransform& right);

