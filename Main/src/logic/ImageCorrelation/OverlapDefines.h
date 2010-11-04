/*
	#define the classes for overlap between Fov and Fov, Cad and Fov, and Fiducial and Fov
*/

#pragma once

#include "MosaicImage.h"
#include "CorrelationPair.h"

// Base class for overlap between image and image
class Overlap		
{
public:
	Overlap();

	Overlap(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type,		
		Image* pMaskImg = NULL);

	Overlap(const Overlap& b);
	void operator=(const Overlap& b);

	void config(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type,		
		Image* pMaskImg = NULL);

	Image* GetFirstImage() {return _pImg1;};

	unsigned int Columns() {return _iColumns;};
	unsigned int Rows() {return _iRows;};

	bool IsValid() {return _bValid;};

	bool DoIt();

protected:
	bool CalCoarseCorrPair();
	bool ChopOverlap();

private:
	Image* _pImg1;
	Image* _pImg2;
	DRect _validRect;
	OverlapType _type;

	Image* _pMaskImg;

	bool _bValid;

	unsigned int _iColumns;
	unsigned int _iRows;

	CorrelationPair _coarsePair;
	list<CorrelationPair> _finePairList;
};

// Overlap between FOV image FOV image
class FovFovOverlap: public Overlap
{
public:
	FovFovOverlap(
		MosaicImage*	pMosaic1,
		MosaicImage*	pMosaic2,
		pair<unsigned int, unsigned int> ImgPos1,
		pair<unsigned int, unsigned int> ImgPos2,
		DRect validRect,
		bool bHasMask);

	bool IsValid();

private:
		MosaicImage*	_pMosaic1;
		MosaicImage*	_pMosaic2;
		pair<unsigned int, unsigned int> _ImgPos1;
		pair<unsigned int, unsigned int> _ImgPos2;
		bool _bHasMask;
};

class CadFovOverlap: public Overlap
{
public:
	CadFovOverlap(
		MosaicImage* pMosaic,
		pair<unsigned int, unsigned int> ImgPos,
		Image* pCadImg,
		DRect validRect);

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _ImgPos;
	Image* _pCadImg;
};


class FidFovOverlap: public Overlap
{
public:
	FidFovOverlap(
		MosaicImage*	pMosaic,
		pair<unsigned int, unsigned int> ImgPos,
		Image* pFidImg,
		double _dXcenter,
		double _dYcenter,
		DRect validRect);

	double GetFiducialXPos() {return _dCenterX;};
	double GetFiducialYPos() {return _dCenterY;};

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _ImgPos;

	Image* _pFidImg;

	double _dCenterX;
	double _dCenterY;
};

