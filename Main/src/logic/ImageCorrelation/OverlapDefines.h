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

	Image* GetFirstImage() const {return _pImg1;};

	unsigned int Columns() const {return _iColumns;};
	unsigned int Rows() {return _iRows;};

	virtual bool IsValid() const {return _bValid;};
	bool IsProcessed() const {return _bProcessed;};

	bool DoIt();
	bool Reset();

protected:
	bool CalCoarseCorrPair();
	bool ChopOverlap();	
	bool _bValid;
	bool _bProcessed;

private:
	Image* _pImg1;
	Image* _pImg2;
	DRect _validRect;
	OverlapType _type;

	Image* _pMaskImg;

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
		pair<unsigned int, unsigned int> ImgPos1, // first = camera, second = trigger
		pair<unsigned int, unsigned int> ImgPos2,
		DRect validRect,
		bool bHasMask);

	MosaicImage* GetFirstMosaicImage() const {return _pMosaic1;};
	pair<unsigned int, unsigned int> GetFirstImagePosition() const {return _imgPos1;};

	MosaicImage* GetSecondMosaicImage() const {return _pMosaic2;};
	pair<unsigned int, unsigned int> GetSecondImagePosition() const {return _imgPos2;};

	bool IsValid() const;

private:
		MosaicImage*	_pMosaic1;
		MosaicImage*	_pMosaic2;
		pair<unsigned int, unsigned int> _imgPos1;
		pair<unsigned int, unsigned int> _imgPos2;
		bool _bHasMask;
};

class CadFovOverlap: public Overlap
{
public:
	CadFovOverlap(
		MosaicImage* pMosaic,
		pair<unsigned int, unsigned int> ImgPos, // first = camera, second = trigger
		Image* pCadImg,
		DRect validRect);

	MosaicImage* GetMosaicImage() const {return _pMosaic;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	Image* GetCadImage() const {return _pCadImg;};

	bool IsValid() const;

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;
	Image* _pCadImg;
};


class FidFovOverlap: public Overlap
{
public:
	FidFovOverlap(
		MosaicImage*	pMosaic,
		pair<unsigned int, unsigned int> ImgPos, // first = camera, second = trigger
		Image* pFidImg,
		double _dXcenter,
		double _dYcenter,
		DRect validRect);

	MosaicImage* GetMosaicImage() const {return _pMosaic;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	Image* GetFidImage() const {return _pFidImg;};

	double GetFiducialXPos() const {return _dCenterX;};
	double GetFiducialYPos() const {return _dCenterY;};

	bool IsValid() const;

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;

	Image* _pFidImg;

	double _dCenterX;
	double _dCenterY;
};

