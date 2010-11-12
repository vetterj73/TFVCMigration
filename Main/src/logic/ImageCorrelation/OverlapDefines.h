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

	// Get/set
	Image* GetFirstImage() const {return _pImg1;};
	Image* GetSecondImage() const {return _pImg2;};

	unsigned int Columns() const {return _iColumns;};
	unsigned int Rows() {return _iRows;};

	bool IsValid() const {return _bValid;};
	bool IsProcessed() const {return _bProcessed;};

	CorrelationPair* GetCoarsePairPtr() {return &_coarsePair;};
	list<CorrelationPair>* GetFinePairListPtr()  {return &_finePairList;};

	// Do alignment and reset
	bool DoIt();
	bool Reset();

protected:
	bool CalCoarseCorrPair();
	bool ChopOverlap();	
	bool _bValid;
	bool _bProcessed;	
	
	CorrelationPair _coarsePair;
	list<CorrelationPair> _finePairList;

private:
	Image* _pImg1;
	Image* _pImg2;
	DRect _validRect;
	OverlapType _type;

	Image* _pMaskImg;

	unsigned int _iColumns;
	unsigned int _iRows;
};

// Overlap between FOV image and FOV image
// Image with lower illumination index is the first image
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
	unsigned int GetFirstTriggerIndex() const {return _imgPos1.second;};
	unsigned int GetFirstCameraIndex() const {return _imgPos1.first;};
	pair<unsigned int, unsigned int> GetFirstImagePosition() const {return _imgPos1;};

	MosaicImage* GetSecondMosaicImage() const {return _pMosaic2;};
	unsigned int GetSecondTriggerIndex() const {return _imgPos2.second;};
	unsigned int GetSecondCameraIndex() const {return _imgPos2.first;};
	pair<unsigned int, unsigned int> GetSecondImagePosition() const {return _imgPos2;};

	bool IsReadyToProcess() const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicImage*	_pMosaic1;
	MosaicImage*	_pMosaic2;
	pair<unsigned int, unsigned int> _imgPos1;
	pair<unsigned int, unsigned int> _imgPos2;
	bool _bHasMask;
};

// Overlap between CAD image and FOV image
// Fov image is always the first image
class CadFovOverlap: public Overlap
{
public:
	CadFovOverlap(
		MosaicImage* pMosaic,
		pair<unsigned int, unsigned int> ImgPos, // first = camera, second = trigger
		Image* pCadImg,
		DRect validRect);

	MosaicImage* GetMosaicImage() const {return _pMosaic;};
	unsigned int GetTriggerIndex() const {return _imgPos.second;};
	unsigned int GetCameraIndex() const {return _imgPos.first;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	
	Image* GetCadImage() const {return _pCadImg;};

	bool IsReadyToProcess() const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;
	Image* _pCadImg;
};

// Overlap between Fiducial image and FOV image
// Fov image is always the first image
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
	unsigned int GetTriggerIndex() const {return _imgPos.second;};
	unsigned int GetCameraIndex() const {return _imgPos.first;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	Image* GetFidImage() const {return _pFidImg;};

	double GetFiducialXPos() const {return _dFidCenterX;};
	double GetFiducialYPos() const {return _dFidCenterY;};

	bool IsReadyToProcess() const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicImage*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;

	Image* _pFidImg;

	double _dFidCenterX;
	double _dFidCenterY;
};

