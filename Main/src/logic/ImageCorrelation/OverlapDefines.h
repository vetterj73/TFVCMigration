#pragma once

#include "MosaicImage.h"
#include "UIRect.h"

// Contains the correlation result
class CorrelationResult
{
public:

	CorrelationResult();

	CorrelationResult(
		double		dRowOffset,
		double		dColOffset,
		double		dCorrCoeff,
		double		dAmbigScore);

	CorrelationResult(const CorrelationResult& b);

	void operator=(const CorrelationResult& b);

	// CorrelationResult properties
	double		RowOffset;
	double		ColOffset;
	double		CorrCoeff;
	double		AmbigScore;
};

typedef enum { 
	Row_To_Row			= 10,
	Col_To_Col			= 20,
	CAD					= 30,
	FID					= 35,
	NULL_OVERLAP		= 40
} OverlapType;

// The image and ROI pair for correlation (contains result)
class CorrelationPair
{
public:
	CorrelationPair();

	CorrelationPair(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		pair<unsigned int, unsigned int> topLeftCorner2,
		OverlapType type,
		Image* pMaskImage = NULL);

	CorrelationPair(const CorrelationPair& b);

	void operator=(const CorrelationPair& b);

	Image* GetFirstImg()  {return _pImg1;};
	Image* GetSecondIMg() {return _pImg2;};
	UIRect GetFirstRoi()  {return _roi1;};
	UIRect GetSecondRoi() {return _roi2;};

	Image* GetMaskImg() {return _pMaskImage;};

	bool GetCorrelationResult(CorrelationResult* pResult);

	OverlapType GetOverlapType() {return _type;};

private:
	Image* _pImg1;
	Image* _pImg2;
	Image* _pMaskImage;

	UIRect _roi1;
	UIRect _roi2;

	OverlapType _type;

	bool _bIsProcessed;

	CorrelationResult _result;
};

// Base class for overlap between image and image
class Overlap		
{
public:
	Overlap();

	Overlap(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type);

	Overlap(const Overlap& b);
	void operator=(const Overlap& b);

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

	bool _bValid;

	CorrelationPair _coarsePair;
	list<CorrelationPair> _finePairList;
};

// Overlap between FOV image FOV image
class FovFovOverlap: public Overlap
{
	MosaicImage*	pMosaic1;
	MosaicImage*	pMosaic2;
	pair<unsigned int, unsigned int> ImgPos1;
	pair<unsigned int, unsigned int> ImgPos2;
};

class FovCadOverlap: public Overlap
{
	MosaicImage*	pMosaic;
	pair<unsigned int, unsigned int> ImgPos;

	Image* pCadImg;
};


class FovFidOverlap: public Overlap
{
	MosaicImage*	pMosaic;
	pair<unsigned int, unsigned int> ImgPos;

	Image* pFidImg;

	double _dXcenter;
	double _dYcenter;
};

