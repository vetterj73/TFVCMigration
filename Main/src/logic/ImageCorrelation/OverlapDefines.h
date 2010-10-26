#pragma once

#include "MosaicImage.h"
#include "UIRect.h"

class CorrelationResult
{
public:

	CorrelationResult();

	void operator=(const CorrelationResult& b)
	{
		RowOffset	= b.RowOffset;
		ColOffset	= b.ColOffset;
		CorrCoeff	= b.CorrCoeff;
		AmbigScore	= b.AmbigScore;
	}

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

class CorrelationPair
{
public:
	CorrelationPair(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		UIRect roi2,
		OverlapType type);

	Image* GetFirstImg()  {return _pImg1;};
	Image* GetSecondIMg() {return _pImg2;};
	UIRect GetFirstRoi()  {return _roi1;};
	UIRect GetSecondRoi() {return _roi2;};

	CorrelationResult GetCorrelationResult();

private:
	Image* _pImg1;
	Image* _pImg2;

	UIRect _roi1;
	UIRect _roi2;

	OverlapType _Type;

	bool		IsProcessed;

	CorrelationResult _result;
};

class Overlap		
{
public:
	Overlap(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		UIRect roi2,
		OverlapType type);

private:
	CorrelationPair coarsePair;
	CorrelationPair* finePairs;
};

class ImgImgOverlap: public Overlap
{
	MosaicImage*	pMosaic1;
	MosaicImage*	pMosaic2;
	pair<unsigned int, unsigned int> ImgPos1;
	pair<unsigned int, unsigned int> ImgPos2;
};

class ImgCadOverlap: public Overlap
{
	MosaicImage*	pMosaic;
	pair<unsigned int, unsigned int> ImgPos;

	Image* pCadImg;
};


class ImgFidOverlap: public Overlap
{
	MosaicImage*	pMosaic;
	pair<unsigned int, unsigned int> ImgPos;

	Image* pFidImg;

	double _dXcenter;
	double _dYcenter;
};

