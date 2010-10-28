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
	Fov_To_Fov			= 10,
	Cad_To_Fov			= 20,
	Fid_To_Fov			= 30,
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

