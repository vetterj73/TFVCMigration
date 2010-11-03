#pragma once

#include "Image.h"
#include "UIRect.h"
#include "Bitmap.h"

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
// Mask image must be with the first image now
class CorrelationPair
{
public:
	CorrelationPair();

	CorrelationPair(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		pair<unsigned int, unsigned int> topLeftCorner2,
		unsigned int iDecim,
		unsigned int iColSearchExpansion,
		unsigned int iRowSearchExpansion,
		OverlapType type,
		Image* pMaskImage = NULL);

	CorrelationPair(const CorrelationPair& b);

	void operator=(const CorrelationPair& b);

	Image* GetFirstImg()  {return _pImg1;};
	Image* GetSecondImg() {return _pImg2;};
	UIRect GetFirstRoi()  {return _roi1;};
	UIRect GetSecondRoi() {return _roi2;};

	Image* GetMaskImg() {return _pMaskImg;};

	void SetCorrlelationResult(CorrelationResult result);

	bool GetCorrelationResult(CorrelationResult* pResult);

	OverlapType GetOverlapType() {return _type;};

	unsigned int Columns() {return _roi1.Columns();};
	unsigned int Rows() {return _roi1.Rows();};

	bool IsProcessed() {return _bIsProcessed;};

	bool AdjustRoiBaseOnResult(CorrelationPair* pPair) const;

	bool DoAlignment();

	bool ChopCorrPair(
		unsigned int iNumBlockX, 
		unsigned int iNumBlockY, 
		unsigned int iBlockWidth, 
		unsigned int iBlockHeight,
		unsigned int iBlockDecim,
		unsigned int iBlockColSearchExpansion,
		unsigned int iBlockRowSearchExpansion,
		list<CorrelationPair>* pOutPairList) const;

	void DumpImg(string sFileName) const;
	bool DumpImgWithResult(string sFileName) const;

private:
	Image* _pImg1;
	Image* _pImg2;
	Image* _pMaskImg;

	UIRect _roi1;
	UIRect _roi2;

	OverlapType _type;

	bool _bIsProcessed;
	unsigned int _iDecim;	// For regoff
	unsigned int _iColSearchExpansion; // For NGC
	unsigned int _iRowSearchExpansion;

	CorrelationResult _result;

	unsigned int _iMinSize;	// internal setting
};