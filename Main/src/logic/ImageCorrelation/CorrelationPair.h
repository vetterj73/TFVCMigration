/*
	#define the classes for Alignment/correlation pair and results
*/

#pragma once

#include "Image.h"
#include "UIRect.h"
#include "Bitmap.h"

// Contains the correlation result
class CorrelationResult
{
public:

	//Constructors 
	CorrelationResult();

	CorrelationResult(
		double		dRowOffset,
		double		dColOffset,
		double		dCorrCoeff,
		double		dAmbigScore);

	CorrelationResult(const CorrelationResult& b);

	void operator=(const CorrelationResult& b);

	// Default values
	void Default();

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
	// Constructors
	CorrelationPair();

	CorrelationPair(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		pair<unsigned int, unsigned int> topLeftCorner2,  // (column row)
		unsigned int iDecim,
		unsigned int iColSearchExpansion,
		unsigned int iRowSearchExpansion,
		OverlapType type,
		Image* pMaskImage = NULL);

	CorrelationPair(const CorrelationPair& b);

	void operator=(const CorrelationPair& b);

	// Get/set functions
	Image* GetFirstImg() const {return _pImg1;};
	Image* GetSecondImg() const {return _pImg2;};
	UIRect GetFirstRoi() const {return _roi1;};
	UIRect GetSecondRoi() const {return _roi2;};

	Image* GetMaskImg() const {return _pMaskImg;};

	void SetCorrlelationResult(CorrelationResult result);

	bool GetCorrelationResult(CorrelationResult* pResult) const;
	CorrelationResult GetCorrelationResult() const;

	OverlapType GetOverlapType() const {return _type;};

	unsigned int Columns() const {return _roi1.Columns();};
	unsigned int Rows() const {return _roi1.Rows();};

	bool IsValid() const {return _roi1.IsValid();};

	bool IsProcessed() const {return _bIsProcessed;};

	// Do alignment and reset
	bool DoAlignment(bool bApplyCorrSizeUpLimit=false, bool* pbCorrSizeReduced=NULL);	
	bool Reset();

	// For overlap process
	bool AdjustRoiBaseOnResult(CorrelationPair* pPair) const;
	
	bool ChopCorrPair(
		unsigned int iNumBlockX, 
		unsigned int iNumBlockY, 
		unsigned int iBlockWidth, 
		unsigned int iBlockHeight,
		unsigned int iBlockDecim,
		unsigned int iBlockColSearchExpansion,
		unsigned int iBlockRowSearchExpansion,
		list<CorrelationPair>* pOutPairList) const;

	void NorminalCenterInWorld(double* fdx, double* fdy);

	// For Debug
	void DumpImg(string sFileName) const;
	bool DumpImgWithResult(string sFileName) const;

protected:
	bool SqRtCorrelation(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced);
	bool NGCCorrelation(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced);
	int MaskedNgc(UIRect tempRoi, UIRect searchRoi);

private:
	Image* _pImg1;
	Image* _pImg2;
	Image* _pMaskImg;

	UIRect _roi1;
	UIRect _roi2;

	OverlapType _type;

	bool _bIsProcessed;
	bool _bUsedNgc;	// Ngc is used for calcaulation
	unsigned int _iDecim;	// For regoff
	unsigned int _iColSearchExpansion; // For NGC
	unsigned int _iRowSearchExpansion;

	CorrelationResult _result;
};