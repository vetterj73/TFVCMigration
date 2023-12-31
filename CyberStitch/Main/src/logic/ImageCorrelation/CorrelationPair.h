/*
	#define the classes for Alignment/correlation pair and results
*/

#pragma once

#include "Image.h"
#include "UIRect.h"


class Overlap;

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
		unsigned int iNgcColSearchExpansion,
		unsigned int iNgcRowSearchExpansion,
		Overlap* _pOverlap,
		Image* pMaskImage = NULL);

	CorrelationPair(const CorrelationPair& b);

	void operator=(const CorrelationPair& b);

	// Get/set functions
	Image* GetFirstImg() const {return _pImg1;};
	Image* GetSecondImg() const {return _pImg2;};
	UIRect GetFirstRoi() const {return _roi1;};
	UIRect GetSecondRoi() const {return _roi2;};

	Image* GetMaskImg() const {return _pMaskImg;};
	void SetMaskImg(Image* pMaskImg) {_pMaskImg = pMaskImg;};
	void SetUseMask(bool bValue) { _bUseMask = bValue;};

	bool IsUseNgc() {return _bUsedNgc;};

	void SetCorrlelationResult(CorrelationResult result);

	bool GetCorrelationResult(CorrelationResult* pResult) const;
	CorrelationResult GetCorrelationResult() const;

	Overlap* GetOverlapPtr() {return _pOverlap;};
	void SetOverlapPtr(Overlap* pValue) { _pOverlap = pValue;};
	OverlapType GetOverlapType();

	unsigned int Columns() const {return _roi1.Columns();};
	unsigned int Rows() const {return _roi1.Rows();};

	bool IsValid() const {return _roi1.IsValid();};

	bool IsProcessed() const {return _bIsProcessed;};
	bool IsGoodForSolver() const {return _bGood4Solver;};
	void SetIsGoodForSolver(bool bValue) {_bGood4Solver = bValue;};

	// Do alignment and reset
	bool DoAlignment(bool bBayerSkipDemosaic, bool bApplyCorrSizeUpLimit=false, bool* pbCorrSizeReduced=NULL);	
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

	void NominalCenterInWorld(double* fdx, double* fdy);

	int GetIndex() {return _iIndex;};
	void SetIndex(int iValue) {_iIndex = iValue;}; 

	// For Debug
	void DumpImg(string sFileName) const;
	bool DumpImgWithResult(string sFileName) const;

protected:
	bool SqRtCorrelation(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced);
	bool NGCCorrelation(bool bSmooth, bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced);
	int MaskedNgc(bool bSmooth, UIRect tempRoi, UIRect searchRoi);

private:
	Image* _pImg1;
	Image* _pImg2;
	Image* _pMaskImg;

	UIRect _roi1;
	UIRect _roi2;

	Overlap* _pOverlap;
	int _iIndex;

	bool _bUseMask;
	bool _bIsProcessed;
	bool _bGood4Solver;
	bool _bUsedNgc;	// Ngc is used for calcaulation
	unsigned int _iDecim;	// For regoff
	unsigned int _iNgcColSearchExpansion; // For NGC
	unsigned int _iNgcRowSearchExpansion;

	CorrelationResult _result;
};