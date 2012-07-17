/*
	#define the classes for overlap between Fov and Fov, Cad and Fov, and Fiducial and Fov
*/

#pragma once

#include "MosaicTile.h"
#include "MosaicLayer.h"
#include "CorrelationPair.h"
#include "VsFinderCorrelation.h"
#include "CyberNgcFiducialCorrelation.h"
#include "JobThread.h"
#include "CorrelationParameters.h"
#include "CorrelationFlags.h"
using namespace MosaicDM;

// Base class for overlap between image and image
class Overlap : CyberJob::Job		
{
public:
	Overlap();
	Overlap(const Overlap& b);
	void operator=(const Overlap& b);

	~Overlap();

	void Config(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type,	
		bool bApplyCorrSizeUpLimit,
		MaskInfo* pMaskInfo = NULL);

	// Get/set
	Image* GetFirstImage() const {return _pImg1;};
	Image* GetSecondImage() const {return _pImg2;};

	unsigned int Columns() const {return _iColumns;};
	unsigned int Rows() {return _iRows;};

	bool IsValid() const {return _bValid;};
	bool IsProcessed() const {return _bProcessed;};
	bool IsGoodForSolver() const {return _bGood4Solver;};
	void SetIsGoodForSolver(bool bValue) {_bGood4Solver = bValue;};

	CorrelationPair* GetCoarsePair() {return &_coarsePair;};
	list<CorrelationPair>* GetFinePairListPtr()  {return &_finePairList;};

	OverlapType GetOverlapType() {return _type;};

	bool HasMaskPanelImage();
	void SetUseMask(bool bValue);
	void SetSkipCoarseAlign(bool bValue) {_bSkipCoarseAlign = bValue;};

	// Do alignment and reset
	void Run();
	bool Reset();

protected:
	bool CalCoarseCorrPair();
	bool ChopOverlap();	
	bool _bValid;
	bool _bProcessed;	
	bool _bGood4Solver;

	CorrelationPair _coarsePair;
	list<CorrelationPair> _finePairList;
	
	//For mask
	Image* _pMaskImg;	// Mask image is with first Fov/image
	MaskInfo* _pMaskInfo;
	
	virtual bool DumpOvelapImages()=0;
	virtual bool DumpResultImages()=0;

private:
	Image* _pImg1;
	Image* _pImg2;
	DRect _validRect;
	OverlapType _type;

	unsigned int _iColumns;
	unsigned int _iRows;

	bool _bApplyCorrSizeUpLimit;

	// For Mask
	bool _bUseMask;
	bool _bSkipCoarseAlign; 
};

// Overlap between FOV image and FOV image
// Image with lower layer index is the first image
class FovFovOverlap: public Overlap
{
public:
	FovFovOverlap(
		MosaicLayer*	pLayer1,
		MosaicLayer*	pLayer2,
		TilePosition ImgPos1, 
		TilePosition ImgPos2,
		DRect validRect,
		bool bApplyCorrSizeUpLimit,
		MaskInfo* pMaskInfo);

	MosaicLayer* GetFirstMosaicLayer() const {return _pLayer1;};
	unsigned int GetFirstTriggerIndex() const {return _imgPos1.iTrigIndex;};
	unsigned int GetFirstCameraIndex() const {return _imgPos1.iCamIndex;};
	TilePosition GetFirstImagePosition() const {return _imgPos1;};

	MosaicLayer* GetSecondMosaicLayer() const {return _pLayer2;};
	unsigned int GetSecondTriggerIndex() const {return _imgPos2.iTrigIndex;};
	unsigned int GetSecondCameraIndex() const {return _imgPos2.iCamIndex;};
	TilePosition GetSecondImagePosition() const {return _imgPos2;};

	bool IsReadyToProcess() const;

	bool IsFromLayerTrigs(
		unsigned int Layer1,
		unsigned int iTrig1,
		unsigned int Layer2,
		unsigned int iTrig2) const;

	bool IsFromSameDevice() const;

	bool IsAdjustedBasedOnCoarseAlignment() const { return _bAdjustedBaseOnCoarse;};
	void SetAdjustedBasedOnCoarseAlignment(bool bValue) {_bAdjustedBaseOnCoarse = bValue;};

	double CalWeightSum();

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();
	//bool HasMask() {return _maskInfo.;};

private:
	MosaicLayer*	_pLayer1;
	MosaicLayer*	_pLayer2;
	TilePosition _imgPos1;
	TilePosition _imgPos2;

	bool _bAdjustedBaseOnCoarse;  // Status 
};

// Overlap between CAD image and FOV image
// Fov image is always the first image
class CadFovOverlap: public Overlap
{
public:
	CadFovOverlap(
		MosaicLayer* pLayer,
		TilePosition ImgPos,
		Image* pCadImg,
		DRect validRect);

	MosaicLayer* GetMosaicLayer() const {return _pLayer;};
	unsigned int GetTriggerIndex() const {return _imgPos.iTrigIndex;};
	unsigned int GetCameraIndex() const {return _imgPos.iCamIndex;};
	TilePosition GetImagePosition() const {return _imgPos;};
	
	Image* GetCadImage() const {return _pCadImg;};

	bool IsReadyToProcess() const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicLayer*	_pLayer;
	TilePosition _imgPos;
	Image* _pCadImg;
};

// Overlap between Fiducial image and FOV image
// Fov image is always the first image
class FidFovOverlap: public Overlap
{
public:
	FidFovOverlap(
		MosaicLayer*	pLayer,
		TilePosition ImgPos, // first = camera, second = trigger
		Image* pFidImg,
		double dXcenter,
		double dYcenter,
		unsigned int iFidIndex,
		DRect validRect);

	void SetVsFinder(unsigned int iTemplateID);
	void SetNgcFid(unsigned int iTemplateID); 

	MosaicLayer* GetMosaicLayer() const {return _pLayer;};
	unsigned int GetTriggerIndex() const {return _imgPos.iTrigIndex;};
	unsigned int GetCameraIndex() const {return _imgPos.iCamIndex;};
	TilePosition GetImagePosition() const {return _imgPos;};
	Image* GetFidImage() const {return _pFidImg;};

	double GetFiducialXPos() const {return _dFidCenterX;};
	double GetFiducialYPos() const {return _dFidCenterY;};

	int GetFiducialIndex() const {return _iFidIndex;};

	bool IsReadyToProcess() const;
	
	double GetWeightForSolver(); 

	bool CalFidCenterBasedOnTransform(ImgTransform trans, double* pdx, double* pdy);
	

	// For Vsfinder
	FiducialSearchMethod GetFiducialSearchMethod() const {return _fidSearchMethod;};
	bool VsfinderAlign();
	bool NgcFidAlign();

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicLayer*	_pLayer;
	TilePosition _imgPos;

	Image* _pFidImg;

	double _dFidCenterX;
	double _dFidCenterY;

	unsigned int _iFidIndex;

	FiducialSearchMethod _fidSearchMethod;
	unsigned int _iTemplateID; // for Vsfinder or CyberNgc
};

