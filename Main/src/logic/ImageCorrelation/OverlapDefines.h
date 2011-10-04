/*
	#define the classes for overlap between Fov and Fov, Cad and Fov, and Fiducial and Fov
*/

#pragma once

#include "MosaicLayer.h"
#include "CorrelationPair.h"
#include "VsFinderCorrelation.h"
#include "CyberNgcFiducialCorrelation.h"
#include "JobThread.h"
#include "CorrelationParameters.h"
using namespace MosaicDM;

// Base class for overlap between image and image
class Overlap : CyberJob::Job		
{
public:
	Overlap();
	Overlap(const Overlap& b);
	void operator=(const Overlap& b);

	void config(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type,	
		bool bApplyCorrSizeUpLimit,
		Image* pMaskImg = NULL);

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

	virtual bool DumpOvelapImages()=0;
	virtual bool DumpResultImages()=0;

private:
	Image* _pImg1;
	Image* _pImg2;
	DRect _validRect;
	OverlapType _type;

	Image* _pMaskImg;

	unsigned int _iColumns;
	unsigned int _iRows;

	bool _bApplyCorrSizeUpLimit;
};

// Overlap between FOV image and FOV image
// Image with lower illumination index is the first image
class FovFovOverlap: public Overlap
{
public:
	FovFovOverlap(
		MosaicLayer*	pMosaic1,
		MosaicLayer*	pMosaic2,
		pair<unsigned int, unsigned int> ImgPos1, // first = camera, second = trigger
		pair<unsigned int, unsigned int> ImgPos2,
		DRect validRect,
		bool bApplyCorrSizeUpLimit,
		bool bHasMask,
		bool bFromSameDevice);

	MosaicLayer* GetFirstMosaicImage() const {return _pMosaic1;};
	unsigned int GetFirstTriggerIndex() const {return _imgPos1.second;};
	unsigned int GetFirstCameraIndex() const {return _imgPos1.first;};
	pair<unsigned int, unsigned int> GetFirstImagePosition() const {return _imgPos1;};

	MosaicLayer* GetSecondMosaicImage() const {return _pMosaic2;};
	unsigned int GetSecondTriggerIndex() const {return _imgPos2.second;};
	unsigned int GetSecondCameraIndex() const {return _imgPos2.first;};
	pair<unsigned int, unsigned int> GetSecondImagePosition() const {return _imgPos2;};
	bool IsFromSameDevice() {return _bFromSameDevice;};

	bool IsReadyToProcess() const;

	bool IsFromIllumTrigs(
		unsigned int iLlum1,
		unsigned int iTrig1,
		unsigned int iLlum2,
		unsigned int iTrig2) const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();
	bool HasMask() {return _bHasMask;};

private:
	MosaicLayer*	_pMosaic1;
	MosaicLayer*	_pMosaic2;
	pair<unsigned int, unsigned int> _imgPos1;
	pair<unsigned int, unsigned int> _imgPos2;
	bool _bHasMask;
	bool _bFromSameDevice;
};

// Overlap between CAD image and FOV image
// Fov image is always the first image
class CadFovOverlap: public Overlap
{
public:
	CadFovOverlap(
		MosaicLayer* pMosaic,
		pair<unsigned int, unsigned int> ImgPos, // first = camera, second = trigger
		Image* pCadImg,
		DRect validRect);

	MosaicLayer* GetMosaicImage() const {return _pMosaic;};
	unsigned int GetTriggerIndex() const {return _imgPos.second;};
	unsigned int GetCameraIndex() const {return _imgPos.first;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	
	Image* GetCadImage() const {return _pCadImg;};

	bool IsReadyToProcess() const;

	// For debug
	bool DumpOvelapImages();
	bool DumpResultImages();

private:
	MosaicLayer*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;
	Image* _pCadImg;
};

// Overlap between Fiducial image and FOV image
// Fov image is always the first image
class FidFovOverlap: public Overlap
{
public:
	FidFovOverlap(
		MosaicLayer*	pMosaic,
		pair<unsigned int, unsigned int> ImgPos, // first = camera, second = trigger
		Image* pFidImg,
		double _dXcenter,
		double _dYcenter,
		DRect validRect);

	void SetVsFinder(unsigned int iTemplateID);
	void SetNgcFid(unsigned int iTemplateID); 

	MosaicLayer* GetMosaicImage() const {return _pMosaic;};
	unsigned int GetTriggerIndex() const {return _imgPos.second;};
	unsigned int GetCameraIndex() const {return _imgPos.first;};
	pair<unsigned int, unsigned int> GetImagePosition() const {return _imgPos;};
	Image* GetFidImage() const {return _pFidImg;};

	double GetFiducialXPos() const {return _dFidCenterX;};
	double GetFiducialYPos() const {return _dFidCenterY;};

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
	MosaicLayer*	_pMosaic;
	pair<unsigned int, unsigned int> _imgPos;

	Image* _pFidImg;

	double _dFidCenterX;
	double _dFidCenterY;

	FiducialSearchMethod _fidSearchMethod;
	unsigned int _iTemplateID; // for Vsfinder or CyberNgc
};

