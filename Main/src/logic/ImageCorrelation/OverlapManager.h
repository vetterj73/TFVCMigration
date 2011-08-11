/*
 Class to create and manage the overlaps
*/

#pragma once
#include "CorrelationFlags.h"
#include "OverlapDefines.h"
#include "Panel.h"
#include "VsFinderCorrelation.h"
#include "CyberNgcFiducialCorrelation.h"
#include "MosaicSet.h"
#include "JobManager.h"

using namespace MosaicDM;
class Panel;

typedef list<CadFovOverlap> CadFovOverlapList;
typedef CadFovOverlapList::iterator CadFovOverlapListIterator;
typedef list<FovFovOverlap> FovFovOverlapList;
typedef FovFovOverlapList::iterator FovFovOverlapListIterator;
typedef list<FidFovOverlap> FidFovOverlapList;
typedef FidFovOverlapList::iterator FidFovOverlapListIterator;

class OverlapManager
{
public:
	OverlapManager(
		MosaicSet* pMosaicSet,
		Panel* pPanel,
		unsigned int numThreads);
	~OverlapManager(void);

	bool ResetforNewPanel();

	bool DoAlignmentForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex);

	FovFovOverlapList* GetFovFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	CadFovOverlapList* GetCadFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	FidFovOverlapList* GetFidFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	int GetMaskCreationStage() {return _iMaskCreationStage;};
	unsigned int MaxCorrelations() const;
	unsigned int MaxMaskCorrelations() const;
	unsigned int MaxNumTriggers() {return(_iNumTriggers);};
	unsigned int MaxNumCameras() {return(_iNumCameras);};

	DRect GetValidRect() {return _validRect;};
	Panel* GetPanel() {return _pPanel;};
	double GetCadImageResolution() {return _pPanel->GetPixelSizeX();};

	Image* GetCadImage() {return _pCadImg;};
	Image* GetPanelMaskImage() {return _pPanelMaskImg;};
	MosaicSet *GetMosaicSet(){return _pMosaicSet;};
	bool FinishOverlaps();

protected:
	bool IsCadImageNeeded();
	bool IsMaskImageNeeded();

	void CreateFovFovOverlaps();	
	void CreateCadFovOverlaps();
	void CreateFidFovOverlaps();
	
	bool CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2);

	void CalMaskCreationStage();

	unsigned int MaxCorrelations(unsigned int* piIllumIndices, unsigned int iNumIllums) const;
	bool IsFovFovOverlapForIllums(FovFovOverlap* pOverlap, unsigned int* piIllumIndices, unsigned int iNumIllums) const;

	bool CreateFiducialImages();

	static void RenderFiducial(
		Image* pImg, 
		Feature* pFid, 
		double resolution, 
		double dScale);

	bool CreateVsfinderTemplates();
	bool CreateNgcFidTemplates();

private:	
	MosaicSet *_pMosaicSet;
	Panel* _pPanel;
	DRect _validRect;

	Image* _pCadImg;
	Image* _pPanelMaskImg;

	unsigned int _iNumCameras;
	unsigned int _iNumTriggers;

	Image* _pFidImages;
	
	// A[Mosaic Index][Row(y) Index][Column(x) Index]
	FovFovOverlapList*** _fovFovOverlapLists;
	CadFovOverlapList*** _cadFovOverlapLists;
	FidFovOverlapList*** _fidFovOverlapLists;

	unsigned int _iMinOverlapSize;

	int _iMaskCreationStage;

	// For vsfinder
	unsigned int* _pVsFinderTempIds;

	// For CyberNgc
	unsigned int* _pNgcFidTempIds;

	CyberJob::JobManager *_pJobManager;
};

