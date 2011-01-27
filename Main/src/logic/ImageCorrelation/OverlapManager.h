/*
 Class to create and manage the overlaps
*/

#pragma once
#include "CorrelationFlags.h"
#include "OverlapDefines.h"
#include "Panel.h"
#include "VsFinderCorrelation.h"
#include "MosaicSet.h"

using namespace MosaicDM;
class Panel;
class JobManager;
class OverlapManager
{
public:
	OverlapManager(
		MosaicSet* pMosaicSet,
		Panel* pPanel);
	~OverlapManager(void);

	bool ResetforNewPanel();

	bool DoAlignmentForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex);

	list<FovFovOverlap>* GetFovFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	list<CadFovOverlap>* GetCadFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	list<FidFovOverlap>* GetFidFovListForFov(
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
	list<FovFovOverlap>*** _fovFovOverlapLists;
	list<CadFovOverlap>*** _cadFovOverlapLists;
	list<FidFovOverlap>*** _fidFovOverlapLists;

	unsigned int _iMinOverlapSize;

	int _iMaskCreationStage;

	// For vsfinder
	VsFinderCorrelation* _pVsfinderCorr;
	unsigned int* _pVsFinderTempIds;
	JobManager *_pJobManager;
};

