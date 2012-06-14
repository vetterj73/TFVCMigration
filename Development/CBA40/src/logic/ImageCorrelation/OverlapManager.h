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
#include "FiducialResult.h"
#include "PanelEdgeDetection.h"

using namespace MosaicDM;
class Panel;

typedef list<CadFovOverlap> CadFovOverlapList;
typedef CadFovOverlapList::iterator CadFovOverlapListIterator;
typedef list<FovFovOverlap> FovFovOverlapList;
typedef FovFovOverlapList::iterator FovFovOverlapListIterator;
typedef list<FovFovOverlap*> FovFovOverlapPtrList;
typedef FovFovOverlapPtrList::iterator FovFovOverlapPtrListIterator;
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

	bool DoAlignment4AllFiducial(bool bForCurPanel, bool bHasEdgeFidInfo=false);

	FovFovOverlapPtrList* GetFovFovPtrListForFov(
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

	FovFovOverlapList* GetFovFovOvelapSetPtr() {return &_fovFovOverlapSet;};

	FidFovOverlapList* GetCurPanelFidFovList4Fid(
		unsigned int iFidIndex) const;

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
	PanelEdgeDetection* GetEdgeDetector() {return _pEdgeDetector;};

	bool FinishOverlaps();

	PanelFiducialResultsSet* GetFidResultsSetPoint() { return _pFidResultsSet;};
	void CreateFiducialResultSet(bool bCurPanel);

	bool FovFovAlignConsistCheckForPanel(int* piCoarseInconsistNum, int* piFineInconsistNum);

	// for supplement overlaps
	int CalSupplementOverlaps();
	FovFovOverlapList* GetSupplementOverlaps();

protected:
	bool IsCadImageNeeded();
	bool IsMaskImageNeeded();

	void CreateFovFovOverlaps();	
	void CreateCadFovOverlaps();
	void CreateFidFovOverlaps(bool bForCurPanel=false, bool bHasEdgeFidInfo = false);
	
	bool CreateFovFovOverlapsForTwoLayer(unsigned int iIndex1, unsigned int iIndex2);

	void CalMaskCreationStage();

	unsigned int MaxCorrelations(unsigned int* piLayerIndices, unsigned int iNumLayer) const;
	bool IsFovFovOverlapForLayers(FovFovOverlap* pOverlap, unsigned int* piLayerIndices, unsigned int iNumLayer) const;

	void CreateFiducialImage(
		Image* pImage, 
		Feature* pFeature,
		double dExpandX,
		double dExpandY);

	static void RenderFiducial(
		Image* pImg, 
		Feature* pFid, 
		double resolution, 
		double dScale,
		double dExpandX,
		double dExpandY);

	int CreateNgcFidTemplate(
		Image* pImage, 
		Feature* pFeature,
		bool bFidBrighterThanBackground,
		bool bFiducialAllowNegativeMatch);

	bool CreateFidOverlapForLayer(
		MosaicLayer *pLayer, 
		Image* pFidImage, 	
		int iFidIndex,
		unsigned int iTemplateID,
		bool bSingleOverlap,
		int iMinOvelapWidth,
		int iMinOvelapHeight);  

	void CreateFidFovOverlaps4Fid(
		Feature* pFidFeature,		
		int iFidIndex,
		double dSearchExpX,
		double dSearchExpY,
		Image* pFidImage);

	bool FovFovAlignConsistCheckForTwoLayer(
		unsigned int iLayer1, unsigned int iLayer2,
		int* piCoarseInconsistNum, int* piFineInconsistNum);

	bool FovFovAlignConsistChekcForTwoTrig(
		unsigned int iLayer1, unsigned int iTrig1,
		unsigned int iLayer2, unsigned int iTrig2,
		int* piCoarseInconsistNum, int* piFineInconsistNum);

	int FovFovCoarseInconsistCheck(list<FovFovOverlap*>* pList);
	int FovFovFineInconsistCheck(list<FovFovOverlap*>* pList);

	// For supplement overlaps
	bool IsValid4SupplementCheck(FovFovOverlap* pOverlap);
	bool AddSingleSupplementOverlap(
		MosaicLayer* pLayer,
		unsigned int iTrigIndex, 
		unsigned int iCamIndex,
		bool bNexTrigIncrease);
	int AddSupplementOverlapsforSingleOvelap(FovFovOverlap* pOverlap);
	int AddSupplementOverlaps();

private:	
	MosaicSet *_pMosaicSet;
	Panel* _pPanel;
	DRect _validRect;
	unsigned int _numThreads;

	Image* _pCadImg;
	Image* _pPanelMaskImg;

	unsigned int _iNumCameras;
	unsigned int _iNumTriggers;

	Image* _pFidImages;

	// A[Layer Index][Trigger Index][Camera Index]
	FovFovOverlapPtrList*** _fovFovOverlapPtrLists;
	FovFovOverlapList _fovFovOverlapSet;
	CadFovOverlapList*** _cadFovOverlapLists;
	FidFovOverlapList*** _fidFovOverlapLists;

	unsigned int _iMinOverlapSize;

	int _iMaskCreationStage;

	CyberJob::JobManager *_pJobManager;

	PanelFiducialResultsSet* _pFidResultsSet;

	// For Panel Edge detection
	PanelEdgeDetection* _pEdgeDetector;	
	FidFovOverlapList* _curPanelFidFovOverlapLists;	
	Image* _pCurPanelFidImages;

	// For supplememt overlaps
	FovFovOverlapList _supFovFovOvelapList;
};

