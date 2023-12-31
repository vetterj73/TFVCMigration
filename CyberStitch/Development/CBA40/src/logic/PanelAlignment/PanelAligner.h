/*
	This is the unmanaged interface class of alignment tool 
*/
#pragma once
#include "MosaicSet.h"
#include "Logger.h"
#include "RobustSolver.h"
#include "RobustSolverFOV.h"
#include "RobustSolverCM.h"
#include "RobustSolverIterative.h"
#include "CorrelationParameters.h"
#include <map>
#include "OverlapManager.h"
using std::map;

using namespace MosaicDM;

typedef void (*ALIGNMENTDONE_CALLBACK)(bool status);

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool ChangeProduction(MosaicSet* pSet, Panel *pPanel);

	void ResetForNextPanel();

	bool ImageAddedToMosaicCallback(
		unsigned int iLayerIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);

	LoggableObject* GetLogger() {return &LOG;};

	///
	///	Saves a 3 Channel image to disk.  This is somewhat of a helper function
	/// for diagnostics.  The images are expected to be all of the same width, height and stride
	/// 
	bool Save3ChannelImage(
		string filePath, 
		unsigned char *pChannel1, unsigned char* pChannel2, unsigned char* pChannel3, 
		int numRows, int numColumns);

	bool Save3ChannelImage(
		string filePath,
		unsigned char *pChannel1, int iSpan1,
		unsigned char* pChannel2, int iSpan2,
		unsigned char* pChannel3, int iSpan3,
		int numColumns, int numRows);

	void LogFiducialOverlaps(bool bLog);
	void LogOverlaps(bool bLog);
	void LogTransformVectors(bool bLog);
	void LogPanelEdgeDebugImages(bool bLog);
	void NumThreads(unsigned int numThreads);
	void FiducialSearchExpansionXInMeters(double fidSearchXInMeters);
	void FiducialSearchExpansionYInMeters(double fidSearchYInMeters);
	void UseCyberNgc4Fiducial();
	void UseProjectiveTransform(bool bValue);
	void UseCameraModelStitch(bool bValue);
	void UseCameraModelIterativeStitch(bool bValue);
	void EnableFiducialAlignmentCheck(bool bValue);
	void SetPanelEdgeDetection(
		bool bDetectPanelEdge, 
		int iLayer4Edge,
		bool bConveyorLeft2Right,
		bool bConveyorFixedFrontRail);
	void SetCalibrationWeight(double dValue);

	FidFovOverlapList* GetLastProcessedFids();
	
	PanelFiducialResultsSet* GetFidResultsSetPoint() {return _pOverlapManager->GetFidResultsSetPoint();};

	void RegisterAlignmentDoneCallback(ALIGNMENTDONE_CALLBACK pCallback, void* pContext);
	void UnregisterAlignmentDoneCallback();

	MosaicSet* GetMosaicSet() {return _pSet;};

protected:
	// CleanUp internal stuff for new production or desctructor
	void CleanUp();

	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const;
	bool CreateImageOrderInSolver(
		unsigned int* piLayerIndices, 
		unsigned iNumLayer, 
		map<FovIndex, unsigned int>* pOrderMap) const;

	bool IsReadyToCreateMasks() const;
	bool CreateMasks();
	bool CreateTransforms();
	void AddOverlapResults2Solver(RobustSolver* solver, bool bUseFiducials, bool bPinPanelWithCalibration=false);
	bool AlignWithPanelEdge(const EdgeInfo* pEdgeInfo, int iFidIndex = -1);
	bool UseEdgeInfomation();
	void AddCurPanelFidOverlapResults(RobustSolver* solver);
	void AddCurPanelFidOverlapResultsForPhyiscalFiducial(RobustSolver* solver, int iIndex);
	void AddSupplementOverlapResults(RobustSolver* solver);

	int FiducialAlignmentCheckOnCalibration();
	bool PickOneAlign4EachPanelFiducial();

	// For debug
	void DisturbFiducialAlignment();
	void TestGetImagePatch();
	void TestSingleImagePatch();
	void OutputTransforms(string fileName);
	void OutputFiducialForSolver(string fileName);

private:
	FidFovOverlapList _lastProcessedFids;

	HANDLE _queueMutex;

	ALIGNMENTDONE_CALLBACK _registeredAlignmentDoneCallback;
	void * _pCallbackContext;
	void FireAlignmentDone(bool status);

	// Inputs
	MosaicSet* _pSet;		
	Panel* _pPanel;
	OverlapManager* _pOverlapManager;
	int _iMaskCreationStage;
	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;
	RobustSolver* _pMaskSolver;
	map<FovIndex, unsigned int> _maskMap;
	bool _bMasksCreated;
	bool _bResultsReady;

	int _iNumFovProced;

	// for debug
	int _iPanelCount;
};

// moved here because RobustSolver needs to use the PanelFiducialResultsSet class 
// (moved to avoid circular references)
// Check the validation of fiducial results
class FiducialResultCheck
{
public:
	FiducialResultCheck(PanelFiducialResultsSet* pFidSet, RobustSolver* pSolver);

	int CheckFiducialResults();
private:
	PanelFiducialResultsSet* _pFidSet;
	RobustSolver* _pSolver;

	list<FiducialDistance> _fidDisList;
};

