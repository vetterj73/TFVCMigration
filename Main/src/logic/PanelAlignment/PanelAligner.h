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

#include <ctime>

using namespace MosaicDM;

typedef void (*ALIGNMENTDONE_CALLBACK)(bool status);

#ifdef PANELALIGNER_EXPORTS
#define PANELALIGNER_API __declspec(dllexport) 
#else
#define PANELALIGNER_API __declspec(dllimport) 
#endif

class PanelAligner
{
public:
	PANELALIGNER_API PanelAligner(void);
	PANELALIGNER_API ~PanelAligner(void);

	PANELALIGNER_API bool ChangeProduction(MosaicSet* pSet, Panel *pPanel);

	PANELALIGNER_API void ResetForNextPanel();

	PANELALIGNER_API bool ImageAddedToMosaicCallback(
		unsigned int iLayerIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);

	PANELALIGNER_API LoggableObject* GetLogger();

	///
	///	Saves a 3 Channel image to disk.  This is somewhat of a helper function
	/// for diagnostics.  The images are expected to be all of the same width, height and stride
	/// 
	PANELALIGNER_API bool Save3ChannelImage(
		string filePath, 
		unsigned char *pChannel1, unsigned char* pChannel2, unsigned char* pChannel3, 
		int numRows, int numColumns);

	PANELALIGNER_API bool Save3ChannelImage(
		string filePath,
		unsigned char *pChannel1, int iSpan1,
		unsigned char* pChannel2, int iSpan2,
		unsigned char* pChannel3, int iSpan3,
		int numColumns, int numRows);

	PANELALIGNER_API void LogFiducialOverlaps(bool bLog);
	PANELALIGNER_API void LogOverlaps(bool bLog);
	PANELALIGNER_API void LogTransformVectors(bool bLog);
	PANELALIGNER_API void LogPanelEdgeDebugImages(bool bLog);
	PANELALIGNER_API void NumThreads(unsigned int numThreads);
	PANELALIGNER_API void FiducialSearchExpansionXInMeters(double fidSearchXInMeters);
	PANELALIGNER_API void FiducialSearchExpansionYInMeters(double fidSearchYInMeters);
	PANELALIGNER_API void UseCyberNgc4Fiducial();
	PANELALIGNER_API void UseProjectiveTransform(bool bValue);
	PANELALIGNER_API void UseCameraModelStitch(bool bValue);
	PANELALIGNER_API void UseCameraModelIterativeStitch(bool bValue);
	PANELALIGNER_API void SetUseTwoPassStitch(bool bValue);
	PANELALIGNER_API void EnableFiducialAlignmentCheck(bool bValue);
	PANELALIGNER_API void SetPanelEdgeDetection(
		bool bDetectPanelEdge, 
		int iLayer4Edge,
		bool bConveyorLeft2Right,
		bool bConveyorFixedFrontRail);
	PANELALIGNER_API void SetCalibrationWeight(double dValue);
	PANELALIGNER_API void SetSkipDemosaic(bool bValue);

	PANELALIGNER_API FidFovOverlapList* GetLastProcessedFids();
	
	PANELALIGNER_API PanelFiducialResultsSet* GetFidResultsSetPoint();

	PANELALIGNER_API void RegisterAlignmentDoneCallback(ALIGNMENTDONE_CALLBACK pCallback, void* pContext);
	PANELALIGNER_API void UnregisterAlignmentDoneCallback();

	PANELALIGNER_API MosaicSet* GetMosaicSet();

	// Overall alignment time for a single panel (only valid when demosaic and alignment are seperated)
	PANELALIGNER_API double GetAlignmentTime();
	
	PANELALIGNER_API bool GetCamModelPanelHeight(unsigned int iDeviceIndex, double pZCoef[16]);

protected:
	// CleanUp internal stuff for new production or desctructor
	void CleanUp();

	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const;
	bool CreateImageOrderInSolver(
		unsigned int* piLayerIndices, 
		unsigned iNumLayer, 
		map<FovIndex, unsigned int>* pOrderMap) const;

	bool CreateTransforms();
	void AddOverlapResults2Solver(
		RobustSolver* solver, 
		bool bUseFiducials, 
		bool bPinPanelWithCalibration=false,
		bool bUseNominalTransform=true);
	bool AlignWithPanelEdge(const EdgeInfo* pEdgeInfo, int iFidIndex = -1);
	bool UseEdgeInfomation();
	void AddCurPanelFidOverlapResults(RobustSolver* solver);
	void AddCurPanelFidOverlapResultsForPhyiscalFiducial(RobustSolver* solver, int iIndex);
	void AddSupplementOverlapResults(RobustSolver* solver);

	int FiducialAlignmentCheckOnCalibration();
	bool PickOneAlign4EachPanelFiducial();

	// For mask
	void CalTransformsWithMask();

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
	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;
	bool _bResultsReady;

	int _iNumFovProced;

	// for debug
	int _iPanelCount;
	clock_t _StartTime;
	double _dAlignmentTime;
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

