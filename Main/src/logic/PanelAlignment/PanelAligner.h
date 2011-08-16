/*
	This is the unmanaged interface class of alignment tool 
*/
#pragma once
#include "MosaicSet.h"
#include "Logger.h"
#include "RobustSolver.h"
#include "CorrelationParameters.h"
#include <map>
#include "OverlapManager.h"
using std::map;

using namespace MosaicDM;

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
		string filePath, unsigned char *pChannel1, unsigned char* pChannel2, unsigned char* pChannel3, 
		int numRows, int numColumns);

	void LogFiducialOverlaps(bool bLog);
	void LogOverlaps(bool bLog);
	void LogMaskVectors(bool bLog);
	void NumThreads(unsigned int numThreads);
	void FiducialSearchExpansionXInMeters(double fidSearchXInMeters);
	void FiducialSearchExpansionYInMeters(double fidSearchYInMeters);
	void UseCyberNgc4Fiducial();

	FidFovOverlapList* GetLastProcessedFids();
	
	FiducialResultsSet* GetFidResultsSetPoint() {return _pOverlapManager->GetFidResultsSetPoint();};

protected:
	// CleanUp internal stuff for new production or desctructor
	void CleanUp();

	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const;
	bool CreateImageOrderInSolver(
		unsigned int* piIllumIndices, 
		unsigned iNumIllums, 
		map<FovIndex, unsigned int>* pOrderMap) const;

	bool IsReadyToCreateMasks() const;
	bool CreateMasks();
	bool CreateTransforms();
	void AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex);


private:
	FidFovOverlapList _lastProcessedFids;

	HANDLE _queueMutex;

	// Inputs
	MosaicSet* _pSet;		
	OverlapManager* _pOverlapManager;
	int _iMaskCreationStage;
	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;
	RobustSolver* _pMaskSolver;
	map<FovIndex, unsigned int> _maskMap;
	bool _bMasksCreated;
	bool _bResultsReady;
};

