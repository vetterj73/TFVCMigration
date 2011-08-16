#pragma once

#include "panel.h" // include "Fearure.h" will lead confusion
#include "OverlapDefines.h"
#include <list>
using std::list;

// Fiducial alignment result for one physical fiducial
class FiducialResults
{
public:
	FiducialResults(void);
	~FiducialResults(void);

	void SetFeaure(Feature* pFeature) {_pFeature = pFeature;};
	void AddFidFovOvelapPoint(FidFovOverlap* pOverlap)
		{_fidFovOverlapPointList.push_back(pOverlap);};
	void LogResults();
	
	list<FidFovOverlap*>* GetResultListPtr() {return(&_fidFovOverlapPointList);};

	double CalConfidence();

	int GetId() {return _pFeature->GetId();};
	double GetCadX() {return _pFeature->GetCadX();};
	double GetCadY() {return _pFeature->GetCadY();};

private:
	Feature* _pFeature;
	list<FidFovOverlap*> _fidFovOverlapPointList;
};

// Collection of fiducial alignment results
class FiducialResultsSet
{
public:
	FiducialResultsSet(unsigned int iSize);
	~FiducialResultsSet();

	void LogResults();

	double  CalConfidence();

	int Size() {return(_iSize); };

	FiducialResults* GetFiducialResultsPtr(unsigned int i) {return &(_pResultSet[i]);};

private:
	int _iSize;
	FiducialResults* _pResultSet;
};



