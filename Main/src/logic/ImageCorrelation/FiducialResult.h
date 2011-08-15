#pragma once

#include "panel.h" // include "Fearure.h" will lead confusion
#include "OverlapDefines.h"
#include <list>
using std::list;

// Fiducial alignment result for one physical fiducial
class FiducialResult
{
public:
	FiducialResult(void);
	~FiducialResult(void);

	void SetFeaure(Feature* pFeature) {_pFeature = pFeature;};
	void AddFidFovOvelapPoint(FidFovOverlap* pOverlap)
		{_fidFovOverlapPointList.push_back(pOverlap);};
	void LogResults();

	double CalConfidence();

private:
	Feature* _pFeature;
	list<FidFovOverlap*> _fidFovOverlapPointList;
};

// Collection of fiducial alignment results
class FiducialResultSet
{
public:
	FiducialResultSet(unsigned int iSize);
	~FiducialResultSet();

	void LogResults();

	double  CalConfidence();

	FiducialResult* GetFiducialResultPoint(unsigned int i) {return &(_pResultSet[i]);};

private:
	int _iSize;
	FiducialResult* _pResultSet;
};



