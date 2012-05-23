#pragma once

#include "panel.h" // include "Fearure.h" will lead confusion
#include "OverlapDefines.h"
//#include "RobustSolver.h"
#include <list>
using std::list;

// Fiducial alignment result for one physical/panel fiducial
class PanelFiducialResults
{
public:
	PanelFiducialResults(void);
	~PanelFiducialResults(void);

	void SetFeaure(Feature* pFeature) {_pFeature = pFeature;};
	void AddFidFovOvelapPoint(FidFovOverlap* pOverlap)
		{_fidFovOverlapPointList.push_back(pOverlap);};
	void LogResults();
	
	list<FidFovOverlap*>* GetFidOverlapListPtr() {return(&_fidFovOverlapPointList);};

	Feature* GetFeaturePtr() {return _pFeature;};

	double CalConfidence(int iDeviceIndex = -1);

	int GetId() {return _pFeature->GetId();};
	double GetCadX() {return _pFeature->GetCadX();};
	double GetCadY() {return _pFeature->GetCadY();};

private:
	Feature* _pFeature;
	list<FidFovOverlap*> _fidFovOverlapPointList;
};

// Collection of fiducial alignment results
class PanelFiducialResultsSet
{
public:
	PanelFiducialResultsSet(unsigned int iSize);
	~PanelFiducialResultsSet();

	void LogResults();

	double  CalConfidence(int iDeviceIndex = -1);
	double		GetPanelSkew(){return _dPanelSkew;};	
	void		SetPanelSkew(double skew){_dPanelSkew = skew;};	
	double		GetXscale(){return _dXscale;};	
	void		SetXscale(double scale){_dXscale= scale;};	
	double		GetYscale(){return _dYscale;};	
	void		SetYscale(double scale){_dYscale= scale;};	
	
	int Size() {return(_iSize); };

	PanelFiducialResults* GetPanelFiducialResultsPtr(unsigned int i) {return &(_pResultSet[i]);};

private:
	int _iSize;
	PanelFiducialResults* _pResultSet;
	double				_dPanelSkew;		// estimate of skew of panel (unit-less number indicating how much the board has warped from rectangular)
	double				_dXscale;			// estimates of scale error in X and Y directions
	double				_dYscale;
};

// World distance between two fiducials in the fiducial overlap
class FiducialDistance
{
public:
	FiducialDistance::FiducialDistance(
	FidFovOverlap* pFidOverlap1,
	ImgTransform trans1,
	FidFovOverlap* pFidOverlap2,
	ImgTransform trans2);

	bool IsWithOverlap(FidFovOverlap* pFidOverlap);
	double CalTranScale();
	void NormalizeTransDis(double dScale);

	FidFovOverlap* _pFidOverlap1;
	FidFovOverlap* _pFidOverlap2;
	double _dCadDis;
	double _dTransDis;
	bool _bValid;
	bool _bFromOutlier;
	bool _bNormalized;
};

/*// Check the validation of fiducial results
class FiducialResultCheck
{
public:
	FiducialResultCheck(PanelFiducialResultsSet* pFidSet, RobustSolver* pSolver);

	int CheckFiducialResults();
private:
	PanelFiducialResultsSet* _pFidSet;
	RobustSolver* _pSolver;

	list<FiducialDistance> _fidDisList;
};*/



