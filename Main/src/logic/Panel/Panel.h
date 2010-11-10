/*
	describes the CAD for one panel, including results
*/

#pragma once

#include "Feature.h"
#include <map>
using std::map;

typedef map<int, Feature*> FeatureList;
typedef FeatureList::iterator FeatureListIterator;

class Panel
{
public:

	typedef enum {
		UNINITIALIZED	= 0,
		UNINSPECTED		= 10,
		INSPECTING		= 20,
		INSPECTED		= 30
	 }State;
	
	/* constructor */	Panel();

	/* destructor */	~Panel();

	static bool			_debug;
	static double		_padInspectionAreaLong;
	static double		_padInspectionAreaShort;

	int					AddFeature(Feature*);
	void				RemoveFeature(int featureId);
	unsigned int		NumberOfFeatures();

	int					AddFiducial(Feature*);
	void				RemoveFiducial(int fiducialId);
	unsigned int		NumberOfFiducials();

	FeatureListIterator beginFeatures();
	FeatureListIterator endFeatures();
	FeatureListIterator beginFiducials();
	FeatureListIterator endFiducials();

	void				ClearFeatures();
	void				ClearFiducials();
	void				Name(string name);
	string				Name();

	double				xExtentOfPads();
	double				yExtentOfPads();
	double				xLength();
	double				yLength();
	void				xLength(double x);
	void				yLength(double y);

	double				xCadOrigin();
	double				yCadOrigin();
	void                xCadOrigin(double x);
	void                yCadOrigin(double y);

	void				Status(State newStatus);
	State				Status();

	void				ResetForNewInspection();

	Feature *			GetFirstFeature();
	Feature *			GetNextFeature();

	Feature *			GetFirstFiducial();
	Feature *			GetNextFiducial();

private:
	Panel(const Panel& p){};
	void operator=(const Panel& p){};

	FeatureList			Pads;
	FeatureList			Fids;

	FeatureListIterator _currentFeature;
	FeatureListIterator	_currentFiducial;

	string				_name;

	double				LengthX;
	double				LengthY;

	// these values specify the location
	// of the CAD origin within the Panel's
	// frame of reference
	double				XCadOrigin;
	double				YCadOrigin;
	unsigned short   	_apertureIndex;

	State				_status;
};