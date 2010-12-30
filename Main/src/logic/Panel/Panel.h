/*
	describes the CAD for one panel, including results
*/

#pragma once

#include "LoggableObject.h"
#include "Feature.h"
#include "Image.h"
#include <map>
using std::map;
typedef map<int, Feature*> FeatureList;
typedef FeatureList::iterator FeatureListIterator;

using namespace LOGGER;
class Panel : public LoggableObject
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

	static double		_padInspectionAreaLong;
	static double		_padInspectionAreaShort;

	int GetNumPixelsInX(double pixelSizeInMeters)
	{return(int) floor((LengthX/pixelSizeInMeters)+0.5);}

	int GetNumPixelsInY(double pixelSizeInMeters)
	{return(int) floor((LengthY/pixelSizeInMeters)+0.5);}

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

	void SetCadBuffer(unsigned char *buffer)
	{
		_cadBuffer = buffer;
	}
	
	void SetMaskBuffer(unsigned char *buffer)
	{
		_maskBuffer = buffer;
	}
	
	unsigned char* GetCadBuffer()
	{return _cadBuffer;}

	unsigned char* GetMaskBuffer()
	{return _maskBuffer;}


private:
	unsigned char *_cadBuffer;
	unsigned char *_maskBuffer;

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