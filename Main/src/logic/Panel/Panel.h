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
	
	/* constructor */	Panel(double lengthX, double lengthY, double pixelSizeX, double pixelSizeY);

	/* destructor */	~Panel();

	static double		_padInspectionAreaLong;
	static double		_padInspectionAreaShort;

	int GetNumPixelsInX() {return(int) floor((LengthX/PixelSizeX)+0.5);}

	int GetNumPixelsInY() {return(int) floor((LengthY/PixelSizeY)+0.5);}

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
	void				ClearBuffers();
	void				Name(string name);
	string				Name();

	double				xExtentOfPads();
	double				yExtentOfPads();
	double				xLength(){return LengthX;};
	double				yLength(){return LengthY;};

	double				GetPixelSizeX(){return PixelSizeX;};
	double				GetPixelSizeY(){return PixelSizeX;};

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

	/**
		These buffers are created/populated on first time access.
		Client can choose when to create this buffer by calling the function
		at the appropriate time (if it is ever needed).
	*/
	unsigned char* GetCadBuffer();
	unsigned char* GetHeightImageBuffer(double dHeightResolution);
	unsigned char* GetHeightImageBuffer();
	unsigned char* GetMaskBuffer(int iCadExpansion);
	unsigned short* GetAperatureBuffer();
	bool HasCadBuffer(){return _cadBuffer!=NULL;};
	bool HasHeightImageBuffer(){return _heightImageBuffer!=NULL;}
	bool HasMaskBuffer(){return _maskBuffer!=NULL;};
	bool HasAperatureBuffer(){return _aperatureBuffer!=NULL;};

	double GetMaxComponentHeight();
	double GetHeightResolution() {return _dHeightResolution;};;

private:
	unsigned char* _cadBuffer;
	unsigned char* _heightImageBuffer;
	unsigned char* _maskBuffer;
	unsigned short* _aperatureBuffer;

	double _dHeightResolution;

	Panel(){};
	Panel(const Panel& p){};
	void operator=(const Panel& p){};

	FeatureList			Pads;
	FeatureList			Fids;

	FeatureListIterator _currentFeature;
	FeatureListIterator	_currentFiducial;

	string				_name;

	double				LengthX;
	double				LengthY;
	double				PixelSizeX;
	double				PixelSizeY;

	// these values specify the location
	// of the CAD origin within the Panel's
	// frame of reference
	double				XCadOrigin;
	double				YCadOrigin;
	unsigned short   	_apertureIndex;

	State				_status;
};