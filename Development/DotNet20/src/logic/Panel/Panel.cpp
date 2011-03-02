#include "Panel.h"
#include "Cad2Img.h"
//#include "System.h"

double Panel::_padInspectionAreaLong;
double Panel::_padInspectionAreaShort;

Panel::Panel(double lengthX, double lengthY, double pixelSizeX, double pixelSizeY) :
	_name(""),
	_status(UNINITIALIZED),
	LengthX(lengthX),
	LengthY(lengthY),
	PixelSizeX(pixelSizeX),
	PixelSizeY(pixelSizeY),
	XCadOrigin(0.0),
	YCadOrigin(0.0)
{
	Pads.clear();
	_apertureIndex = (unsigned short) pow(2.0, (int)(sizeof(unsigned short)*8)) - 1;
	_currentFeature = endFeatures();
	_currentFiducial = endFiducials();
	_padInspectionAreaLong = .60;
	_padInspectionAreaShort = 1.20;
	_cadBuffer = NULL;
	_maskBuffer = NULL;
	_aperatureBuffer = NULL;
}

Panel::~Panel()
{
	ClearFeatures();
	ClearFiducials();
	ClearBuffers();
}

void Panel::ClearBuffers()
{
	if(_aperatureBuffer != NULL)
	{
		delete[] _aperatureBuffer;
		_aperatureBuffer = NULL;
	}

	if(_maskBuffer != NULL)
	{
		delete[] _maskBuffer;
		_maskBuffer = NULL;
	}

	if(_cadBuffer != NULL)
	{
		delete[] _cadBuffer;
		_cadBuffer = NULL;
	}
}

void Panel::Name(string name)
{
	_name = name;
}

string Panel::Name()
{
	return _name;
}

void Panel::Status(State newStatus)
{
	_status = newStatus;
}

Panel::State Panel::Status()
{
	return _status;
}

// not optimized for speed
double Panel::xExtentOfPads()
{
	double minX(999999);
	double maxX(0);

	for(FeatureListIterator feature(beginFeatures()); feature!=endFeatures(); ++feature)
	{
		Point pMin = feature->second->GetBoundingBox().Min();
		Point pMax = feature->second->GetBoundingBox().Max();

		minX = pMin.x<minX?pMin.x:minX;
		maxX = pMax.x>maxX?pMax.x:maxX;
	}

	return (maxX-minX);
}

void Panel::ResetForNewInspection()
{
	for(FeatureListIterator feature = beginFeatures(); feature!=endFeatures(); feature++)
	{
		feature->second->ResetResults();
	}
}

void Panel::ClearFeatures()
{
	for(FeatureListIterator pad = beginFeatures(); pad!= endFeatures(); ++pad)
		delete pad->second;

	Pads.clear();
	_apertureIndex = (unsigned short) pow(2.0, (int)(sizeof(unsigned short)*8)) - 1;

	ClearBuffers();
}

void Panel::ClearFiducials()
{
	for(FeatureListIterator fid=beginFiducials(); fid!=endFiducials(); ++fid)
		delete fid->second;

	Fids.clear();
}

// not optimized for speed
double Panel::yExtentOfPads()
{
	double minY(999999);
	double maxY(0);

	for(FeatureListIterator feature(beginFeatures()); feature!=endFeatures(); ++feature)
	{
		Point pMin = feature->second->GetBoundingBox().Min();
		Point pMax = feature->second->GetBoundingBox().Max();

		minY = pMin.y<minY?pMin.y:minY;
		maxY = pMax.y>maxY?pMax.y:maxY;
	}

	return (maxY-minY);
}

int Panel::AddFeature(Feature* pad)
{
	if(!pad->Validate(this))
		return -1;

	if(!(Pads.find(pad->GetId())==endFeatures()))
		return -2;

	pad->SetApertureValue(_apertureIndex);
	_apertureIndex--;
	Pads[pad->GetId()] = pad;
	return 0;
}

void Panel::RemoveFeature(int featureId)
{
	Pads.erase(featureId);
}

int Panel::AddFiducial(Feature* fid)
{
	if(!fid->Validate(this))
		return -1;

	if(!(Fids.find(fid->GetId())==endFiducials()))
		return -2;

	fid->SetApertureValue(0);
	Fids[fid->GetId()] = fid;
	return 0;
}

void Panel::RemoveFiducial(int fiducialId)
{
	Fids.erase(fiducialId);
}

FeatureListIterator Panel::beginFeatures()
{
	return Pads.begin();
}

FeatureListIterator Panel::endFeatures()
{
	return Pads.end();
}

FeatureListIterator Panel::beginFiducials()
{
	return Fids.begin();
}

FeatureListIterator Panel::endFiducials()
{
	return Fids.end();
}

unsigned int Panel::NumberOfFeatures()
{
	return static_cast<unsigned int>(Pads.size());
}

unsigned int Panel::NumberOfFiducials()
{
	return static_cast<unsigned int>(Fids.size());
}

double Panel::xCadOrigin()
{
	return XCadOrigin;
}

void Panel::xCadOrigin(double x)
{
	XCadOrigin = x;
}

double Panel::yCadOrigin()
{
	return YCadOrigin;
}

void Panel::yCadOrigin(double y)
{
	YCadOrigin = y;
}

Feature * Panel::GetFirstFeature()
{
	_currentFeature = beginFeatures();
	return GetNextFeature();
}

Feature * Panel::GetNextFeature()
{
	Feature * pFeature=NULL;

	if(_currentFeature != endFeatures())
	{
		pFeature = _currentFeature->second;
		_currentFeature++;
	}

	return pFeature;

}

Feature * Panel::GetFirstFiducial()
{
	_currentFiducial = beginFiducials();
	return GetNextFiducial();
}

Feature * Panel::GetNextFiducial()
{
	Feature * pFiducial=NULL;

	if(_currentFiducial != endFiducials())
	{
		pFiducial = _currentFiducial->second;
		_currentFiducial++;
	}

	return pFiducial;
}

unsigned char* Panel::GetCadBuffer()
{
	if(_cadBuffer == NULL)
	{
		_cadBuffer = new unsigned char[GetNumPixelsInX()*GetNumPixelsInY()];
		memset(_cadBuffer, 0, GetNumPixelsInX()*GetNumPixelsInY());	
		Cad2Img::DrawCAD(this, _cadBuffer, false);
	}
	return _cadBuffer;
}

unsigned char* Panel::GetMaskBuffer()
{
	/// @todo !!!!!!!!!!!!!!!!!!
	/// Not quite sure how to handle this yet... what if we want different scaled masks
	return NULL;


	if(_maskBuffer == NULL)
	{
		_maskBuffer = new unsigned char[GetNumPixelsInX()*GetNumPixelsInY()];
		memset(_maskBuffer, 0, GetNumPixelsInX()*GetNumPixelsInY());	
	}

	return _maskBuffer;
}

unsigned short* Panel::GetAperatureBuffer()
{
	if(_aperatureBuffer == NULL)
	{
		_aperatureBuffer = new unsigned short[GetNumPixelsInX()*GetNumPixelsInY()];
		memset(_aperatureBuffer, 0, GetNumPixelsInX()*GetNumPixelsInY()*2);	
		Cad2Img::DrawAperatures(this, _aperatureBuffer, false);
	}

	return _aperatureBuffer;
}

