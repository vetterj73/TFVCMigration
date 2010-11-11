#include "Panel.h"
//#include "System.h"

Panel::Panel() :
	_name(""),
	_status(UNINITIALIZED),
	LengthX(0.0),
	LengthY(0.0),
	XCadOrigin(0.0),
	YCadOrigin(0.0)
{
	Pads.clear();
	_apertureIndex = (unsigned short) pow(2.0, (int)(sizeof(unsigned short)*8)) - 1;
	_currentFeature = endFeatures();
	_currentFiducial = endFiducials();
	_padInspectionAreaLong = .60;
	_padInspectionAreaShort = 1.20;
}

Panel::~Panel()
{
	ClearFeatures();
	ClearFiducials();
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

void Panel::xLength(double x)
{
	LengthX = x;
}

void Panel::yLength(double y)
{
	LengthY = y;
}

double Panel::xLength()
{
	return LengthX;
}

double Panel::yLength()
{
	return LengthY;
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
