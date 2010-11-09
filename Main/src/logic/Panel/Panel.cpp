#include "Panel.h"
#include "System.h"

Panel::Panel() :
	_name(""),
	_status(UNINITIALIZED),
	LengthX(0.0),
	LengthY(0.0),
	XCadOrigin(0.0),
	YCadOrigin(0.0)
{
	Pads.clear();
	_apertureIndex = (Word) pow(2.0, (int)(sizeof(Word)*8)) - 1;
	_currentFeature = endFeatures();
	_currentFiducial = endFiducials();
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
	_apertureIndex = (Word) pow(2.0, (int)(sizeof(Word)*8)) - 1;
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

SPIStatusCode Panel::AddFeature(Feature* pad)
{
	if(!pad->Validate(LengthX, LengthY))
		return STATUS_FEATURE_OUT_OF_BOUNDS;

	if(!(Pads.find(pad->GetId())==endFeatures()))
		return STATUS_FEATURE_ID_IN_USE;

	pad->SetApertureValue(_apertureIndex);
	_apertureIndex--;
	Pads[pad->GetId()] = pad;
	return STATUS_SUCCESS;
}

void Panel::RemoveFeature(int featureId)
{
	Pads.erase(featureId);
}

SPIStatusCode Panel::AddFiducial(Feature* fid)
{
	if(!fid->Validate(LengthX, LengthY))
		return STATUS_FEATURE_OUT_OF_BOUNDS;

	if(!(Fids.find(fid->GetId())==endFiducials()))
		return STATUS_FEATURE_ID_IN_USE;

	fid->SetApertureValue(0);
	Fids[fid->GetId()] = fid;
	return STATUS_SUCCESS;
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
