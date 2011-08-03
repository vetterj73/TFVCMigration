#pragma once
#include "JobThread.h"
#include "UIRect.h"

class Image;
class MorphJob : public CyberJob::Job
{
public:
	MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow);

	virtual void Run();

protected:
	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
};

class MorphWithHeightJob : public MorphJob
{
public:
	MorphWithHeightJob(
		Image* pStitchedImage, 
		Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		Image* pHeightImage, 
		double dHeightResolution,
		double dPupilDistance);

	void Run();

protected:
	Image* _pHeightImage;
	double _dHeightResolution;
	double _dPupilDistance;
};

