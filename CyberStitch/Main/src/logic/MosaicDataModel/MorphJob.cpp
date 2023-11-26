#include "StdAfx.h"
#include "MorphJob.h"
#include "Image.h"
#include "ColorImage.h"


MorphJob::MorphJob(
	Image* pStitchedImage, 
	Image *pFOV, 
	unsigned int firstCol,
	unsigned int firstRow,
	unsigned int lastCol,
	unsigned int lastRow,
	bool bDemosaic,
	BayerType type,
	bool bGaussianDemosaic,
	Image* pHeightImage, 
	double dHeightResolution,
	double dPupilDistance)
{
	_rect.FirstColumn = firstCol;
	_rect.FirstRow = firstRow;
	_rect.LastColumn = lastCol;
	_rect.LastRow = lastRow;	

	_bDemosaic = bDemosaic;
	_type = type;
	_bGaussianDemosaic = bGaussianDemosaic;

	_pHeightImage = pHeightImage;
	_dHeightResolution = dHeightResolution;
	_dPupilDistance = dPupilDistance;

	_pStitched = pStitchedImage;
	_pFOV = pFOV;
}


void MorphJob::Run()
{
	if(_pStitched !=NULL && _rect.IsValid())
	{	
		Image* pImg =  _pFOV;
		bool bIsYCrCb = true;
		
		// Do Demosaic if it is necessary
		if(_bDemosaic)
		{
			if(!_bGaussianDemosaic)
			{
				pImg = new ColorImage(YCrCb, true);
				((ColorImage*)pImg)->DemosaicFrom(_pFOV, _type);
			}
			else
			{
				pImg = new ColorImage(BGR, true);
				((ColorImage*)pImg)->DemosaicFrom_Gaussian(_pFOV, _type);
				bIsYCrCb = false;
			}
		}
		_pStitched->MorphFrom(pImg, bIsYCrCb, _rect, _pHeightImage, _dHeightResolution, _dPupilDistance);

		// Clean up if it is necessary
		if(_bDemosaic)
			delete pImg;
	}
}