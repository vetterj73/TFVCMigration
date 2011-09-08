#include "StdAfx.h"
#include "MorphJob.h"
#include "Image.h"
#include "ColorImage.h"


MorphJob::MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		bool bColor)
{
	_rect.FirstColumn = firstCol;
	_rect.FirstRow = firstRow;
	_rect.LastColumn = lastCol;
	_rect.LastRow = lastRow;

	_pStitched = pStitchedImage;
	_pFOV = pFOV;
	_bColor = bColor;
}


void MorphJob::Run()
{
	if(_pStitched !=NULL && _rect.IsValid())
	{	
		if(_bColor)
			((ColorImage*)_pStitched)->ColorMorphFrom((ColorImage*)_pFOV, _rect);
		else
			_pStitched->MorphFrom(_pFOV, _rect);
	}
}

MorphWithHeightJob::MorphWithHeightJob(
		Image* pStitchedImage, 
		Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		Image* pHeightImage, 
		double dHeightResolution,
		double dPupilDistance)
	:MorphJob(pStitchedImage, pFOV, 
		firstCol, firstRow, lastCol, lastRow)
{
	_pHeightImage = pHeightImage;
	_dHeightResolution = dHeightResolution;
	_dPupilDistance = dPupilDistance;
}

void MorphWithHeightJob::Run()
{
	if(_pStitched !=NULL && _rect.IsValid())
		_pStitched->MorphFromWithHeight(_pFOV, _rect, _pHeightImage, _dHeightResolution, _dPupilDistance);
}