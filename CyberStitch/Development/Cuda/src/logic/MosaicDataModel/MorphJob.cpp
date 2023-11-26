#include "StdAfx.h"
#include "MorphJob.h"
#include "Image.h"


MorphJob::MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow)
{
	_rect.FirstColumn = firstCol;
	_rect.FirstRow = firstRow;
	_rect.LastColumn = lastCol;
	_rect.LastRow = lastRow;

	_pStitched = pStitchedImage;
	_pFOV = pFOV;
}


void MorphJob::Run()
{
	if(_pStitched !=NULL && _rect.IsValid())
		_pStitched->MorphFrom(_pFOV, _rect);
}
