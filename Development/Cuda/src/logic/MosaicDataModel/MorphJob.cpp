#include "StdAfx.h"
#include "MorphJob.h"
#include "Image.h"


MorphJob::MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		unsigned int ordinal)
{
	_rect.FirstColumn = firstCol;
	_rect.FirstRow = firstRow;
	_rect.LastColumn = lastCol;
	_rect.LastRow = lastRow;

	_pStitched = pStitchedImage;
	_pFOV = pFOV;
	_ordinal = ordinal;
}


void MorphJob::Run()
{
	//printf_s("Thread execution: oridinal - %ld;\n", _ordinal);

	if(_pStitched !=NULL && _rect.IsValid())
		_pStitched->MorphFrom(_pFOV, _rect);
}

bool MorphJob::GPURun(CyberJob::GPUJobStream *jobStream)
{
	bool results = true; // true = conversion complete
	
	//printf_s("GPUJob execution: oridinal - %ld;\n", _ordinal );

	if(_pStitched !=NULL && _rect.IsValid())
		results = _pStitched->GPUMorphFrom(_pFOV, _rect, jobStream);

	return results;
}
