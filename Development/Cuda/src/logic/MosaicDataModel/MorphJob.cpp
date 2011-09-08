#include "StdAfx.h"
#include "MorphJob.h"
#include "Image.h"

bool CudaBufferRegister(unsigned char *ptr, size_t size);
bool CudaBufferUnregister(unsigned char *ptr);

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

	CudaBufferRegister(_pFOV->GetBuffer(), _pFOV->Columns()*_pFOV->Rows());
}

MorphJob::~MorphJob()
{
	CudaBufferUnregister(_pFOV->GetBuffer());
}

void MorphJob::Run()
{
	//printf_s("Thread execution: oridinal - %ld;\n", _ordinal);

	if(_pStitched !=NULL && _rect.IsValid())
		_pStitched->MorphFrom(_pFOV, _rect);
}

void ImageMorphCudaDelete(CyberJob::GPUJobStream *jobStream);
void ClearMorphJobStream(CyberJob::GPUJobStream *jobStream)
{
	ImageMorphCudaDelete(jobStream);
}

CyberJob::GPUJob::GPUJobStatus MorphJob::GPURun(CyberJob::GPUJobStream *jobStream)
{
	CyberJob::GPUJob::GPUJobStatus results = CyberJob::GPUJob::GPUJobStatus::COMPLETED; // true = conversion complete
	
	//printf_s("GPUJob execution: oridinal - %ld;\n", _ordinal );

	if(_pStitched !=NULL && _rect.IsValid())
		results = _pStitched->GPUMorphFrom(_pFOV, _rect, jobStream);

	return results;
}
