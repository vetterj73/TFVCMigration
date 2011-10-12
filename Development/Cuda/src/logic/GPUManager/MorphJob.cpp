//#include "StdAfx.h"
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

	CyberGPU::CudaBufferRegister(_pFOV->GetBuffer(), _pFOV->Columns()*_pFOV->Rows());
}

MorphJob::~MorphJob()
{
	CyberGPU::CudaBufferUnregister(_pFOV->GetBuffer());
}

void MorphJob::Run()
{
	//printf_s("Thread execution: oridinal - %ld;\n", _ordinal);

	if(_pStitched !=NULL && _rect.IsValid())
		_pStitched->MorphFrom(_pFOV, _rect);
}

CyberGPU::CGPUJob::GPUJobStatus MorphJob::GPURun(CyberGPU::GPUStream *jobStream)
{
	CyberGPU::CGPUJob::GPUJobStatus results = CyberGPU::CGPUJob::GPUJobStatus::COMPLETED; // true = conversion complete
	
	//printf_s("GPUJob execution: oridinal - %ld;\n", _ordinal );
	_pStitched->GetTransform();

	if(_pStitched !=NULL && _rect.IsValid())
		results = GPUMorphFrom(_pStitched, _pFOV, _rect, jobStream);

	return results;
}

// This image's ROI content is mapped from pImgIn
CyberGPU::CGPUJob::GPUJobStatus MorphJob::GPUMorphFrom(const Image* pStitched, const Image* pImgIn, UIRect roi, CyberGPU::GPUStream *jobStream)
{
	CyberGPU::CGPUJob::GPUJobStatus results = CyberGPU::CGPUJob::GPUJobStatus::COMPLETED; // true = conversion complete
	/*

	[x]			[Row_in]		[Row_out]
	[y] ~= TIn*	[Col_in] ~=Tout*[Col_out]
	[1]			[1     ]		[	   1]

	[Row_in]					[Row_out]	[A00 A01 A02]	[Row_out]
	[Col_in] ~= Inv(TIn)*TOut*	[Col_out] =	[A10 A11 A12] *	[Col_out]
	[1	   ]					[1		]	[A20 A21 1	]	[1		]

	[Col_in]	[A11 A10 A12]	[Col_out]
	[Row_in] ~=	[A01 A00 A02] *	[Row_out]
	[1	   ]	[A21 A20 1	]	[1		]
	*/

	// Validation check (only for 8-bit image)
	if(pStitched->GetBytesPerPixel()/*_bytesPerPixel*/ != 1) return(CyberGPU::CGPUJob::GPUJobStatus::COMPLETED);
	
	// Create tansform matrix from (Col_out, Row_out) to (Col_in, Row_in)
	ImgTransform tIn_inv = pImgIn->GetTransform().Inverse();
	ImgTransform t = tIn_inv * pStitched->GetTransform(); //_thisToWorld;
	double dTemp[3][3];
	t.GetMatrix(dTemp);
	double dT[3][3];

	dT[0][0] = dTemp[1][1];
	dT[0][1] = dTemp[1][0];
	dT[0][2] = dTemp[1][2];
	
	dT[1][0] = dTemp[0][1];
	dT[1][1] = dTemp[0][0];
	dT[1][2] = dTemp[0][2];

	dT[2][0] = dTemp[2][1];
	dT[2][1] = dTemp[2][0];
	dT[2][2] = dTemp[2][2];

	// Sanity check
	if((roi.FirstColumn+roi.Columns()>pStitched->PixelRowStride()/*_pixelRowStride*/) || (pImgIn->Columns()>pImgIn->PixelRowStride())
		|| (pImgIn->Columns() < 2) || (pImgIn->Rows() < 2)
		|| (roi.Columns() <= 0) || (roi.Rows() <= 0))
		return(CyberGPU::CGPUJob::GPUJobStatus::COMPLETED);

	// !!! GPU can currently only do affine transform
	if(dT[2][0] != 0 || dT[2][1] != 0 || dT[2][2] != 1) return CyberGPU::CGPUJob::GPUJobStatus::COMPLETED; // true means done

	//startTick = clock();//Obtain current tick

	// GPU based image morph
	results = GPUImageMorph(jobStream,
		pImgIn->GetBuffer(), pImgIn->PixelRowStride(),
		pImgIn->Columns(), pImgIn->Rows(), 
		pStitched->GetBuffer()/*_buffer*/, pStitched->PixelRowStride()/*_pixelRowStride*/,
		roi.FirstColumn, roi.FirstRow,
		roi.Columns(), roi.Rows(),
		dT);

	//deltaTicks += clock() - startTick;//calculate the difference in ticks

	//if (ImageMorph_loop == 189)
	//{
	//	printf_s("ImageMorph %d; ticks - %ld\n", ImageMorph_loop, deltaTicks);
	//	PrintTicks();
	//}

	//ImageMorph_loop += 1;

	return(results);
}
