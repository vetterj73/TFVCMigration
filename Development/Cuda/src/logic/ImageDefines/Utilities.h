/*
	The collection of utlilty functions
*/

#pragma once

#include "GPUManager.h"

// Inverse a matrix,
// inMatrix: input matrix, data stored row by row
// outMatrix: output Matrix, data stored row by row
// rows and cols: size of matrix 
void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols);


bool CudaBufferRegister(unsigned char *ptr, size_t size);
bool CudaBufferUnregister(unsigned char *ptr);

CyberJob::CGPUJob::GPUJobStatus GPUImageMorph(CyberJob::GPUStream *jobStream, unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]);

//void ImageMorphCudaDelete( CyberJob::GPUStream *jobStream);

// Fill a ROI of the output image by transforming the input image
// Both output image and input image are 8bits/pixel (can add 16bits/pixel support easily)
// pInBuf, iInSpan, iInWidth and iInHeight: input buffer and its span, width and height
// pOutBuf and iOutspan : output buffer and its span
// iROIWidth, iHeight: the size of buffer need to be transformed
// iOutROIStartX, iOutROIStartY, iOutROIWidth and iOutROIHeight: the ROI of the output image
// dTrans: the 3*3 transform from [Col_out Row_out] to [Col_in, Row_in]
bool ImageMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]);


// Modified from Eric Rudd's BayerLum() function
// Convert Bayer image into Luminance
// Output data only valid int the range of columns [2, nCols-3] and rows [2 nRows-3]
void BayerToLum(                
   int            ncols,		// Image dimensions
   int            nrows,
   unsigned char  bayer[],      // Input 8-bit Bayer image 
   int            bstride,      // Addressed as bayer[col + row*bstride] 
   unsigned char  lum[],        // output Luminance image 
   int            lstride);      // Addressed as out[col + row*ostride] 

// 
//	This will give the number of pixels in a common way...
//
int GetNumPixels(double size, double pixelSize);


// 2D Morphological process (a Warp up of Rudd's morpho2D) 
void Morpho_2d(
	unsigned char* pbBuf,
	unsigned int iSpan,
	unsigned int iXStart,
	unsigned int iYStart,
	unsigned int iBlockWidth,
	unsigned int iBlockHeight,
	unsigned int iKernelWidth, 
	unsigned int iKernelHeight, 
	int iType);