/*
	The collection of utlilty functions
*/

#pragma once

// Inverse a matrix,
// inMatrix: input matrix, data stored row by row
// outMatrix: output Matrix, data stored row by row
// rows and cols: size of matrix 
void inverse3x3(	
	const double* inMatrix,
	double* outMatrix);
void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols);

// Fill a ROI of the output image with a height map by transforming the input image if heigh map exists
// Support convert YCrCb seperate channel to BGR combined channels, or grayscale (one channel) only
// Assume the center of image corresponding a vertical line from camera to object surface
// Both output image and input image are 8bits/pixel (can add 16bits/pixel support easily)
// pInBuf, iInSpan, iInWidth and iInHeight: input buffer and its span, width and height
// pOutBuf and iOutspan : output buffer and its span
// iROIWidth, iHeight: the size of buffer need to be transformed
// iOutROIStartX, iOutROIStartY, iOutROIWidth and iOutROIHeight: the ROI of the output image
// dInvTrans: the 3*3 transform from output image to input image
// iNumChannels: number of image channels, must be 1 or 3
// pHeightImage and iHeightSpan, height image buffer and its span, if it is NULL, don't adjust for height map
// dHeightResolution: the height represented by each grey level
// dPupilDistance: camera pupil distance
// dPerpendicalPixelX and dPerpendicalPixelY, the pixel corresponding to the point in the panel surface 
// that its connection with camera center is vertical to panel surface
bool ImageMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3], unsigned int iNumChannels,
	unsigned char* pHeightImage=0, unsigned int iHeightSpan=0,
	double dHeightResolution=0, double dPupilDistance=0,
	double dPerpendicalPixelX=0, double dPerpendicalPixelY=0);

// Fast version of morph for grayscale image and use Nearest neightborhood
bool ImageGrayNNMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]);

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

// pData1 = clip(pData1-pData2)
template<typename T>
void ClipSub(
	T* pData1, unsigned int iSpan1, 
	T* pData2, unsigned int iSpan2,
	unsigned int iWidth, unsigned int iHeight);

enum COLORSTYLE
{
	YCrCb,
	RGB,
	BGR,
	YONLY,
};

enum BayerType 
{                           // Don't change order of enums
   BGGR,
   GBRG,
   GRBG,
   RGGB
};

void BayerLum(						// Bayer interpolation 
   int            ncols,			// Image dimensions 
   int            nrows,
   unsigned char  bayer[],			// Input 8-bit Bayer image
   int            bstride,			// Addressed as bayer[col + row*bstride]  
   BayerType      order,			// Bayer pattern order; use the enums in bayer.h
   unsigned char  out[],			// In/Out 24-bit BGR/YCrCb image, allocated outside and filled inside of function
   int            ostride,			// Addressed as out[col + row*ostride]
   COLORSTYLE     type,				// Type of color BGR/YCrCb/Y
   bool			  bChannelSeperate);	// true, the channel stored seperated