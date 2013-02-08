#include "Utilities.h"
#include "lsqrpoly.h" 
#include "math.h"
#include "morpho.h"
#include <string.h>


// Inverse 3x3 matrix
void inverse3x3(	
	const double* inMatrix,
	double* outMatrix)
{
	// hard coded inverse for a 3x3 array (fed in as a [9] vector)
	// non-working starting point found on website 'stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c'
	// in case of a singular input matrix return a singular matrix (map all space to pixel 0,0)
	double determinant =    +inMatrix[0]*(inMatrix[4]*inMatrix[8]-inMatrix[7]*inMatrix[5])
                        -inMatrix[1]*(inMatrix[3]*inMatrix[8]-inMatrix[5]*inMatrix[6])
                        +inMatrix[2]*(inMatrix[3]*inMatrix[7]-inMatrix[4]*inMatrix[6]);
	if (abs(determinant) < 1e-16) // test will be OK as long as pixel size > 0.01 um
	{
		for (unsigned int i(0); i < 8; i++)
			outMatrix[i] = 0;
		outMatrix[8] = 1.;
	}
	else
	{
		double invdet = 1/determinant;
		outMatrix[0] =  (inMatrix[4]*inMatrix[8]-inMatrix[7]*inMatrix[5])*invdet;
		outMatrix[1] = -(inMatrix[1]*inMatrix[8]-inMatrix[2]*inMatrix[7])*invdet;
		outMatrix[2] =  (inMatrix[1]*inMatrix[5]-inMatrix[2]*inMatrix[4])*invdet;
		outMatrix[3] = -(inMatrix[3]*inMatrix[8]-inMatrix[5]*inMatrix[6])*invdet;
		outMatrix[4] =  (inMatrix[0]*inMatrix[8]-inMatrix[2]*inMatrix[6])*invdet;
		outMatrix[5] = -(inMatrix[0]*inMatrix[5]-inMatrix[3]*inMatrix[2])*invdet;
		outMatrix[6] =  (inMatrix[3]*inMatrix[7]-inMatrix[6]*inMatrix[4])*invdet;
		outMatrix[7] = -(inMatrix[0]*inMatrix[7]-inMatrix[6]*inMatrix[1])*invdet;
		outMatrix[8] =  (inMatrix[0]*inMatrix[4]-inMatrix[3]*inMatrix[1])*invdet;
	}
}

// Solve the least square problem AX = b
// iRows and iCols: size of matrix A
// X: output, the least square results
// resid: output, residual
void LstSqFit(
	const double *A, unsigned int nRows, unsigned int nCols, 
	const double *b, double *X, double *resid)
{
	// a very generic least square solver wrapper
	// takes input arrays in 'C' format and does the requried steps to call Eric's qr tools
	// note that b is NOT overwritten (unlike a raw call to qr)
	// resid is a vector of b-Ax values
	double		*workspace;
	double 		*bCopy;
	bCopy = new double[nRows];
	int SizeofFidFitA = nCols * nRows;
	workspace = new double[SizeofFidFitA];
	
	// transpose for Eric's code
	for(unsigned int row(0); row<nRows; ++row)
	{
		bCopy[row] = b[row];
		for(unsigned int col(0); col<nCols; ++col)
			workspace[col*nRows+row] = A[row*nCols+col];
	}
	// solve copied from SolveX()
	double*	sigma = new double[nCols];

	// perform factorization of system A = QR
	int qrdRet = 
		qrd(
				workspace,    /* System matrix, m-by-n */
				sigma,      /* Diagonals of R (caller reserves n elements) */
				nRows,      /* Number of rows in system matrix */
				nRows,		/* Spacing between columns in system matrix */
				nCols ); /* Number of columns in system matrix */

							/* Return values

								 0 - Normal completion.
								 1 - Matrix was of incompatible dimensions.
								 2 - Singular system matrix. */
	//LOG.FireLogEntry(LogTypeSystem, "qrdRet %d", qrdRet);
	//for (unsigned int row(0); row < nCols; row++)
	//		LOG.FireLogEntry(LogTypeSystem, "sigma %d, %.3e",row, sigma[row]);
   double condition =
	   rcond(               /* Reciprocal condition number estimator */
			   nCols,    /* Number of unknowns */
			   workspace,     /* QR factorization returned from qrd() */
			   nRows,       /* Spacing between columns of qr[] */
			   sigma        /* R diagonals from qrd() */
			);
   //LOG.FireLogEntry(LogTypeSystem, "condition %f", condition);

	qrsolv (
				workspace,	/* Factored system matrix, from qrd() */
				sigma,		/* Diagonal of R, from qrd() */
				&bCopy[0],	/* Constant vector, overwritten by solution */
				nRows,		/* Number of rows in system matrix */
				nRows,		/* Spacing between columns in system matrix */
				nCols,	/* Number of columns in system matrix */
				1     );	/*	0 - premultiply  b  by Q
								1 - premultiply  b  by inv(QR)
								2 - premultiply  b  by QR
								3 - premultiply  b  by inv(R^T R)
								4 - premultiply  b  by Q inv(R^T)
								5 - premultiply  b  by R
								6 - premultiply  b  by inv(R)

								 qrsolv() always returns 0 */

	for (unsigned int j(0); j<nCols; ++j)
		X[j] = bCopy[j];
	
	for(unsigned int row(0); row<nRows; ++row)
	{
		resid[row] = b[row] ;
		for (unsigned int col(0); col<nCols; ++col)
		  resid[row] -=  X[col]*A[row*nCols+col];
	}
	delete [] sigma;
	delete [] workspace;
	delete [] bCopy;
}

// Fill a ROI of the output image with a height map by transforming the input image if heigh map exists
// Support convert YCrCb/BGR seperate channel to BGR combined channels, or grayscale (one channel) only
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
	bool bIsYCrCb,
	unsigned char* pHeightImage, unsigned int iHeightSpan,
	double dHeightResolution, double dPupilDistance,
	double dPerpendicalPixelX, double dPerpendicalPixelY) 
{
	// Sanity check
	if( ((iNumChannels!=3) && (iNumChannels!=1)) ||
		(iOutROIStartX+iOutROIWidth>iOutSpan/iNumChannels) || (iInWidth>iInSpan)
		|| (iInWidth <2) || (iInHeight<2)
		|| (iOutROIWidth<=0) || (iOutROIHeight<=0))
		return(false);

	bool bAdjustForHeight = !(pHeightImage==NULL);

	// If pupil distance less than 0.1mm, return false
	if(bAdjustForHeight &&  dPupilDistance < 1e-4)
		return(false);
			
	unsigned int iY, iX;

	// Whether it is an affine transform
	bool bAffine;
	if(dInvTrans[2][0] == 0 &&
		dInvTrans[2][1] == 0 &&
		dInvTrans[2][2] == 1)
		bAffine = true;
	else
		bAffine = false;

	unsigned char* pOutLine = pOutBuf + iOutROIStartY*iOutSpan;
	int iInSpanP1 = iInSpan+1;

	// some local variable
	unsigned char *pInCh[3];
	for(unsigned int i=0; i<iNumChannels; i++)
	{
		pInCh[i] = pInBuf + i*iInSpan*iInHeight;
	}

	int iflrdX, iflrdY;
	int iPix0, iPix1, iPixW, iPixWP1;
	int iPixDiff10;
	double dT01__dIY_T02, dT11__dIY_T12, dT21__dIY_T22;
	double dIX, dIY, dX, dY, dDiffX, dDiffY;
	double dVal;
 
	double dDividedPupilDistrance=0;
	if(bAdjustForHeight) 
		dDividedPupilDistrance = 1.0/dPupilDistance;
	if(bAffine)
	{
		for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
		{
			dIY = (double) iY;
			dT01__dIY_T02 = dInvTrans[0][1] * dIY + dInvTrans[0][2];
			dT11__dIY_T12 = dInvTrans[1][1] * dIY + dInvTrans[1][2];

			for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
			{
				dIX = (double) iX;
				dX = dInvTrans[0][0]*dIX + dT01__dIY_T02;
				dY = dInvTrans[1][0]*dIX + dT11__dIY_T12;

				// Adjust for height
				if(bAdjustForHeight)
				{
					int iHeightInPixel = pHeightImage[iY*iHeightSpan+iX];
					if(iHeightInPixel >0)
					{
						double dHeight = iHeightInPixel *dHeightResolution;		// Height in physical unit
						//double dRatio = dHeight/(dPupilDistance-dHeight);		// Sacrifice some speed for performance
						double dRatio = dHeight*dDividedPupilDistrance;			// Sacrifice some performance for speed
						double dOffsetX = (dX-dPerpendicalPixelX)*dRatio;
						double dOffsetY = (dY-dPerpendicalPixelY)*dRatio;
						dX += dOffsetX;
						dY += dOffsetY;
					}
				}

			  /* Check if back projection is outside of the input image range. Note that
				 2x2 interpolation touches the pixel to the right and below right,
				 so right and bottom checks are pulled in a pixel. */
				if ((dX < 0) | (dY < 0) |
				  (dX >= iInWidth-1) | (dY >= iInHeight-1)) 
				{
					if(iNumChannels==1)
						pOutLine[iX] = 0x00;	/* Clipped */
					else
					{
						pOutLine[iX*3] = 0x00;	/* Clipped */
						pOutLine[iX*3+1] = 128;	/* Clipped */
						pOutLine[iX*3+2] = 128;	/* Clipped */
					}
				}
				else 
				{

					for(unsigned int i=0; i<iNumChannels; i++)
					{
						/* Compute fractional differences */
						iflrdX =(int)dX;
						iflrdY = (int)dY;
			    
						unsigned char* pbPixPtr = pInCh[i] + iflrdX + iflrdY * iInSpan;

						iPix0   = (int) pbPixPtr[0];
						iPix1   = (int) pbPixPtr[1];
						iPixW   = (int) pbPixPtr[iInSpan];
						iPixWP1 = (int) pbPixPtr[iInSpanP1];

						iPixDiff10 = iPix1 - iPix0;

						dDiffX = dX - (double) iflrdX;
						dDiffY = dY - (double) iflrdY;

						pOutLine[iX*iNumChannels+i] = (unsigned char)((double) iPix0			
							+ dDiffY * (double) (iPixW - iPix0)	
							+ dDiffX * ((double) iPixDiff10	
							+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
							+ 0.5);
					}
				}
			} //ix
			pOutLine += iOutSpan;		/* Next line in the output buffer */
		} // iy
	}
	else 
	{	/* Perspective transform: Almost identical to the above. */
		for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
		{
			dIY = (double) iY;
			dT01__dIY_T02 = dInvTrans[0][1] * dIY + dInvTrans[0][2];
			dT11__dIY_T12 = dInvTrans[1][1] * dIY + dInvTrans[1][2];
			dT21__dIY_T22 = dInvTrans[2][1] * dIY + dInvTrans[2][2];

			for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
			{
				dIX = (double) iX;
				dVal = 1.0 / (dInvTrans[2][0]*dIX + dT21__dIY_T22);
				dX = (dInvTrans[0][0]*dIX + dT01__dIY_T02) * dVal;
				dY = (dInvTrans[1][0]*dIX + dT11__dIY_T12) * dVal;

				// Adjust for height
				if(bAdjustForHeight)
				{
					int iHeightInPixel = pHeightImage[iY*iHeightSpan+iX];
					if(iHeightInPixel >0)
					{
						double dHeight = iHeightInPixel *dHeightResolution;		// Height in physical unit
						//double dRatio = dHeight/(dPupilDistance-dHeight);		// Sacrifice some speed for performance
						double dRatio = dHeight*dDividedPupilDistrance;			// Sacrifice some performance for speed
						double dOffsetX = (dX-dPerpendicalPixelX)*dRatio;
						double dOffsetY = (dY-dPerpendicalPixelY)*dRatio;
						dX += dOffsetX;
						dY += dOffsetY;
					}
				}

				if ((dX < 0) | (dY < 0) |
					(dX >= iInWidth-1) | (dY >= iInHeight-1)) 
				{
					if(iNumChannels==1)
						pOutLine[iX] = 0x00;	/* Clipped */
					else
					{
						pOutLine[iX*3] = 0x00;	/* Clipped */
						pOutLine[iX*3+1] = 128;	/* Clipped */
						pOutLine[iX*3+2] = 128;	/* Clipped */
					}
				}
				else 
				{
					for(unsigned int i=0; i<iNumChannels; i++)
					{
							/* Compute fractional differences */
						iflrdX =(int)dX;
						iflrdY = (int)dY;
			    
						unsigned char* pbPixPtr = pInCh[i] + iflrdX + iflrdY * iInSpan;

						iPix0   = (int) pbPixPtr[0];
						iPix1   = (int) pbPixPtr[1];
						iPixW   = (int) pbPixPtr[iInSpan];
						iPixWP1 = (int) pbPixPtr[iInSpanP1];

						iPixDiff10 = iPix1 - iPix0;

						dDiffX = dX - (double) iflrdX;
						dDiffY = dY - (double) iflrdY;

						pOutLine[iX*iNumChannels+i] = (unsigned char)((double) iPix0			
							+ dDiffY * (double) (iPixW - iPix0)	
							+ dDiffX * ((double) iPixDiff10	
							+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
							+ 0.5);
					}
				}
			} // ix
			pOutLine += iOutSpan;
		} // iy
    } // else

	// YCrCb to BGR
	if(bIsYCrCb && (iNumChannels==3))
	{
		pOutLine = pOutBuf + iOutROIStartY*iOutSpan;
		for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
		{
			for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX)
			{
				// YCrCb to RGB conversion
				int iTemp[3];
				iTemp[2] = pOutLine[iX*3] + (pOutLine[iX*3+1]-128);								// R
				iTemp[1] = pOutLine[iX*3] - (pOutLine[iX*3+1]-128 + pOutLine[iX*3+2]-128)/2;	// G
				iTemp[0] = pOutLine[iX*3] + (pOutLine[iX*3+2]-128);								// B

				for(int i=0; i<3; i++)
				{
					if(iTemp[i]<0) iTemp[i]=0;
					if(iTemp[i]>255) iTemp[i]=255;
					pOutLine[iX*3+i] = iTemp[i];
				}
			}
			pOutLine += iOutSpan;		// Next line in the output buffer
		}
	}

	return(true);
}


// Fast version of morph for grayscale image and use Nearest neightborhood
bool ImageGrayNNMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3])
{
	// Sanity check
	if( iOutROIStartX+iOutROIWidth > iOutSpan || iInWidth > iInSpan
		|| iInWidth < 2 || iInHeight < 2
		|| iOutROIWidth <= 0 || iOutROIHeight <= 0)
		return(false);

	unsigned int iY, iX;

	// Whether it is an affine transform
	bool bAffine;
	if(dInvTrans[2][0] == 0 &&
		dInvTrans[2][1] == 0 &&
		dInvTrans[2][2] == 1)
		bAffine = true;
	else
		bAffine = false;

	unsigned char* pOutLine = pOutBuf + iOutROIStartY*iOutSpan;
	int iInSpanP1 = iInSpan+1;

	int iflrdX, iflrdY;
	double dT01__dIY_T02, dT11__dIY_T12, dT21__dIY_T22;
	double dIX, dIY, dX, dY, dVal;
	 
	double dDividedPupilDistrance=0;
	if(bAffine)
	{
		for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
		{
			dIY = (double) iY;
			dT01__dIY_T02 = dInvTrans[0][1] * dIY + dInvTrans[0][2];
			dT11__dIY_T12 = dInvTrans[1][1] * dIY + dInvTrans[1][2];

			for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
			{
				dIX = (double) iX;
				dX = dInvTrans[0][0]*dIX + dT01__dIY_T02;
				dY = dInvTrans[1][0]*dIX + dT11__dIY_T12;

			  /* Check if back projection is outside of the input image range. Note that
				 2x2 interpolation touches the pixel to the right and below right,
				 so right and bottom checks are pulled in a pixel. */
				if ((dX < 0) || (dY < 0) ||
				  (dX >= iInWidth-1) || (dY >= iInHeight-1)) 
				{
					pOutLine[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					iflrdX =(int)dX;
					iflrdY = (int)dY;
			    
					unsigned char* pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;
					pOutLine[iX] = *pbPixPtr;
				}
			} //ix
			pOutLine += iOutSpan;		/* Next line in the output buffer */
		} // iy
	}
	else 
	{	/* Perspective transform: Almost identical to the above. */
		for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
		{
			dIY = (double) iY;
			dT01__dIY_T02 = dInvTrans[0][1] * dIY + dInvTrans[0][2];
			dT11__dIY_T12 = dInvTrans[1][1] * dIY + dInvTrans[1][2];
			dT21__dIY_T22 = dInvTrans[2][1] * dIY + dInvTrans[2][2];

			for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
			{
				dIX = (double) iX;
				dVal = 1.0 / (dInvTrans[2][0]*dIX + dT21__dIY_T22);
				dX = (dInvTrans[0][0]*dIX + dT01__dIY_T02) * dVal;
				dY = (dInvTrans[1][0]*dIX + dT11__dIY_T12) * dVal;

				if ((dX < 0) | (dY < 0) |
					(dX >= iInWidth-1) | (dY >= iInHeight-1)) 
				{
					pOutLine[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					/* Compute fractional differences */
					iflrdX =(int)dX;
					iflrdY = (int)dY;
			    
					unsigned char* pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;
					pOutLine[iX] = *pbPixPtr;
				}
			} // ix
			pOutLine += iOutSpan;
		} // iy
    } // else

	return(true);
}

  
int GetNumPixels(double sizeInMeters, double pixelSize)
{
	return(int)floor((sizeInMeters/pixelSize)+0.5);
}

// 2D Morphological process (a Warp up of Rudd's morpho2D)
// Dilation, erosion and so on
void Morpho_2d(
	unsigned char* pbBuf,
	unsigned int iSpan,
	unsigned int iXStart,
	unsigned int iYStart,
	unsigned int iBlockWidth,
	unsigned int iBlockHeight,
	unsigned int iKernelWidth, 
	unsigned int iKernelHeight, 
	int iType)
{
	int CACHE_REPS = CACHE_LINE/sizeof(unsigned char);
	int MORPHOBUF1 = 2+CACHE_REPS;

	// Morphological process
	unsigned char *pbWork;
	int iMem  = (2 * iBlockWidth) > ((int)(MORPHOBUF1)*iBlockHeight) ? (2 * iBlockWidth) : ((int)(MORPHOBUF1)*iBlockHeight); 
	pbWork = new unsigned char[iMem];

	morpho2d(iBlockWidth, iBlockHeight, pbBuf+ iYStart*iSpan + iXStart, iSpan, pbWork, iKernelWidth, iKernelHeight, iType);

	delete [] pbWork;
}

// pData1 = clip(pData1-pData2)
template<typename T>
void ClipSub(
	T* pData1, unsigned int iSpan1, 
	T* pData2, unsigned int iSpan2,
	unsigned int iWidth, unsigned int iHeight)
{
	T* pLine1 = pData1;
	T* pLine2 = pData2;
	for(unsigned int iy=0; iy<iHeight; iy++)
	{
		for(unsigned int ix=0; ix<iWidth; ix++)
		{
			int iTemp = (int)pLine1[ix] - (int)pLine2[ix];
			if(iTemp<0) iTemp = 0;
			pLine1[ix] = (T)iTemp;
		}
		pLine1 += iSpan1;
		pLine2 += iSpan2;
	}
}

// Create Image and Image16 instances
template void ClipSub(
	unsigned char* pData1, unsigned int iSpan1, 
	unsigned char* pData2, unsigned int iSpan2,
	unsigned int iWidth, unsigned int iHeight);
template void ClipSub(
	unsigned short* pData1, unsigned int iSpan1, 
	unsigned short* pData2, unsigned int iSpan2,
	unsigned int iWidth, unsigned int iHeight);


static INLINE int clip(int value) {
   if (value <   0) value = 0;
   if (value > 255) value = 255;
   return value;
}

// Modified from Rudd's BayerLum()
void BayerLum(						// Bayer interpolation 
   int            ncols,			// Image dimensions 
   int            nrows,
   unsigned char  bayer[],			// Input 8-bit Bayer image
   int            bstride,			// Addressed as bayer[col + row*bstride]  
   BayerType      order,			// Bayer pattern order; use the enums in bayer.h
   unsigned char  out[],			// In/Out 24-bit BGR/YCrCb or 8-bit Y(luminance) image, 
									// allocated outside and filled inside of function
   int            ostride,			// Addressed as out[col*NumOfChannel+ChannelIndex + row*ostride] if channels are combined
									// or out[col + row*ostride + (ChannelIndex-1)*ostride*nrows] if channels are seperated
   COLORSTYLE     type,				// Type of color BGR/YCrCb/Y
   bool			  bChannelSeperate)	// true, the channel stored seperated

   /* BayerLum() performs Bayer interpolation by linear filtering, as
   described in

      Eric P. Rudd and Swaminathan Manickam, "A Luminance-Based Bayer
      Interpolation Filter"
      http://doc.cyberoptics.com/Research/ruddpapers/lumeval.pdf

   The philosophy behind this filter is that luminance (Y) is the most
   important and that chrominance errors are less visible.  Accordingly, a
   5x5 kernel is used to reconstruct Y, and 3x3 (bilinear) filters are used to
   compute R-Y and B-Y.  The outputs of these three filters are then combined.
   This approach is faster than using a large filter kernel for all three
   colors.

   Most of the parameters are obvious, but "order" deserves a little more
   explanation.  Assuming that bayer[] begins at the southwest (SW) corner, the
   Bayer pattern at that corner can assume four different configurations:

      G R   (BGGR)
      B G

      R G   (GBRG)
      G B

      B G   (GRBG)
      G R

      G B   (RGGB)
      R G

   Use the enums defined in bayer.h to specify which order to use. */
{
	int b, by, col, colinc, g, r, row, ry, y;
	int ColOrder, RowOrder;

   /* Housekeeping */

	ColOrder = order & 1;               /* Bit 0 */
	RowOrder = (order & 2) >> 1;        /* Bit 1 */

   /* Compute luminance with maximally-flat filter having zeros at
   (+/-f, +/-f) and (0, +/-f) and (+/-f, 0).  Compute r-y and b-y from
   RGB values with bilinear filter.  Finally combine the results.
   */

	unsigned char *bptr = bayer + 2*bstride;	// For input bayer image
	unsigned char *optr = out + 2*ostride;		// For output image with combined channels
	unsigned char *c0Ptr = out + 2*ostride;		// For output image with seperated channels
	unsigned char *c1Ptr = c0Ptr + nrows*ostride;
	unsigned char *c2Ptr = c1Ptr + nrows*ostride;
	for (row=2; row<nrows-2; row++) 
	{
		for (col=2; col<ncols-2; col++) 
		{
			y =
				+ 156*(
                   +bptr[col                ]
                 )

				+ 30*(
                   +bptr[col     + bstride]
                   +bptr[col     - bstride]
                   +bptr[col - 1            ]
                   +bptr[col + 1            ]
                )

				- 20*(
                   +bptr[col     + 2*bstride]
                   +bptr[col     - 2*bstride]
                   +bptr[col + 2            ]
                   +bptr[col - 2            ]
                )

				+ 16*(
                   +bptr[col - 1 + bstride]
                   +bptr[col + 1 + bstride]
                   +bptr[col - 1 - bstride]
                   +bptr[col + 1 - bstride]
                )

				+    (
                   +bptr[col - 1 + 2*bstride]
                   +bptr[col + 1 + 2*bstride]
                   +bptr[col - 2 + bstride]
                   +bptr[col + 2 + bstride]
                   +bptr[col - 2 - bstride]
                   +bptr[col + 2 - bstride]
                   +bptr[col - 1 - 2*bstride]
                   +bptr[col + 1 - 2*bstride]
                )

				-  3*(
                   +bptr[col - 2 + 2*bstride]
                   +bptr[col + 2 + 2*bstride]
                   +bptr[col - 2 - 2*bstride]
                   +bptr[col + 2 - 2*bstride]
                );

			// Only collect luminance
			if(type == YONLY)
			{
				optr[col] = clip(y/256);
				continue;
			}

			if ((row&1) == RowOrder) 
			{
				if ((col&1) == ColOrder) 
				{	/* Blue */
					b = 64*bptr[col];

					g = 16*(
						+bptr[col  -bstride]
						+bptr[col-1     ]
						+bptr[col+1     ]
						+bptr[col  +1*bstride]
									 );

					r = 16*(
						+bptr[col-1-bstride]
						+bptr[col+1-bstride]
						+bptr[col-1+bstride]
						+bptr[col+1+bstride]
									 );
				} 
				else 
				{                   /* Green in blue  row */
					b = 32*(
						+bptr[col-1     ]
						+bptr[col+1     ]
									 );

					g = 8*(
					   +bptr[col-1-bstride]
					   +bptr[col+1-bstride]
					   +4*bptr[col]
					   +bptr[col-1+bstride]
					   +bptr[col+1+bstride]
									 );

					r = 32*(
					   +bptr[col  -bstride]
					   +bptr[col  +bstride]
								 );
				}
			}
			else 
			{
				if ((col&1) == ColOrder) 
				{ /* Green in red row */
					b = 32*(
						+bptr[col  -bstride]
						+bptr[col  +bstride]
								 );

					g = 8*(
						+bptr[col-1-bstride]
						+bptr[col+1-bstride]
						+4*bptr[col]
						+bptr[col-1+bstride]
						+bptr[col+1+bstride]
								 );

					r = 32*(
						+bptr[col-1     ]
						+bptr[col+1     ]
								 );
				} 
				else 
				{   /* Red */
					b = 16*(
						+bptr[col-1-bstride]
						+bptr[col+1-bstride]
						+bptr[col-1+bstride]
						+bptr[col+1+bstride]
								 );

					g = 16*(
						+bptr[col  -bstride]
						+bptr[col-1     ]
						+bptr[col+1     ]
						+bptr[col  +bstride]
								 );

					r = 64*bptr[col];
				}
			}

         /* At this point, r, g, b = 64 times desired value and
            y = 256 times desired value. */

			ry = 3*r - 2*g -   b;		// 256 time of desired value
			by =  -r - 2*g + 3*b;

			switch(type)
			{
			/* 
				Cr = 3/4*R - 1/2*G - 1/4*B
				Cb = -1/4*R -1/2*g + 3/4*B
				When R, G, B in range of [0, 255]
				Cr and Cb in range of [-255*3/4, 255*3/4] = [-191, 191], Which is out or range of 2^8 [-128, 127]
				If scale Cr and Cb down by 2, RGB image converted from YCrCb will lose some resolution in intensity.
				For circuit board, the chance of Cr and Cb out of [-128, 127] is rare
				Therefore, in order to save storage, Cr and Cb is clip into [-128, 127]+128 = [0, 255]
			*/
			case YCrCb:
				if(bChannelSeperate)
				{
					c0Ptr[col] = clip(y/256);
					c1Ptr[col] = clip(ry/256+128);
					c2Ptr[col] = clip(by/256+128);
					/* For debug
					int iCr = ry/256;
					int iCb = by/256;
					if(iCr>127 || iCr<-128 || iCb>127 || iCb<-128)
					{
						int ihh=10;
					}//*/
				}
				else
				{
					optr[col*3] = clip(y/256);
					optr[col*3+1] = clip(ry/256+128);
					optr[col*3+2] = clip(by/256+128);
				}
				break;

			case RGB:
				if(bChannelSeperate)
				{
					c0Ptr[col] = clip((y + ry)/256);			// R
					c1Ptr[col] = clip((2*y - ry - by)/512);		// G
					c2Ptr[col] = clip((y + by)/256);			// B
				}
				else
				{

					optr[col*3] = clip((y + ry)/256);			// R
					optr[col*3+1]= clip((2*y - ry - by)/512);	// G
					optr[col*3+2] = clip((y + by)/256);			// B
				}
				break;

			case BGR:
				if(bChannelSeperate)
				{
					c0Ptr[col] = clip((y + by)/256);			// B
					c1Ptr[col] = clip((2*y - ry - by)/512);		// G
					c2Ptr[col] = clip((y + ry)/256);			// R
				}
				else
				{
					optr[col*3] = clip((y + by)/256);			// B
					optr[col*3+1]= clip((2*y - ry - by)/512);	// G
					optr[col*3+2] = clip((y + ry)/256);			// R
				}
			}
		}
		bptr += bstride;
		optr += ostride;
		c0Ptr+= ostride;
		c1Ptr+= ostride;
		c2Ptr+= ostride;
	}

   /* Because of the 5x5 luminance kernel, the normal interpolation scheme
   cannot get any closer than two pixels to the border of the image.  We use
   a bilinear filter to get one rank of pixels closer.  Since not very many
   pixels need to be set here, some tricky programming is used to reduce the
   code size at the expense of speed.  The "if" statement after the row "for"
   loop makes the inner loop visit columns

      1, 2, ..., ncols-3, ncols-2

   for rows 1 and nrows-2, and just columns 1 and ncols-2 for the interior
   rows.  The outermost rank of pixels is bordered by a simple copy from the
   adjacent BGR pixels. The copies to the first and last pixels in each row
   are carried out by simple assignment statements after the inner loop below;
   the copies to rows 0 and ncols-1 are handled by calls to memcpy(). */

	bptr = bayer + bstride;
	optr = out + ostride;
	c0Ptr = out + ostride;
	c1Ptr = c0Ptr + nrows*ostride;
	c2Ptr = c1Ptr + nrows*ostride;
	for (row=1; row<nrows-1; row++) 
	{
		if ((row==1) || (row==nrows-2)) 
		{
			colinc = 1;
		} 
		else 
		{
			colinc = ncols - 3;
		}

		for (col=1; col<ncols-1; col+=colinc) 
		{
			if ((row&1) == RowOrder) 
			{
				if ((col&1) == ColOrder) 
				{ /* Blue */
					b = 64*bptr[col];

					g = 16*(
						+bptr[col  -1*bstride]
						+bptr[col-1     ]
						+bptr[col+1     ]
						+bptr[col  +1*bstride]
								 );

					r = 16*(
						+bptr[col-1-1*bstride]
						+bptr[col+1-1*bstride]
						+bptr[col-1+1*bstride]
						+bptr[col+1+1*bstride]
								 );
				} 
				else 
				{                   /* Green in blue  row */
					b = 32*(
						+bptr[col-1     ]
						+bptr[col+1     ]
                             );

					g = 8*(
						+bptr[col-1-1*bstride]
						+bptr[col+1-1*bstride]
						+4*bptr[col]
						+bptr[col-1+1*bstride]
						+bptr[col+1+1*bstride]
                             );

					r = 32*(
						+bptr[col  -1*bstride]
						+bptr[col  +1*bstride]
                             );
				}
			}
			else 
			{
				if ((col&1) == ColOrder) 
				{ /* Green in red row */
					b = 32*(
						+bptr[col  -1*bstride]
						+bptr[col  +1*bstride]
                             );

					g = 8*(
						+bptr[col-1-1*bstride]
						+bptr[col+1-1*bstride]
						+4*bptr[col]
						+bptr[col-1+1*bstride]
						+bptr[col+1+1*bstride]
                             );

					r = 32*(
						+bptr[col-1     ]
						+bptr[col+1     ]
                             );
				}
				else 
				{                   /* Red */
					b = 16*(
						+bptr[col-1-1*bstride]
						+bptr[col+1-1*bstride]
						+bptr[col-1+1*bstride]
						+bptr[col+1+1*bstride]
                             );

					g = 16*(
						+bptr[col  -1*bstride]
						+bptr[col-1     ]
						+bptr[col+1     ]
						+bptr[col  +1*bstride]
                             );

					r = 64*bptr[col];
				}
			}

         /* At this point, r, g, b = 64 times desired value */
			switch(type)
			{
			case YONLY:
				optr[col] = r/256 + g/128 +b/256;
				break;

			case YCrCb:
				if(bChannelSeperate)
				{
					c0Ptr[col] = r/256 + g/128 +b/256;				// Y
					c1Ptr[col] = clip(r/64-(int)c0Ptr[col]+128);	// Cr
					c2Ptr[col] = clip(b/64-(int)c0Ptr[col]+128);	// Cb
				}
				else
				{
					optr[col*3] = r/256 + g/128 +b/256;					// Y
					optr[col*3+1] = clip(r/64-(int)optr[col*3]+128);	// Cr
					optr[col*3+2] = clip(b/64-(int)optr[col*3]+128);	// Cb
				}
				break;

			case RGB:
				if(bChannelSeperate)
				{
					c0Ptr[col] = clip(r/64);			// R
					c1Ptr[col] = clip(g/64);			// G
					c2Ptr[col] = clip(b/64);			// B
				}
				else
				{

					optr[col*3] = clip(r/64);			// R
					optr[col*3+1]= clip(g/64);			// G
					optr[col*3+2] = clip(b/64);			// B
				}
				break;

			case BGR:
				if(bChannelSeperate)
				{
					c0Ptr[col] = clip(b/64);			// B
					c1Ptr[col] = clip(g/64);			// G
					c2Ptr[col] = clip(r/64);			// R
				}
				else
				{
					optr[col*3] = clip(b/64);			// B
					optr[col*3+1]= clip(g/64);			// G
					optr[col*3+2] = clip(r/64);			// R
				}
				break;
			}
		}

		if(YONLY)
		{
			optr[0] = optr[1];
			optr[ncols-1] = optr[ncols-2];
		}
		else
		{
			if(bChannelSeperate)
			{
				c0Ptr[0] = c0Ptr[1];
				c1Ptr[0] = c1Ptr[1];
				c2Ptr[0] = c2Ptr[1];
				c0Ptr[ncols-1] = c0Ptr[ncols-2];
				c1Ptr[ncols-1] = c1Ptr[ncols-2];
				c2Ptr[ncols-1] = c2Ptr[ncols-2];
			}
			else
			{	

				optr[0] = optr[3];
				optr[1] = optr[4];
				optr[2] = optr[5];
				optr[3*ncols-3] = optr[3*ncols-6];
				optr[3*ncols-2] = optr[3*ncols-5];
				optr[3*ncols-1] = optr[3*ncols-4];

			}
		}
		bptr += bstride;
		optr += ostride;
		c0Ptr+= ostride;
		c1Ptr+= ostride;
		c2Ptr+= ostride;
   }

   if(type == YONLY) // One channel
   {
	   	memcpy(out, out+ostride, ncols*sizeof(*out));
		memcpy(out+ostride*(nrows-1), out+ostride*(nrows-2), ncols*sizeof(*out));
   }
   else	// Three channels
   {
	   if(bChannelSeperate)
	   {
			unsigned char* tempPtr = out;
			for(int i=0; i<3; i++)
			{
				memcpy(tempPtr, tempPtr+ostride, ncols*sizeof(*out));
				memcpy(tempPtr+ostride*(nrows-1), tempPtr+ostride*(nrows-2), ncols*sizeof(*out));
				tempPtr += ostride*nrows;
			}
	   }
	   else
	   {
			memcpy(out, out+ostride, ncols*sizeof(*out)*3);
			memcpy(out+ostride*(nrows-1), out+ostride*(nrows-2), ncols*sizeof(*out)*3);
	   }
   }
}

// Return NULL if failed, otherwise the buffer for luminance
unsigned char* Bayer2Lum_rect(
	int				iBayerCols,		// Bayer Buffer dimensions 
	int				iBayerRows,
	unsigned char*	pBayer,			// Input 8-bit Bayer image
	int				iBayerStride,	// Addressed as bayer[col + row*bstride]  
	BayerType		order,			// Bayer pattern order; use the enums in bayer.h
	UIRect			rectIn,			// Input rect for Roi 
	UIRect*			pRectOut)		// Output rect for Roi
{
	// Normalize for bayer pattern
	int iFirstCol =  rectIn.FirstColumn/2*2;		// First one must be dividable by 2
	int iLastCol = (rectIn.LastColumn%2)==1 ? rectIn.LastColumn : rectIn.LastColumn+1;
	if(iLastCol > iBayerCols-1) iLastCol = ((iBayerCols-1)%2)==1 ? iBayerCols-1 : iBayerCols-2;

	int iFirstRow =  rectIn.FirstRow/2*2;			// First one must be dividable by 2
	int iLastRow = (rectIn.LastRow%2)==1 ? rectIn.LastRow : rectIn.LastRow+1;
	if(iLastRow > iBayerRows-1) iLastRow = ((iBayerRows-1)%2)==1 ? iBayerRows-1 : iBayerRows-2;

	pRectOut->FirstColumn = (unsigned int)iFirstCol;
	pRectOut->LastColumn = (unsigned int)iLastCol;
	pRectOut->FirstRow = (unsigned int)iFirstRow;
	pRectOut->LastRow = (unsigned int)iLastRow;

	// Allocate buffer
	unsigned int iSize = pRectOut->Size();
	if(iSize == 0)
		return(NULL);

	unsigned char* pOut = new unsigned char[iSize];

	// Bayer to luminace conversion
	BayerLum(					// Bayer interpolation 
		pRectOut->Columns(),	// Dimensions 
		pRectOut->Rows(),
		pBayer + iBayerStride*pRectOut->FirstRow + pRectOut->FirstColumn,		
								// Input 8-bit Bayer image
		iBayerStride,			// Addressed as bayer[col + row*bstride]  
		order,					// Bayer pattern order; use the enums in bayer.h
		pOut,					// In/Out 24-bit BGR/YCrCb or 8-bit Y(luminance) image, 
								// allocated outside and filled inside of function
		pRectOut->Columns(),	// Addressed as out[col*NumOfChannel+ChannelIndex + row*ostride] if channels are combined
								// or out[col + row*ostride + (ChannelIndex-1)*ostride*nrows] if channels are seperated
		YONLY,					// Type of color BGR/YCrCb/Y
		false);					// true, the channel stored seperated

	return(pOut);
}

// 1D smooth
void inline Smooth1d_B2L(unsigned char* pcInLine, unsigned char* pcOutLine, unsigned int iLength) 
{
	// Out[i] = in[i-2]/8+in[i-1]/8+in[i]/4+in[i+1]/4+in[i+2]/8+in[i+3]/8
	// Filter must be even in size and symmetry 
	
	// Values in middle
	for(unsigned int i = 2; i<iLength-3; i++)
	{
		pcOutLine[i] = (unsigned char)(
				((int)pcInLine[i] + (int)pcInLine[i+1])/4
			+	((int)pcInLine[i-2] + (int)pcInLine[i-1] + (int)pcInLine[i+2] + (int)pcInLine[i+3])/8
			);
	}

	// First two values
	pcOutLine[0] = pcOutLine[2];
	pcOutLine[1] = pcOutLine[2];
	
	// Last three values
	pcOutLine[iLength-3] = pcOutLine[iLength-4];
	pcOutLine[iLength-2] = pcOutLine[iLength-4];
	pcOutLine[iLength-1] = pcOutLine[iLength-4];
}

// Bayer pattern to luminance conversion by smooth filter for image registration
// Conversion is fast but not accurate
void Smooth2d_B2L(
	unsigned char* pcInBuf, unsigned int iInSpan,
	unsigned char* pcOutBuf, unsigned int iOutSpan,
	unsigned int iWidth, unsigned int iHeight)
{
	unsigned char* pcTempBuf = new unsigned char[iWidth*iHeight];
	unsigned char* pcTempInLine = new unsigned char[iHeight];
	unsigned char* pcTempOutLine = new unsigned char[iHeight];

	// Smooth for each row (pcInBuf->pcTempBuf)
	unsigned char* pcInLine = pcInBuf;
	unsigned char* pcOutLine = pcTempBuf;
	for(unsigned int i=0; i<iHeight; i++)
	{
		Smooth1d_B2L(pcInLine, pcOutLine, iWidth);
		pcInLine += iInSpan;
		pcOutLine += iWidth;
	}

	// Smooth for each column (pcTempBuf->pcOutBuf)
	for(unsigned int i=0; i<iWidth; i++)
	{
		// Input column
		int iIndex = i;
		for(unsigned int j=0; j<iHeight; j++)
		{
			pcTempInLine[j] =  pcTempBuf[iIndex];
			iIndex += iWidth;
		}

		Smooth1d_B2L(pcTempInLine, pcTempOutLine, iHeight);
		
		// Output 
		iIndex = i;
		for(unsigned int j=0; j<iHeight; j++)
		{
			pcOutBuf[iIndex] =  pcTempOutLine[j];
			iIndex += iOutSpan;
		}
	}

	// Clean up
	delete [] pcTempBuf;
	delete [] pcTempInLine;
	delete [] pcTempOutLine;
}

// Demosaic based on Gaussian interploation
void Demosaic_Gaussian(
	int				iNumCol,			// Image dimensions 
	int				iNumRow,
	unsigned char*	pcBayer,			// Input 8-bit Bayer image
	int				iBayerStr,			// Addressed as bayer[col + row*bstride] 
	BayerType		order,				// Bayer pattern order; use the enums in bayer.h
	unsigned char*	pcOut,				// In/Out 24-bit BGR/YCrCb or 8-bit Y(luminance) image, 
										// allocated outside and filled inside of function
	int				iOutStr,			// Addressed as out[col*NumOfChannel+ChannelIndex + row*iOutStr] if channels are combined
										// or out[col + row*iOutStr + (ChannelIndex-1)*iOutStr*iNumRow] if channels are seperated
	COLORSTYLE		type,				// Type of color BGR/YCrCb
	bool			bChannelSeperate)	// true, the channel stored seperated)
{
	/* Gaussian filter	[	1	2	1	] 
						[	2	4	2	]
						[	1	2	1	]
	*/	
	
	// for pixel(1,1) with orign (0,0)
	bool bRowR, bColG;
	switch(order)
	{
	case BGGR:
		bRowR = true;
		bColG = false;
		break;
	case GBRG:
		bRowR = true;
		bColG = true;
		break;
	case GRBG:
		bRowR = false;
		bColG = true;
		break;
	case RGGB:
		bRowR = false;
		bColG = false;
		break;
	}

	// For All pixele except ones on the edges
	unsigned int iR, iG, iB;
	unsigned char* pLine = pcBayer + iBayerStr;	// Second row
	int iStep = 3;
	if(bChannelSeperate)
		iStep = 1;
	unsigned char* pcLineB = pcOut + iOutStr; // Second row
	unsigned char* pcLineG = pcLineB + 1;
	unsigned char* pcLineR = pcLineG + 1;
	if(bChannelSeperate)
	{
		pcLineG = pcLineB + iOutStr*iNumRow;
		pcLineR = pcLineG + iOutStr*iNumRow;
	}
	unsigned char *pcR, *pcG, *pcB; 
	for(int iy=1; iy<iNumRow-1; iy++)
	{
		// The second pixel in row
		pcR = pcLineR + iStep;
		pcG = pcLineG + iStep;
		pcB = pcLineB + iStep;
		for(int ix=1; ix<iNumCol-1; ix++)
		{
			if(bRowR)		
			{				// G B G
				if(bColG)	// R G R
				{			// G B G
					iR = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
					*pcR = (unsigned char)(iR>>1);

					iG = (((unsigned int)pLine[ix])<<2) +
						(unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1]+
						(unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
					*pcG = (unsigned char)(iG>>3);

					iB = (unsigned int)pLine[ix-iBayerStr] + (unsigned int)pLine[ix+iBayerStr];
					*pcB = (unsigned char)(iB>>1);
				}			// B G B
				else		// G R G
				{			// B G B
					*pcR = pLine[ix];

					iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1] + 
						(unsigned int)pLine[ix-iBayerStr] + (unsigned int)pLine[ix+iBayerStr];
					*pcG = (unsigned char)(iG>>2);

					iB = (unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1]+
						(unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
					*pcB = (unsigned char)(iB>>2);
				}
			}
			else
			{				// G R G
				if(bColG)	// B G B
				{			// G R G
					iR = (unsigned int)pLine[ix-iBayerStr] + (unsigned int)pLine[ix+iBayerStr];
					*pcR = (unsigned char)(iR>>1);

					iG = (((unsigned int)pLine[ix])<<2) +
						(unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1]+
						(unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
					*pcG = (unsigned char)(iG>>3);

					iB = (unsigned int)pLine[ix-1]+ (unsigned int)pLine[ix+1];
					*pcB = (unsigned char)(iB>>1);
				}			// R G R
				else		// G B G
				{			// R G R
					iR = (unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1]+
						(unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
					*pcR = (unsigned char)(iR>>2);

					iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1] + 
						(unsigned int)pLine[ix-iBayerStr]+ (unsigned int)pLine[ix+iBayerStr];
					*pcG = (unsigned char)(iG>>2);

					*pcB = pLine[ix];
				}
			}

			if(type == YCrCb)
			{
				int iTemp[3];
				iTemp[0] = ((int)(*pcR) + (int)(*pcG)*2 + (int)(*pcB))>>2;		// Y	
				iTemp[1] = (int)(*pcR) - iTemp[0] + 128;			// Cr
				iTemp[2] = (int)(*pcB) - iTemp[0] + 128;			// Cb

				// Write back
				for(int i=0; i<3; i++)
				{
					if(iTemp[i]<0) iTemp[i]=0;
					if(iTemp[i]>255) iTemp[i]=255;
					if(i==0)
						*pcB = (unsigned char)iTemp[i];
					else if(i==1)
						*pcG = (unsigned char)iTemp[i];
					else
						*pcR = (unsigned char)iTemp[i];
				}
			}
		
			bColG = !bColG;
			pcR += iStep;	// Move one pixel in the same row
			pcG	+= iStep;
			pcB += iStep;
		}
		// Move one row
		pLine += iBayerStr;
		pcLineR += iOutStr;
		pcLineG += iOutStr;
		pcLineB += iOutStr;

		bRowR = !bRowR;
		bColG = !bColG;
	}

	// For first line begin with the second pixel
	bRowR = !bRowR;			// pixel(0,1) relative to pixel(1,1)
	bColG = !bColG;
	pLine = pcBayer;		// first row
	pcB = pcOut + iStep;	// Pixel(0,1)
	pcG = pcB+1;
	pcR = pcG+1;
	if(bChannelSeperate)
	{
		pcG = pcB + iOutStr*iNumRow;
		pcR = pcG + iOutStr*iNumRow;
	}
	for(int ix=1; ix<iNumCol-1; ix++)
	{
		if(bRowR)		
		{
			if(bColG)	// R G R
			{			// G B G
				iR = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcR = (unsigned char)(iR/2); 

				*pcG = pLine[ix];

				*pcB = pLine[ix+iBayerStr]; 
			}
			else		// G R G
			{			// B G B
				*pcR = pLine[ix];

				iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcG = (unsigned char)(iG/2);

				iB =  (unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
				*pcB = (unsigned char)(iB/2);
			}	
		}
		else			 
		{
			if(bColG)	// B G B
			{			// G R G
				*pcR = pLine[ix+iBayerStr];

				*pcG = pLine[ix];

				iB = (unsigned int)pLine[ix-1]+(unsigned int)pLine[ix+1];
				*pcB = (unsigned char)(iB/2); 
			}
			else		// G B G
			{			// R G R
				iR = (unsigned int)pLine[ix+iBayerStr-1] + (unsigned int)pLine[ix+iBayerStr+1];
				*pcR = (unsigned char)(iR/2); 

				iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcG = (unsigned char)(iG/2);

				*pcB = pLine[ix];
			}
		}

		// Move one pixel in the same row
		bColG = !bColG;
		pcR += iStep;
		pcG += iStep;
		pcB += iStep;
	}

	// For last line
	bRowR = !bRowR; // pixel(iNumRow-1,1) relative to pixel (0,1)
	bColG = !bColG;
	pLine = pcBayer + (iNumRow-1)*iBayerStr;	// last row
	pcB = pcOut + (iNumRow-1)*iOutStr+ iStep;	// pixel(iNumRow-1,1)
	pcG = pcB+1;
	pcR = pcG+1;
	if(bChannelSeperate)
	{
		pcG = pcB + iOutStr*iNumRow;
		pcR = pcG + iOutStr*iNumRow;
	}
	for(int ix=1; ix<iNumCol-1; ix++)
	{
		if(bRowR)		
		{				// G B G
			if(bColG)	// R G R
			{			
				iR = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcR = (unsigned char)(iR/2); 

				*pcG = pLine[ix];

				*pcB = pLine[ix-iBayerStr]; 
			}			// B G B
			else		// G R G
			{			
				*pcR = pLine[ix];

				iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcG = (unsigned char)(iG/2);

				iB =  (unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1];
				*pcB = (unsigned char)(iB/2);
			}	
		}
		else			 
		{				// G R G
			if(bColG)	// B G B
			{			
				*pcR = pLine[ix-iBayerStr];

				*pcG = pLine[ix];

				iB = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcB = (unsigned char)(iB/2); 
			}			// R G R
			else		// G B G
			{			
				iR = (unsigned int)pLine[ix-iBayerStr-1] + (unsigned int)pLine[ix-iBayerStr+1];
				*pcR = (unsigned char)(iR/2); 

				iG = (unsigned int)pLine[ix-1] + (unsigned int)pLine[ix+1];
				*pcG = (unsigned char)(iG/2);

				*pcB = pLine[ix];
			}
		}
		// Move one pixel in the same row
		bColG = !bColG;
		pcR += iStep;
		pcG += iStep;
		pcB += iStep;
	}
	
	// Copy to first and last rows
	pcLineB = pcOut;
	pcLineG = pcLineB+1;
	pcLineR = pcLineG+1;
	if(bChannelSeperate)
	{
		pcLineG = pcLineB + iOutStr*iNumRow;
		pcLineR = pcLineG + iOutStr*iNumRow;
	}
	unsigned int iEnd = (iNumCol-1)*iStep;
	for(int iy=0; iy<iNumRow; iy++)
	{
		pcLineR[0] = pcLineR[iStep];
		pcLineG[0] = pcLineG[iStep];
		pcLineB[0] = pcLineB[iStep];
		pcLineR[iEnd] = pcLineR[iEnd-iStep];
		pcLineG[iEnd] = pcLineG[iEnd-iStep];
		pcLineB[iEnd] = pcLineB[iEnd-iStep];

		pcLineR += iOutStr;
		pcLineG += iOutStr;
		pcLineB += iOutStr;
	}
}

// leftM[8] * rightM[8] = outM[8]
void MultiProjective2D(double* leftM, double* rightM, double* outM)
{
    outM[0] = leftM[0]*rightM[0]+leftM[1]*rightM[3]+leftM[2]*rightM[6];
	outM[1] = leftM[0]*rightM[1]+leftM[1]*rightM[4]+leftM[2]*rightM[7];
	outM[2] = leftM[0]*rightM[2]+leftM[1]*rightM[5]+leftM[2]*1;
											 
	outM[3] = leftM[3]*rightM[0]+leftM[4]*rightM[3]+leftM[5]*rightM[6];
	outM[4] = leftM[3]*rightM[1]+leftM[4]*rightM[4]+leftM[5]*rightM[7];
	outM[5] = leftM[3]*rightM[2]+leftM[4]*rightM[5]+leftM[5]*1;
											 
	outM[6] = leftM[6]*rightM[0]+leftM[7]*rightM[3]+1*rightM[6];
	outM[7] = leftM[6]*rightM[1]+leftM[7]*rightM[4]+1*rightM[7];
	double dScale = leftM[6]*rightM[2]+leftM[7]*rightM[5]+1*1;

	if(dScale<0.01 && dScale>-0.01)
		dScale = 0.01;

	for(int i=0; i<8; i++)
		outM[i] = outM[i]/dScale;
}

// (Row, Col) -> (x, y)
void Pixel2World(double* trans, double row, double col, double* px, double* py)
{
    double dScale = 1+ trans[6]*row+trans[7]*col;
    dScale = 1/dScale;
    *px = trans[0] * row + trans[1] * col + trans[2];
    *px *= dScale;
    *py = trans[3] * row + trans[4] * col + trans[5];
    *py *= dScale;
}

// Calculate half size of FOV in world space
void CalFOVHalfSize(double* trans, unsigned int iImW, unsigned int iImH, float* pfHalfW, float* pfHalfH)
{
    // calcualte width
    double dLeftx, dLefty, dRightx, dRighty;
    Pixel2World(trans, (iImH - 1) / 2.0, 0, &dLeftx, &dLefty);
    Pixel2World(trans, (iImH - 1) / 2.0, iImW-1, &dRightx, &dRighty);
    *pfHalfW = (float)((dRighty-dLefty)/2.0);

    // calcualte height
    double dTopx, dTopy, dBottomx, dBottomy;
    Pixel2World(trans, 0, (iImW-1)/2.0, &dTopx, &dTopy);
    Pixel2World(trans, iImH - 1, (iImW-1)/2.0, &dBottomx, &dBottomy);
    *pfHalfH = (float)((dBottomx-dTopx)/2.0);
}

