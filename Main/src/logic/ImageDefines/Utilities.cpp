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
	if(iNumChannels==3)
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

