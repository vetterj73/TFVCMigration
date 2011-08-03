#include "Utilities.h"
#include "lsqrpoly.h" 
#include "math.h"
#include "morpho.h"

// Inverse a matrix,
// inMatrix: input matrix, data stored row by row
// outMatrix: output Matrix, data stored row by row
// rows and cols: size of matrix 
void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols)
{
	double* in = new double[rows*cols];
	
	// Transpose matrix
	unsigned int ix, iy;
	for(iy=0; iy<rows; iy++)
		for(ix=0; ix<cols; ix++)
			in[ix*rows+iy] = inMatrix[iy*cols+ix];

	double*	sigma = new double[cols];
	double* answer = new double[cols*rows];
	double* b = new double [rows];

	// perform factorization of system A = QR
	int qrdRet = 
		qrd(
				in,			/* System matrix, m-by-n */
				sigma,      /* Diagonals of R (caller reserves n elements) */
				rows,     /* Number of rows in system matrix */
				rows,		/* Spacing between columns in system matrix */
				cols );	/* Number of columns in system matrix */

                            /* Return values

                                 0 - Normal completion.
                                 1 - Matrix was of incompatible dimensions.
                                 2 - Singular system matrix. */
   double condition =
	   rcond(               /* Reciprocal condition number estimator */
			   cols,		/* Number of unknowns */
			   in,			/* QR factorization returned from qrd() */
			   rows,       /* Spacing between columns of qr[] */
			   sigma        /* R diagonals from qrd() */
            );

   	for(unsigned int iter(0); iter<rows; ++iter)
	{
		for(unsigned int bIndex(0); bIndex<rows; ++bIndex)
			b[bIndex] = 0;

		b[iter] = 1;

		qrsolv (
					in,			/* Factored system matrix, from qrd() */
					sigma,		/* Diagonal of R, from qrd() */
					b,			/* Constant vector, overwritten by solution */
					rows,		/* Number of rows in system matrix */
					rows,		/* Spacing between columns in system matrix */
					cols,		/* Number of columns in system matrix */
					1     );	/*	0 - premultiply  b  by Q
									1 - premultiply  b  by inv(QR)
									2 - premultiply  b  by QR
									3 - premultiply  b  by inv(R^T R)
									4 - premultiply  b  by Q inv(R^T)
									5 - premultiply  b  by R
									6 - premultiply  b  by inv(R)

									 qrsolv() always returns 0 */

		// copy result into the answer
		for(unsigned int index2(0); index2<rows; ++index2)
			answer[iter*rows+index2] = b[index2];
	}

	// Tranpose matrix
	for(iy=0; iy<rows; iy++)
		for(ix=0; ix<cols; ix++)
			outMatrix[iy*cols+ix] = answer[ix*rows+iy];

	delete [] in;
	delete [] sigma;
	delete [] answer;
	delete [] b;
}

// Fill a ROI of the output image by transforming the input image
// Both output image and input image are 8bits/pixel (can add 16bits/pixel support easily)
// pInBuf, iInSpan, iInWidth and iInHeight: input buffer and its span, width and height
// pOutBuf and iOutspan : output buffer and its span
// iROIWidth, iHeight: the size of buffer need to be transformed
// iOutROIStartX, iOutROIStartY, iOutROIWidth and iOutROIHeight: the ROI of the output image
// dInvTrans: the 3*3 transform from output image to input image (has the same unit as dHeightResolution)
bool ImageMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]) 
{
	// Sanity check
	if((iOutROIStartX+iOutROIWidth>iOutSpan) || (iInWidth>iInSpan)
		|| (iInWidth <2) || (iInHeight<2)
		|| (iOutROIWidth<=0) || (iOutROIHeight<=0))
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

	unsigned char* pbOutBuf = pOutBuf + iOutROIStartY*iOutSpan;
	int iInSpanP1 = iInSpan+1;

	// some local variable
	unsigned char* pbPixPtr;
	int iflrdX, iflrdY;
	int iPix0, iPix1, iPixW, iPixWP1;
	int iPixDiff10;
	double dT01__dIY_T02, dT11__dIY_T12, dT21__dIY_T22;
	double dIX, dIY, dX, dY, dDiffX, dDiffY;
	double dVal;
 
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
				if ((dX < 0) | (dY < 0) |
				  (dX >= iInWidth-1) | (dY >= iInHeight-1)) 
				{
					pbOutBuf[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					/* Compute fractional differences */
					iflrdX = (int)dX;	/* Compared to int-to-double, double-to-int */
					iflrdY = (int)dY;	/*   much more costly */
			    
					/* Compute pointer to input pixel at (dX,dY) */
					pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;

					iPix0   = (int) pbPixPtr[0]; /* The 2x2 neighborhood used */
					iPix1   = (int) pbPixPtr[1];
					iPixW   = (int) pbPixPtr[iInSpan];
					iPixWP1 = (int) pbPixPtr[iInSpanP1];

					iPixDiff10 = iPix1 - iPix0; /* Used twice, so compute once */

					dDiffX = dX - (double) iflrdX;
					dDiffY = dY - (double) iflrdY;

					pbOutBuf[iX] = (unsigned char)((double) iPix0			
						+ dDiffY * (double) (iPixW - iPix0)	
						+ dDiffX * ((double) iPixDiff10	
						+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
						+ 0.5); // for round up 
				}
			} //ix
			pbOutBuf += iOutSpan;		/* Next line in the output buffer */
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
					pbOutBuf[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					/* Compute fractional differences */
					iflrdX =(int)dX;
					iflrdY = (int)dY;
			    
					pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;

					iPix0   = (int) pbPixPtr[0];
					iPix1   = (int) pbPixPtr[1];
					iPixW   = (int) pbPixPtr[iInSpan];
					iPixWP1 = (int) pbPixPtr[iInSpanP1];

					iPixDiff10 = iPix1 - iPix0;

					dDiffX = dX - (double) iflrdX;
					dDiffY = dY - (double) iflrdY;

					pbOutBuf[iX] = (unsigned char)((double) iPix0			
						+ dDiffY * (double) (iPixW - iPix0)	
						+ dDiffX * ((double) iPixDiff10	
						+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
						+ 0.5);
				}
			} // ix
			pbOutBuf += iOutSpan;
		} // iy
    } // else

	return(true);
}

// Fill a ROI of the output image with a height map by transforming the input image
// Assume the center of image corresponding a vertical line from camera to object surface
// Both output image and input image are 8bits/pixel (can add 16bits/pixel support easily)
// pInBuf, iInSpan, iInWidth and iInHeight: input buffer and its span, width and height
// pOutBuf and iOutspan : output buffer and its span
// iROIWidth, iHeight: the size of buffer need to be transformed
// iOutROIStartX, iOutROIStartY, iOutROIWidth and iOutROIHeight: the ROI of the output image
// dInvTrans: the 3*3 transform from output image to input image
// pHeightImage and iHeightSpan, height image buffer and its span
// dHeightResolution: the height represented by each grey level
// dPupilDistance: camera pupil distance
bool ImageMorphWithHeight(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3],
	unsigned char* pHeightImage, unsigned int iHeightSpan,
	double dHeightResolution, double dPupilDistance) 
{
	// Sanity check
	if((iOutROIStartX+iOutROIWidth>iOutSpan) || (iInWidth>iInSpan)
		|| (iInWidth <2) || (iInHeight<2)
		|| (iOutROIWidth<=0) || (iOutROIHeight<=0))
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

	unsigned char* pbOutBuf = pOutBuf + iOutROIStartY*iOutSpan;
	int iInSpanP1 = iInSpan+1;

	// some local variable
	unsigned char* pbPixPtr;
	int iflrdX, iflrdY;
	int iPix0, iPix1, iPixW, iPixWP1;
	int iPixDiff10;
	double dT01__dIY_T02, dT11__dIY_T12, dT21__dIY_T22;
	double dIX, dIY, dX, dY, dDiffX, dDiffY;
	double dVal;
 
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
				if(pHeightImage[iY*iHeightSpan+iX]>0)
				{
					double dHeight = pHeightImage[iY*iHeightSpan+iX]*dHeightResolution;
					double dRatio = dHeight/(dPupilDistance-dHeight);
					double dOffsetX = (dX-iInWidth/2.0)*dRatio;
					double dOffsetY = (dY-iInHeight/2.0)*dRatio;
					dX += dOffsetX;
					dY += dOffsetY;
				}

			  /* Check if back projection is outside of the input image range. Note that
				 2x2 interpolation touches the pixel to the right and below right,
				 so right and bottom checks are pulled in a pixel. */
				if ((dX < 0) | (dY < 0) |
				  (dX >= iInWidth-1) | (dY >= iInHeight-1)) 
				{
					pbOutBuf[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					/* Compute fractional differences */
					iflrdX = (int)dX;	/* Compared to int-to-double, double-to-int */
					iflrdY = (int)dY;	/*   much more costly */
			    
					/* Compute pointer to input pixel at (dX,dY) */
					pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;

					iPix0   = (int) pbPixPtr[0]; /* The 2x2 neighborhood used */
					iPix1   = (int) pbPixPtr[1];
					iPixW   = (int) pbPixPtr[iInSpan];
					iPixWP1 = (int) pbPixPtr[iInSpanP1];

					iPixDiff10 = iPix1 - iPix0; /* Used twice, so compute once */

					dDiffX = dX - (double) iflrdX;
					dDiffY = dY - (double) iflrdY;

					pbOutBuf[iX] = (unsigned char)((double) iPix0			
						+ dDiffY * (double) (iPixW - iPix0)	
						+ dDiffX * ((double) iPixDiff10	
						+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
						+ 0.5); // for round up 
				}
			} //ix
			pbOutBuf += iOutSpan;		/* Next line in the output buffer */
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
					pbOutBuf[iX] = 0x00;	/* Clipped */
				}
				else 
				{
					/* Compute fractional differences */
					iflrdX =(int)dX;
					iflrdY = (int)dY;
			    
					pbPixPtr = pInBuf + iflrdX + iflrdY * iInSpan;

					iPix0   = (int) pbPixPtr[0];
					iPix1   = (int) pbPixPtr[1];
					iPixW   = (int) pbPixPtr[iInSpan];
					iPixWP1 = (int) pbPixPtr[iInSpanP1];

					iPixDiff10 = iPix1 - iPix0;

					dDiffX = dX - (double) iflrdX;
					dDiffY = dY - (double) iflrdY;

					pbOutBuf[iX] = (unsigned char)((double) iPix0			
						+ dDiffY * (double) (iPixW - iPix0)	
						+ dDiffX * ((double) iPixDiff10	
						+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
						+ 0.5);
				}
			} // ix
			pbOutBuf += iOutSpan;
		} // iy
    } // else

	return(true);
}


// Modified from Eric Rudd's BayerLum() function
// Convert Bayer image into Luminance
// Output data only valid int the range of columns [2, nCols-3] and rows [2 nRows-3]
void BayerToLum(                
   int            ncols,		// Image dimensions
   int            nrows,
   unsigned char  bayer[],      // Input 8-bit Bayer image 
   int            bstride,      // Addressed as bayer[col + row*bstride] 
   unsigned char  lum[],        // output Luminance image 
   int            lstride)      // Addressed as out[col + row*ostride] 
{
   unsigned char *bptr, *optr;
   int col, row, y;

   bptr = bayer + 2*bstride;
   optr = lum + 2*lstride;
   for (row=2; row<nrows-2; row++) {
      for (col=2; col<ncols-2; col++) {
         y =
           + 156*(
                   +bptr[col                ]
                 )

           + 30*(
                   +bptr[col     + 1*bstride]
                   +bptr[col     - 1*bstride]
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
                   +bptr[col - 1 + 1*bstride]
                   +bptr[col + 1 + 1*bstride]
                   +bptr[col - 1 - 1*bstride]
                   +bptr[col + 1 - 1*bstride]
                )

           +    (
                   +bptr[col - 1 + 2*bstride]
                   +bptr[col + 1 + 2*bstride]
                   +bptr[col - 2 + 1*bstride]
                   +bptr[col + 2 + 1*bstride]
                   +bptr[col - 2 - 1*bstride]
                   +bptr[col + 2 - 1*bstride]
                   +bptr[col - 1 - 2*bstride]
                   +bptr[col + 1 - 2*bstride]
                )

           -  3*(
                   +bptr[col - 2 + 2*bstride]
                   +bptr[col + 2 + 2*bstride]
                   +bptr[col - 2 + 2*bstride]
                   +bptr[col + 2 + 2*bstride]
                )
         ;

		int iTemp = y/256;
		if(iTemp >= 255) iTemp = 255;
		if(iTemp <= 0) iTemp = 0;
		optr[col] = iTemp;
	  }    
      bptr += bstride;
      optr += lstride;
   }
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
	for(int iy=0; iy<iHeight; iy++)
	{
		for(int ix=0; ix<iWidth; ix++)
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

