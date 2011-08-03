/*
 * Copyright 2011 CyberOptics Corporation.  All rights reserved.
 */

#ifndef _UTILITIES_KERNEL_H_
#define _UTILITIES_KERNEL_H_

#include <stdio.h>
#include "Utilities_kernel.h"

#define TILE_WIDTH 16

__global__ void ConvolutionKernel(unsigned char* dA, unsigned char* dB, unsigned int iInSpan, unsigned int iInHeight,
	unsigned int iInWidth, unsigned int iOutROIWidth, unsigned int iOutROIHeight, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY)
//__global__ void ConvolutionKernel(unsigned char* dA, unsigned char* dB, int wA, int hA, int startX, int startY)
{

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by * 16 + ty;
	int Col = bx * 16 + tx;

	if (Row >= iOutROIHeight || Col >= iOutROIWidth) return;

	// some local variable
	int iflrdX, iflrdY;
	int iPix0, iPix1, iPixW, iPixWP1;
	int iPixDiff10;
	int iOffset;
	double dT01__dIY_T02, dT11__dIY_T12, dT21__dIY_T22;
	double dIX, dIY, dX, dY, dDiffX, dDiffY;
 

	//for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
	//{

	dIY = (double) Row + iOutROIStartY;
	dT01__dIY_T02 = coeffs[0][1] * dIY + coeffs[0][2];
	dT11__dIY_T12 = coeffs[1][1] * dIY + coeffs[1][2];

	//for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
	//{

	dIX = (double) Col + iOutROIStartX;
	dX = coeffs[0][0]*dIX + dT01__dIY_T02;
	dY = coeffs[1][0]*dIX + dT11__dIY_T12;

	/* Check if back projection is outside of the input image range. Note that
		2x2 interpolation touches the pixel to the right and below right,
		so right and bottom checks are pulled in a pixel. */
	if ((dX < 0) | (dY < 0) |
		(dX >= iInWidth-1) | (dY >= iInHeight-1)) 
	{
		dB[Col + Row * iOutROIWidth] = 0x0;	/* Clipped */
		return;
	}

	/* Compute fractional differences */
	iflrdX = (int)dX;	/* Compared to int-to-double, double-to-int */
	iflrdY = (int)dY;	/*   much more costly */
			    
	/* Compute offset to input pixel at (dX,dY) */
	iOffset = iflrdX + iflrdY * iInSpan;

	iPix0   = (int) dA[iOffset]; /* The 2x2 neighborhood used */
	iPix1   = (int) dA[iOffset+1];
	iPixW   = (int) dA[iOffset+iInSpan];
	iPixWP1 = (int) dA[iOffset+iInSpan+1];

	iPixDiff10 = iPix1 - iPix0; /* Used twice, so compute once */

	dDiffX = dX - (double) iflrdX;
	dDiffY = dY - (double) iflrdY;

	dB[Row*(iOutROIWidth) + Col] = (unsigned char)((double) iPix0
		+ dDiffY * (double) (iPixW - iPix0)	
		+ dDiffX * ((double) iPixDiff10	
		+ dDiffY * (double) (iPixWP1 - iPixW - iPixDiff10)) 
		+ 0.5); // for round up 

	//		}
	//	} //ix
	//	pbOutBuf += iOutSpan;		/* Next line in the output buffer */
	//} // iy	//// Identify the row and column of the Pd element to work on
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
