/*
 * Copyright 2011 CyberOptics Corporation.  All rights reserved.
 */

#ifndef _UTILITIES_KERNEL_H_
#define _UTILITIES_KERNEL_H_

#include <stdio.h>

#define TILE_WIDTH 16
#define MAX_ROW_SIZE 2000 // max shared memory is 0x4000 bytes
#define MAX_COL_SIZE 2000 // max shared memory is 0x4000 bytes


__global__ void ImageMorphKernel(unsigned char* dA, unsigned char* dB, unsigned int iInSpan, unsigned int iInHeight,
	unsigned int iInWidth, unsigned int iOutROIWidth, unsigned int iOutROIHeight, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY)
{

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by * 12 + ty;
	int Col = bx * 16 + tx;


	if (Row >= iOutROIHeight || Col >= iOutROIWidth) return;

	// some local variable
	float dX, dY;
	int iflrdX, iflrdY;
	int iPix0, iPix1, iPixW, iPixWP1;
	int iOffset;
	int iPixDiff10;
	float dDiffX, dDiffY, dIX, dIY;
	//float dT01__dIY_T02, dT11__dIY_T12/*, dT21__dIY_T22*/;
 

	//for (iY=iOutROIStartY; iY<iOutROIStartY+iOutROIHeight; ++iY) 
	//{

	dIY = (float) (Row + iOutROIStartY);
	//dT01__dIY_T02 = coeffs[0][1] * dIY + coeffs[0][2];
	//dT11__dIY_T12 = coeffs[1][1] * dIY + coeffs[1][2];

	//for (iX=iOutROIStartX; iX<iOutROIStartX+iOutROIWidth; ++iX) 
	//{

	dIX = (float) (Col + iOutROIStartX);
	//dX = coeffs[0][0]*dIX + dT01__dIY_T02;
	//dY = coeffs[1][0]*dIX + dT11__dIY_T12;
	dX = coeffs[0][0]*dIX + coeffs[0][1] * dIY + coeffs[0][2];
	dY = coeffs[1][0]*dIX + coeffs[1][1] * dIY + coeffs[1][2];

	/* Check if back projection is outside of the input image range. Note that
		2x2 interpolation touches the pixel to the right and below right,
		so right and bottom checks are pulled in a pixel. */
	//if ((dX < 0) | (dY < 0) |
	//	(dX >= iInWidth-1) | (dY >= iInHeight-1)) 
	//{
	//	dB[Col + Row * iOutROIWidth] = 0x0;	/* Clipped */
	//	return;
	//}

	/* Compute fractional differences */
	iflrdX = (int) /*__float2int_rz*/(dX);	/* Compared to int-to-float, float-to-int */
	iflrdY = (int) /*__float2int_rz*/(dY);	/*   much more costly */
	dDiffX = dX - (float)/*__int2float_rn*/(iflrdX);
	dDiffY = dY - (float)/*__int2float_rn*/(iflrdY);
			    
	/* Compute offset to input pixel at (dX,dY) */
	iOffset = iflrdX + iflrdY * iInSpan;

	iPix0   = (int) dA[iOffset]; /* The 2x2 neighborhood used */
	iPix1   = (int) dA[iOffset+1];
	iPixW   = (int) dA[iOffset+iInSpan];
	iPixWP1 = (int) dA[iOffset+iInSpan+1];

	iPixDiff10 = iPix1 - iPix0; /* Used twice, so compute once */

	dB[Row*(iOutROIWidth) + Col] = (unsigned char)((float) /*__int2float_rn*/(iPix0)
		+ dDiffY * (float) /*__int2float_rn*/(iPixW - iPix0)	
		+ dDiffX/*(dX - (float)iflrdX)*/ * ((float) /*__int2float_rn*/(iPixDiff10)	
		+ dDiffY * (float) /*__int2float_rn*/(iPixWP1 - iPixW - iPixDiff10)) 
		+ 0.5); // for round up 

	//		}
	//	} //ix
	//	pbOutBuf += iOutSpan;		/* Next line in the output buffer */
	//} // iy	//// Identify the row and column of the Pd element to work on
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
