/*
 * Copyright 2011 CyberOptics Corporation.  All rights reserved.
 */

#ifndef _UTILITIES_KERNEL_H_
#define _UTILITIES_KERNEL_H_

#include <stdio.h>

#define TILE_WIDTH 16

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

__global__ void ApplyEqualizationKernel(unsigned char* dA, unsigned char* dB, complexf* dZ,
   //float apal[], float bpal[],
   int width, int height,
   int instride, int outstride)
{
	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || col >= width) return;

    dZ[row * outstride + col].r = acurve[dA[row * instride + col]];
    dZ[row * outstride + col].i = bcurve[dB[row * instride + col]];
}

__global__ void DecimHorizontalKernel(complexf* dIn, complexf* dOut,
	int			columns, // input matrix columns
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimx,
	int			kernelsize ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	__shared__ complexf shared[1024];

	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
//	if (row >= height || col >= columns) return;

	int decim2x = decimx*2;
	for (int i=0; i<decimx; ++i)
	{
		shared[decim2x+col*decimx+i] = dIn[row * instride + col*decimx+i];
	}

	// Synchronize to make sure the row is loaded
    __syncthreads();

	int index = 0;
	if (col < decim2x*2) // setup wrapped image data for edge calculations
	{
		if (col >= decim2x)
		{
			index = columns;
		}
		shared[col+index] = shared[col+columns-index];
	}

    // Synchronize to make sure shared memory loaded
    __syncthreads();

	complexf temp;
	temp.r = temp.i = 0.0;

	int position = col*decimx+decim2x;
	if (decimx == 4) position += 1;

	for (int i=0; i<kernelsize; ++i)
	{
		temp.r += dKernel[i]*(shared[position+i+1].r + shared[position-i].r);
		temp.i += dKernel[i]*(shared[position+i+1].i + shared[position-i].i);
	}

	dOut[row * outstride + col].r = temp.r;
	dOut[row * outstride + col].i = temp.i;

	//if (row == 1)
	//{
	//	dOut[col].r = shared[col+8].r;
	//	dOut[col].i = shared[col+8].i;
	//}
}

__global__ void DecimVerticalKernel(complexf* dIn, complexf* dOut,
	int			rows, // input matrix rows
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimy,
	int			kernelsize ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	__shared__ complexf shared[1024];

	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
//	if (row >= height || col >= columns) return;

	int decim2y = decimy*2;

	for (int i=0; i<decimy; ++i)
	{
		shared[decim2y+row*decimy+i] = dIn[(row*decimy+i) * instride + col];
	}

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	int index = 0;
	if (row < decim2y*2) // setup wrapped image data for edge calculations
	{
		if (row >= decim2y)
		{
			index = rows;
		}
		shared[row+index] = shared[row+rows-index];
	}

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	complexf temp;
	temp.r = temp.i = 0.0;

	int position = row*decimy+decim2y;
	if (decimy == 4) position += 1;

	for (int i=0; i<kernelsize; ++i)
	{
		temp.r += dKernel[i]*(shared[position+i+1].r + shared[position-i].r);
		temp.i += dKernel[i]*(shared[position+i+1].i + shared[position-i].i);
	}

	dOut[row * outstride + col].r = temp.r;
	dOut[row * outstride + col].i = temp.i;
}

__global__ void CrossFilterVerticalKernel(complexf* dZ,
	int width, int height, int cw, int stride)
{
	__shared__ complexf shared[1024];

	// Identify the row and column of the Pd element to work on
	int i = threadIdx.y; // column index for shared memory workspace
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x - cw;
	
	if (col < 0) col += width;

	// Load the column from device memory to shared memory; each thread loads
	// one element of each column. The wrap locations are loaded by the first
	// and last threads
	shared[i+1] = dZ[row * stride + col];
	if ((row|col) == 0) shared[1].r = shared[1].i = 0.0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	//if (i == 0 || i == height-1)
	//	shared[((height+1)*i)/(height-1)] = dZ[(height-1-row)*stride + col];
	if (i == 0) shared[0] = shared[height];
	if (i == height-1) shared[height+1] = shared[1];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	dZ[row * stride + col].r = 0.5*shared[i+1].r - 0.25*(shared[i].r + shared[i+2].r);
	dZ[row * stride + col].i = 0.5*shared[i+1].i - 0.25*(shared[i].i + shared[i+2].i);
}

__global__ void CrossFilterHorizontalKernel(complexf* dZ,
	int width, int height, int cw, int stride)
{
	__shared__ complexf shared[1024];

	// Identify the row and column of the Pd element to work on
	int i = threadIdx.x; // row index for shared memory workspace
	int row = blockIdx.y * blockDim.y + threadIdx.y - cw;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < 0) row += height;

	// Load the row from device memory to shared memory; each thread loads
	// one element of each row. The wrap locations are loaded by the first
	// and last threads
	shared[i+1] = dZ[row * stride + col];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	//if (i == 0 || i == width-1)
	//	shared[((width+1)*i)/(width-1)] = dZ[row * stride + width-1-col];
	if (i == 0) shared[0] = shared[width];
	if (i == width-1) shared[width+1] = shared[1];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	dZ[row * stride + col].r = 0.5*shared[i+1].r - 0.25*(shared[i].r + shared[i+2].r);
	dZ[row * stride + col].i = 0.5*shared[i+1].i - 0.25*(shared[i].i + shared[i+2].i);
}

__global__ void ConjugateMultKernel(complexf* dIn, complexf* dOut,
	float* dSum, int width, int height, int stride)
{

	__shared__ float shared[512];

	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	shared[tid] = 0.0; // initialize magnitude for summation in case location is outside image bounds

	if (row <= height/2 && col < width)
	{

		int crow = -row;
		if (crow < 0) crow += height;
		int ccol = -col;
		if (ccol < 0) ccol += width;

		complexf e = dIn[row*width + col];
		complexf f = dIn[crow*width + ccol];

		complexf value;
		value.r = 0.5 * (e.i*f.r + e.r*f.i);
		value.i = 0.25 * (f.r*f.r + f.i*f.i - e.r*e.r - e.i*e.i);

		float mag = sqrtf(hypotf(value.r, value.i));

		// put mag in shared memory for summation
		shared[tid] = mag;
		if (crow == row) shared[tid] *= 0.5;
		
		if (mag != 0.0) mag = 1.0/mag;

		value.r *= mag;
		value.i *= mag;

		dOut[row*width + col] = value;

		if (crow != row) 
		{
			// store complex conjugate
			complexf *fptr = &dOut[crow*width + ccol];
			fptr->r = value.r;
			fptr->i = -value.i;
		}
	}

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	// sum individual magnitudes for block total
	for (unsigned int s=blockDim.x*blockDim.y/*sharedDim*//2; s>0; s>>=1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid+s];
		}
		__syncthreads();
	}

	// output block magnitude summation for later grid summation
	if (tid == 0) dSum[blockIdx.y*gridDim.x + blockIdx.x] = shared[0];
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
