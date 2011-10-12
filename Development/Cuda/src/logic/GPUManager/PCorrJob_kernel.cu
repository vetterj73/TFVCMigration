/*
 * Copyright 2011 CyberOptics Corporation.  All rights reserved.
 */

#ifndef _UTILITIES_KERNEL_H_
#define _UTILITIES_KERNEL_H_

#include <stdio.h>

#define TILE_WIDTH 16
#define MAX_ROW_SIZE 2000 // max shared memory is 0x4000 bytes
#define MAX_COL_SIZE 2000 // max shared memory is 0x4000 bytes


__global__ void ApplyEqualizationKernel(unsigned char* dA, unsigned char* dB, complexf* dZ,
   int width, int height,
   int instride, int outstride)
{
	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || col >= width) return;

	//if (row <= height/2+240/*/2 /*&& col <= width/2*/)
	//{
		dZ[row * outstride + col].r = acurve[dA[row * instride + col]];
		dZ[row * outstride + col].i = bcurve[dB[row * instride + col]];
	//}
}

__global__ void DecimHorizontalKernel(complexf* dIn, complexf* dOut,
	int			columns, // input matrix columns
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimx,
	int			kernelsize ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	__shared__ complexf shared[MAX_ROW_SIZE+16];

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
	__shared__ complexf shared[MAX_COL_SIZE+16];

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
	__shared__ complexf shared[MAX_COL_SIZE+2];

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
	__shared__ complexf shared[MAX_ROW_SIZE+2];

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

	__shared__ float shared[1024];

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
