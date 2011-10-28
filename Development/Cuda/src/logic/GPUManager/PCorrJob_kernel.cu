/*
 * Copyright 2011 CyberOptics Corporation.  All rights reserved.
 */

#ifndef _UTILITIES_KERNEL_H_
#define _UTILITIES_KERNEL_H_

#include <stdio.h>

#define TILE_WIDTH 16
#define TILE_16 16 // must be 16
#define TILE_16_HALF 8 // must be 8


__global__ void ApplyEqualizationKernel(unsigned char* dA, unsigned char* dB, complexf* dZ,
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

__global__ void DecimHorizontalKernelInside(complexf* dIn, complexf* dOut,
	int			columns, // input matrix columns
	int			rows, // input and output matrix rows
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimx,
	unsigned int decimx_mask ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	complexf temp;
	__shared__ complexf shared[TILE_16][2*TILE_16];

	// Identify the row and column of the Pd element to work on
	int tx = threadIdx.x; int ty = threadIdx.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x + TILE_16_HALF;
	
	if (row >= rows || col >= columns-TILE_16_HALF) return;

	// put outside pixels in shared memory
	int offset = 0;
	if (tx >= TILE_16_HALF) offset = TILE_16;
	shared[ty][tx+offset] = dIn[row * instride + col-TILE_16_HALF+offset];

	//put inside pixels real value in shared memory
	shared[ty][tx + TILE_16_HALF].r = dIn[row * instride + col].r;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	unsigned int pixel = tx / decimx;
	unsigned int index = tx & decimx_mask;

	int position = pixel*decimx+TILE_16_HALF;
	if (decimx == 4) position += 1;

	temp.r = 0;
	for (int i=0; i<decimx*2; i+=decimx)
	{
		float coef = dKernel[i+index];
		temp.r += coef*(shared[ty][position+i+index+1].r + shared[ty][position-i-index].r);
	}
	shared[ty][tx+TILE_16_HALF].i = temp.r;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	if (index < decimx/2)
	{
		int i = decimx*2+index;
		float coef = dKernel[i];
		temp.r = coef*(shared[ty][position+i+1].r + shared[ty][position-i].r) +
			shared[ty][tx+TILE_16_HALF].i + shared[ty][tx+decimx/2+TILE_16_HALF].i;
	}
	shared[ty][tx+TILE_16_HALF].i = temp.r;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	if (decimx == 4) temp.r += shared[ty][tx+1+TILE_16_HALF].i;

	//put inside pixels imaginary value in shared memory
	shared[ty][tx + TILE_16_HALF].i = dIn[row * instride + col].i;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	temp.i = 0;
	for (int i=0; i<decimx*2; i+=decimx)
	{
		float coef = dKernel[i+index];
		temp.i += coef*(shared[ty][position+i+index+1].i + shared[ty][position-i-index].i);
	}
	shared[ty][tx+TILE_16_HALF].r = temp.i;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	if (index < decimx/2)
	{
		int i = decimx*2+index;
		float coef = dKernel[i];
		temp.i = coef*(shared[ty][position+i+1].i + shared[ty][position-i].i) +
			shared[ty][tx+TILE_16_HALF].r + shared[ty][tx+decimx/2+TILE_16_HALF].r;
	}
	shared[ty][tx+TILE_16_HALF].r = temp.i;

	// Synchronize to make sure the row is loaded
    __syncthreads();

	if (decimx == 4) temp.i += shared[ty][tx+1+TILE_16_HALF].r;

    // Synchronize to make sure shared memory loaded
    __syncthreads();

	if (index == 0 )
	{
		dOut[row * outstride + col/decimx].r = temp.r;
		dOut[row * outstride + col/decimx].i = temp.i;
	}
}

__global__ void DecimHorizontalKernel(complexf* dIn, complexf* dOut,
	int			columns, // input matrix columns
	int			rows, // input and output matrix rows
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimx,
	int			kernelsize ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	//__shared__ complexf shared[MAX_ROW_SIZE+16];
	extern __shared__ complexf shared[];

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
		float coef = dKernel[i];
		temp.r += coef/*dKernel[i]*/*(shared[position+i+1].r + shared[position-i].r);
		temp.i += coef/*dKernel[i]*/*(shared[position+i+1].i + shared[position-i].i);
		//float coef = tex2D(CyberGPU::texKernel, (float)i, (float)0);
		//temp.r += coef*(shared[position+i+1].r + shared[position-i].r);
		//temp.i += coef*(shared[position+i+1].i + shared[position-i].i);
	}

	//dOut[row * outstride + col].r = temp.r;
	//dOut[row * outstride + col].i = temp.i;
	dOut[col * rows + row].r = temp.r;
	dOut[col * rows + row].i = temp.i;
}

__global__ void DecimVerticalKernel(complexf* dIn, complexf* dOut,
	int			rows, // input matrix rows
	int			instride, // input matrix stride
	int			outstride, // output matrix stride
	int			decimy,
	int			kernelsize ) // 2*decimx + decimx/2; // 2->5, 4->10
{
	extern __shared__ complexf shared[];

	// Identify the row and column of the Pd element to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
//	if (row >= height || col >= columns) return;

	int decim2y = decimy*2;

	for (int i=0; i<decimy; ++i)
	{
		//shared[decim2y+row*decimy+i] = dIn[(row*decimy+i) * instride + col];
		shared[decim2y+row*decimy+i] = dIn[ col * rows + (row*decimy+i)];
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
	extern __shared__ complexf shared[];

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
	extern __shared__ complexf shared[];

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
			//complexf *fptr = &dOut[crow*width + ccol];
			//fptr->r = value.r;
			//fptr->i = -value.i;
			dOut[crow*width + ccol].r = value.r;
			dOut[crow*width + ccol].i = -value.i;
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
