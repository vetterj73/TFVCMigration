/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* 2D Convolution: C = A (*) B, A is the 5x5 kernel matrix, B is the image matrix.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "GPUJobManager.h"
#include "GPUJobStream.h"

// includes, project
#include <cutil.h>

__constant__ double coeffs[3][3];

// includes, kernels
#include <Utilities_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

ByteMatrix AllocateDeviceMatrix(int width, int height);
ByteMatrix AllocateZeroDeviceMatrix(int width, int height);
ByteMatrix AllocateByteMatrix(int width, int height);
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer, cudaStream_t *stream);
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer);
void CopyDeviceMatrixToBuffer(ByteMatrix Mdevice, unsigned char* buffer, int hostSpan);
void CopyDeviceMatrixToHost(ByteMatrix MHost, ByteMatrix Mdevice);
void CopyHostMatrixToBuffer(unsigned char* buffer, ByteMatrix Hdevice, int hostSpan);
void FreeDeviceMatrix(ByteMatrix* M);
void FreeMatrix(ByteMatrix* M);
void FreeByteMatrix(ByteMatrix* M);

void ConvolutionOnDevice(const ByteMatrix A, const ByteMatrix B, ByteMatrix C);

 
static clock_t startXferToTick = 0;//the tick for when we first create an instance
static clock_t startXferFromTick = 0;//the tick for when we first create an instance
static clock_t startKrnlTick = 0;//the tick for when we first create an instance
static clock_t totalXferToTick = 0;//the tick for when we first create an instance
static clock_t totalXferFromTick = 0;//the tick for when we first create an instance
static clock_t totalKrnlTick = 0;//the tick for when we first create an instance

void PrintTicks()
{
	printf_s("\tXfer To ticks - %ld; Kernel ticks - %ld; Xfer From ticks - %ld;\n",totalXferToTick, totalKrnlTick, totalXferFromTick);
}

class ImageMorphContext
{
	public:
		ByteMatrix Ad;
		ByteMatrix Bd;
		ByteMatrix B;
};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
bool GPUImageMorph(
	int phase, CyberJob::GPUJobStream *jobStream,
	unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]) 
{
//int main(int argc, char** argv) {

	//ImageMorphContext *context = (ImageMorphContext*)jobStream->Context();

	//switch (phase)
	//{
	//case 0:
	//	 // setup job context
	//	context = new ImageMorphContext();
	//	jobStream->Context((void *)context);

	//	context->Ad = AllocateDeviceMatrix(iInSpan, iInHeight);
	//	CopyBufferToDeviceMatrix(context->Ad, pInBuf, jobStream->Stream());

	//	break;

	//case 1:
	//	if (context == NULL) return true; // error, complete job

	//	break;
	//}

	startXferToTick = clock();//Obtain current tick

	// Allocate and initialize input device matrices
    ByteMatrix Ad = AllocateDeviceMatrix(iInSpan, iInHeight);
	CopyBufferToDeviceMatrix(Ad, pInBuf);
	
	// Allocate and clear output device matrices
    ByteMatrix Bd = AllocateDeviceMatrix(iOutROIWidth, iOutROIHeight);
    //int size = Bd.width * Bd.height * sizeof(unsigned char);
    //cudaMemset(Bd.elements, 0, size);
    ByteMatrix B = AllocateByteMatrix(iOutROIWidth, iOutROIHeight);

	// Copy coefficients to device constant memory
	cudaMemcpyToSymbol(coeffs, dInvTrans, sizeof(dInvTrans[0])*3, 0);

	totalXferToTick += clock() - startXferToTick;//calculate the difference in ticks

	startKrnlTick = clock();//Obtain current tick

	for (int j=0; j<1/*8*/; ++j)
	{

	// Setup the execution configuration
    dim3 threads(TILE_WIDTH, 12/*TILE_WIDTH*/);
    dim3 grid(((Bd.width - 1) / threads.x) + 1, ((Bd.height - 1) / threads.y) + 1);

  	// Launch the device computation threads!
	ConvolutionKernel<<< grid, threads >>>(Ad.elements, Bd.elements, Ad.width, Ad.height, iInWidth,
		Bd.width, Bd.height, iOutSpan, iOutROIStartX, iOutROIStartY);

	int size = Ad.width * Ad.height * sizeof(unsigned char);
	cudaMemcpy(pOutBuf, Bd.elements, 16, cudaMemcpyDeviceToHost);
	}

	totalKrnlTick += clock() - startKrnlTick;//calculate the difference in ticks

	startXferFromTick = clock();//Obtain current tick

	//CopyDeviceMatrixToBuffer(Bd, pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY, iOutSpan);
	CopyDeviceMatrixToHost(B, Bd);
	CopyHostMatrixToBuffer(pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY, B, iOutSpan);

	// Free matrices
    cudaFree(Ad.elements);
    cudaFree(Bd.elements);
	FreeByteMatrix(&B);

	totalXferFromTick += clock() - startXferFromTick;//calculate the difference in ticks

	return true;
}

// Allocate a device matrix of same size as M.
ByteMatrix AllocateDeviceMatrix(int width, int height)
{
    ByteMatrix Mdevice;
    Mdevice.width = Mdevice.pitch = width;
    Mdevice.height = height;
    Mdevice.elements = NULL;

    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a host matrix of dimensions height*width
ByteMatrix AllocateByteMatrix(int width, int height)
{
    ByteMatrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;

	M.elements = NULL;
	M.elements = (unsigned char*) malloc(size*sizeof(unsigned char));

    return M;
}


// Copy a host matrix to a device matrix.
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer, cudaStream_t *stream)
{
    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
    cudaMemcpyAsync(Mdevice.elements, buffer, size, cudaMemcpyHostToDevice, *stream);
}
// Copy a host matrix to a device matrix.
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer)
{
    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
    cudaMemcpy(Mdevice.elements, buffer, size, cudaMemcpyHostToDevice);
}

void CopyDeviceMatrixToHost(ByteMatrix Mhost, ByteMatrix Mdevice)
{
    int Hsize = Mhost.width * Mhost.height * sizeof(unsigned char);
    int Dsize = Mdevice.width * Mdevice.height * sizeof(unsigned char);
	int size = (Dsize > Hsize) ? Hsize : Dsize ;

	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

void CopyHostMatrixToBuffer(unsigned char* buffer, ByteMatrix Hdevice, int hostSpan)
{
	for (int i=0; i<Hdevice.height; ++i)
	{
		memcpy(buffer+i*hostSpan, Hdevice.elements+i*Hdevice.width, Hdevice.width);
	}
}

// Free a device matrix.
void FreeDeviceMatrix(ByteMatrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host ByteMatrix
void FreeByteMatrix(ByteMatrix* M)
{
    free(M->elements);
    M->elements = NULL;
}
