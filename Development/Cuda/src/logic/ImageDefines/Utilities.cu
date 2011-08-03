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
//Matrix AllocateDeviceMatrix(const Matrix M);
ByteMatrix AllocateZeroDeviceMatrix(int width, int height);
//Matrix AllocatePadDeviceMatrix(const Matrix M);
ByteMatrix AllocateMatrix(int height, int width, int init);
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer);
void CopyDeviceMatrixToBuffer(ByteMatrix Mdevice, unsigned char* buffer, int hostSpan);
//void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyPadToDeviceMatrix(ByteMatrix Mdevice, const ByteMatrix Mhost);
void CopyFromDeviceMatrix(ByteMatrix Mhost, const ByteMatrix Mdevice);
int ReadFile(ByteMatrix* M, char* file_name);
void WriteFile(ByteMatrix M, char* file_name);
void FreeDeviceMatrix(ByteMatrix* M);
void FreeMatrix(ByteMatrix* M);

void ConvolutionOnDevice(const ByteMatrix A, const ByteMatrix B, ByteMatrix C);

 
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
bool GPUImageMorph(unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]) 
{
//int main(int argc, char** argv) {

	// Allocate and initialize input device matrices
    ByteMatrix Ad = AllocateDeviceMatrix(iInSpan, iInHeight);
    CopyBufferToDeviceMatrix(Ad, pInBuf);
	
	// Allocate and clear output device matrices
    ByteMatrix Bd = AllocateDeviceMatrix(iOutROIWidth, iOutROIHeight);
    //int size = Bd.width * Bd.height * sizeof(unsigned char);
    //cudaMemset(Bd.elements, 0, size);

	// Copy coefficients to device constant memory
	cudaMemcpyToSymbol(coeffs, dInvTrans, sizeof(dInvTrans[0])*3, 0);

    // Setup the execution configuration
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(((Bd.width - 1) / threads.x) + 1, ((Bd.height - 1) / threads.y) + 1);

  	// Launch the device computation threads!
	ConvolutionKernel<<< grid, threads >>>(Ad.elements, Bd.elements, Ad.width, Ad.height, iInWidth,
		Bd.width, Bd.height, iOutSpan, iOutROIStartX, iOutROIStartY);

	CopyDeviceMatrixToBuffer(Bd, pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY, iOutSpan);

	// Free matrices
    cudaFree(Ad.elements);
    cudaFree(Bd.elements);

	return 0;
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

//// Allocate and clear a device matrix of same size as M.
//void ZeroDeviceMatrix(ByteMatrix Mdevice)
//{
//    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
//    cudaMemset(Mdevice.elements, 0, size);
//}

//// Allocate a device matrix of dimensions height*width
////	If init == 0, initialize to all zeroes.  
////	If init == 1, perform random initialization.
////  If init == 2, initialize matrix parameters, but do not allocate memory 
//ByteMatrix AllocateMatrix(int height, int width, int init)
//{
//    ByteMatrix M;
//    M.width = M.pitch = width;
//    M.height = height;
//    int size = M.width * M.height;
//    M.elements = NULL;
//    
//    // don't allocate memory on option 2
//    if(init == 2)
//		return M;
//		
//	M.elements = (unsigned char*) malloc(size*sizeof(char));
//
//	for(unsigned int i = 0; i < M.height * M.width; i++)
//	{
//		M.elements[i] = 0xff;
//		//(init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
//		//M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
//		//if(rand() % 2)
//		//	M.elements[i] = - M.elements[i];
//	}
//    return M;
//}	
//

// Copy a host matrix to a device matrix.
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer)
{
    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
    cudaMemcpy(Mdevice.elements, buffer, size, cudaMemcpyHostToDevice);
}

void CopyDeviceMatrixToBuffer(ByteMatrix Mdevice, unsigned char* buffer, int hostSpan)
{
	for (int i=0; i<Mdevice.height; ++i)
	{
		cudaMemcpy(buffer+i*hostSpan, Mdevice.elements+i*Mdevice.width, Mdevice.width, cudaMemcpyDeviceToHost);
	}
}

//// Copy a host matrix to a device matrix.
//void CopyPadToDeviceMatrix(ByteMatrix Mdevice, const ByteMatrix Mhost)
//{
//	if (Mdevice.height == Mhost.height && Mdevice.width == Mhost.width)
//	{
//		int size = Mhost.width * Mhost.height * sizeof(float);
//		Mdevice.height = Mhost.height;
//		Mdevice.width = Mhost.width;
//		Mdevice.pitch = Mhost.pitch;
//		cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
//						cudaMemcpyHostToDevice);
//	}
//	else
//	{
//		int pad = KERNEL_SIZE / 2;
//		//printf("pad-%d\n", pad);
//
//		for (int i=pad; i<Mdevice.height; ++i)
//		{
//			if ( i < Mdevice.height - pad)
//				cudaMemcpy(Mdevice.elements + i*Mdevice.width + pad, Mhost.elements + (i-pad)*Mhost.width, Mhost.width*sizeof(float), 
//					cudaMemcpyHostToDevice);
//		}
//	}
//}

//// Copy a device matrix to a host matrix.
//void CopyFromDeviceMatrix(ByteMatrix Mhost, const ByteMatrix Mdevice)
//{
//    int size = Mdevice.width * Mdevice.height * sizeof(float);
//    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
//					cudaMemcpyDeviceToHost);
//}
//
// Free a device matrix.
void FreeDeviceMatrix(ByteMatrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host ByteMatrix
void FreeMatrix(ByteMatrix* M)
{
    free(M->elements);
    M->elements = NULL;
}
