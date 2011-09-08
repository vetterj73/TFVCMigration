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
#include "../MosaicDataModel/MorphJob.h"

// includes, project
#include <cutil.h>

__constant__ double coeffs[3][3];

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned char* elements;
} ByteMatrix;

// includes, kernels
#include <Utilities_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer, cudaStream_t *stream);
void CopyDeviceMatrixToHost(ByteMatrix MHost, ByteMatrix Mdevice, cudaStream_t *stream);

//ByteMatrix AllocateZeroDeviceMatrix(int width, int height);
//ByteMatrix AllocateByteMatrix(int width, int height);
ByteMatrix AllocateDeviceMatrix(int width, int height);
void ResizeDeviceMatrix(ByteMatrix *Mdevice, int width, int height);
ByteMatrix AllocateHostMatrix(int width, int height);
void ResizeHostMatrix(ByteMatrix *Mhost, int width, int height);
void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer);
void CopyDeviceMatrixToBuffer(ByteMatrix Mdevice, unsigned char* buffer, int hostSpan);
void CopyDeviceMatrixToHost(ByteMatrix MHost, ByteMatrix Mdevice);
void CopyHostMatrixToBuffer(unsigned char* buffer, ByteMatrix Hdevice, int hostSpan);
void FreeDeviceMatrix(ByteMatrix* M);
void FreeMatrix(ByteMatrix* M);
void FreeHostMatrix(ByteMatrix* M);


void ConvolutionOnDevice(const ByteMatrix A, const ByteMatrix B, ByteMatrix C);

class ImageMorphContext
{
	public:
		ByteMatrix Ad;
		ByteMatrix Bd;
		ByteMatrix B;

		cudaEvent_t phaseEvent;
};
 
bool CudaBufferRegister(unsigned char *ptr, size_t size)
{
	cudaError_t error = cudaHostRegister( ptr, size, cudaHostRegisterPortable);   
 
	if (error != cudaSuccess)
		return false;
	
	return true;
}
bool CudaBufferUnregister(unsigned char *ptr)
{
	cudaError_t error = cudaHostUnregister(ptr);   

	if (error != cudaSuccess)
		return false;
	
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
CyberJob::GPUJob::GPUJobStatus GPUImageMorph( CyberJob::GPUJobStream *jobStream,
	unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]) 
{
//int main(int argc, char** argv) {

	MorphJob* temp = (MorphJob*)(jobStream->GPUJob());
	char str[128];

	unsigned int thePhase = jobStream->Phase();
	jobStream->Phase(thePhase+1);

	ImageMorphContext *context = (ImageMorphContext*)jobStream->Context();
	if (context == NULL)
	{
		context = new ImageMorphContext();
		jobStream->Context(context);

		//sprintf_s(str, "Job %d; Allocate;", temp->OrdinalNumber());
		//jobStream->_pGPUJobManager->LogTimeStamp(str);

		context->Ad = AllocateDeviceMatrix(iInSpan, iInHeight);
		context->B = AllocateHostMatrix(iOutROIWidth*2, iOutROIHeight*2);
		context->Bd = AllocateDeviceMatrix(iOutROIWidth*2, iOutROIHeight*2);
	}

	switch (thePhase)
	{
	case 0:

		//sprintf_s(str, "Job %d; Phase %d-0;", temp->OrdinalNumber(), thePhase);
		//jobStream->_pGPUJobManager->LogTimeStamp(str);

		// setup job context

		ResizeDeviceMatrix(&context->Ad, iInSpan, iInHeight);
		ResizeHostMatrix(&context->B, iOutROIWidth, iOutROIHeight);
		ResizeDeviceMatrix(&context->Bd, iOutROIWidth, iOutROIHeight);

		//LARGE_INTEGER timestamp;
		///*assert(*/::QueryPerformanceCounter(&timestamp)/*)*/;

		CopyBufferToDeviceMatrix(context->Ad, pInBuf, jobStream->Stream());

		//sprintf_s(str, "Job %d; Phase %d; Xfer time", temp->OrdinalNumber(), thePhase);
		//jobStream->_pGPUJobManager->DeltaTimeStamp(str, timestamp);

		// Copy coefficients to device constant memory
		cudaMemcpyToSymbolAsync(coeffs, dInvTrans, sizeof(dInvTrans[0])*3, 0, cudaMemcpyHostToDevice, *jobStream->Stream());

		//for (int j=0; j<1/*8*/; ++j)
		{

			// Setup the execution configuration
			dim3 threads(TILE_WIDTH, 12/*TILE_WIDTH*/);
			dim3 grid(((context->Bd.width - 1) / threads.x) + 1, ((context->Bd.height - 1) / threads.y) + 1);

  			// Launch the device computation threads!
			ConvolutionKernel<<< grid, threads, 0, *jobStream->Stream()>>>
				(context->Ad.elements, context->Bd.elements, context->Ad.width, context->Ad.height, iInWidth,
				context->Bd.width, context->Bd.height, iOutSpan, iOutROIStartX, iOutROIStartY);
		}

		return GPUJob::GPUJobStatus::ACTIVE;

	case 1:
		if (context == NULL) return GPUJob::GPUJobStatus::COMPLETED; // error, complete job

		CopyDeviceMatrixToHost(context->B, context->Bd, jobStream->Stream());

		cudaEventCreate(&context->phaseEvent);

		cudaEventRecord(context->phaseEvent);

		return GPUJob::GPUJobStatus::ACTIVE;

	case 2:
		if (context == NULL) return GPUJob::GPUJobStatus::COMPLETED; // error, complete job

		cudaError_t result = cudaEventQuery(context->phaseEvent);

		if (result != cudaSuccess)
		{
			if (result == cudaErrorNotReady)
			{
				//sprintf_s(str, "Job %d; Phase %d; cudaErrorNotReady", temp->OrdinalNumber(), thePhase);
				//jobStream->_pGPUJobManager->LogTimeStamp(str);
			}
			else
			{
				sprintf_s(str, "Job %d; Phase %d; cudaError %d;", temp->OrdinalNumber(), thePhase, result);
				jobStream->_pGPUJobManager->LogTimeStamp(str);
			}

			// maintain current phase to continue to check CopyDeviceMatrixToHost event for completion
			jobStream->Phase(thePhase);

			return GPUJob::GPUJobStatus::WAITING;
		}

		//cudaEventSynchronize(context->phaseEvent); // wait on CopyDeviceMatrixToHost event not used

		// copy morphed FOV image to panel image buffer
		CopyHostMatrixToBuffer(pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY, context->B, iOutSpan);

		cudaEventDestroy(context->phaseEvent);

		return GPUJob::GPUJobStatus::COMPLETED;
	}

	return GPUJob::GPUJobStatus::COMPLETED;
}

void ImageMorphCudaDelete(CyberJob::GPUJobStream *jobStream)
{
	ImageMorphContext *context = (ImageMorphContext*)jobStream->Context();
	if (context != NULL)
	{
		cudaFree(context->Ad.elements);
		cudaFree(context->Bd.elements);
		//cudaFreeHost(context->A.elements);
		//FreeHostMatrix(&context->B);
		cudaFreeHost(context->B.elements);
	}
}

// Allocate a device matrix of same size as M.
ByteMatrix AllocateDeviceMatrix(int width, int height)
{
	ByteMatrix Mdevice;

    Mdevice.width = width;
    Mdevice.height = height;
    Mdevice.size = width * height * sizeof(unsigned char);
    Mdevice.elements = NULL;

    cudaMalloc((void**)&Mdevice.elements, Mdevice.size);

	return Mdevice;
}

void ResizeDeviceMatrix(ByteMatrix *Mdevice, int width, int height)
{
	if (width*height > Mdevice->size)
	{
		if (Mdevice->elements != NULL) cudaFree(Mdevice->elements);
		Mdevice->elements = NULL;
		cudaMalloc( &Mdevice->elements, width*height*sizeof(unsigned char));
		Mdevice->size = width*height;
	}
    Mdevice->width = width;
    Mdevice->height = height;
}

// Allocate a host matrix of dimensions height*width
ByteMatrix AllocateHostMatrix(int width, int height)
{
	ByteMatrix Mhost;

    Mhost.width = width;
    Mhost.height = height;
    Mhost.size = width * height * sizeof(unsigned char);

	Mhost.elements = NULL;
	cudaMallocHost( &Mhost.elements, Mhost.size*sizeof(unsigned char));
	//Mhost.elements = (unsigned char*) malloc(Mhost.size*sizeof(unsigned char));

	return Mhost;
}

void ResizeHostMatrix(ByteMatrix *Mhost, int width, int height)
{
	if (width*height > Mhost->size)
	{
		if (Mhost->elements != NULL)
		{
			 cudaError_t error = cudaFreeHost(Mhost->elements);
			 //delete Mhost.elements;
		}
		Mhost->elements = NULL;
		cudaMallocHost( &Mhost->elements, width*height*sizeof(unsigned char));
		//Mhost.elements = (unsigned char*) malloc(Mhost.size*sizeof(unsigned char));
		Mhost->size = width*height;
	}
    Mhost->width = width;
    Mhost->height = height;
}

//// Allocate a host matrix of dimensions height*width
//ByteMatrix AllocateByteMatrix(int width, int height)
//{
//    ByteMatrix M;
//    M.width = M.pitch = width;
//    M.height = height;
//    int size = M.width * M.height;
//
//	M.elements = NULL;
//	M.elements = (unsigned char*) malloc(size*sizeof(unsigned char));
//
//    return M;
//}
//
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

void CopyDeviceMatrixToHost(ByteMatrix Mhost, ByteMatrix Mdevice, cudaStream_t *stream)
{
    int Hsize = Mhost.width * Mhost.height * sizeof(unsigned char);
    int Dsize = Mdevice.width * Mdevice.height * sizeof(unsigned char);
	int size = (Dsize > Hsize) ? Hsize : Dsize ;

	cudaMemcpyAsync(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost, *stream);
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
void FreeHostMatrix(ByteMatrix* M)
{
    cudaFreeHost(M->elements);
    M->elements = NULL;
}
