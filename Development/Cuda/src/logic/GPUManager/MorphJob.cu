// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "GPUManager.h"
#include "GPUStream.h"
//#include "../MosaicDataModel/MorphJob.h"

// includes, project
#include <cutil.h>
#include <cufft.h>

//typedef struct {
//   float r;
//   float i;
//} complexf;

__constant__ double coeffs[3][3];
//__constant__ float acurve[256];
//__constant__ float bcurve[256];
//__constant__ float dKernel[10];
//__constant__ float dKernel2coef[5];
//__constant__ float dKernel4coef[10];

// includes, kernels
#include <MorphJob_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

//extern "C"
//void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

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

namespace CyberGPU {

	bool CanMapHostMemory()
	{
		struct cudaDeviceProp prop;

		cudaGetDeviceProperties(&prop, 0);
		if (prop.canMapHostMemory) return true;
		else return false;

	}

	bool CudaBufferRegister(unsigned char *ptr, size_t size)
	{
		cudaError_t error = cudaHostRegister( ptr, size, cudaHostRegisterPortable);
		//cudaError_t error = cudaHostRegister( ptr, size, cudaHostRegisterMapped);
 
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

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
CyberGPU::CGPUJob::GPUJobStatus GPUImageMorph( CyberGPU::GPUStream *jobStream,
	unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]) 
{
	//char str[128];
	//MorphJob* temp = (MorphJob*)(jobStream->GPUJob());

	unsigned int thePhase = jobStream->Phase();
	jobStream->Phase(thePhase+1);

	switch (thePhase)
	{
	case 0:

		//sprintf_s(str, "Job %d; Phase %d-0;", temp->OrdinalNumber(), thePhase);
		//jobStream->_pGPUJobManager->LogTimeStamp(str);

		//LARGE_INTEGER timestamp;
		///*assert(*/::QueryPerformanceCounter(&timestamp)/*)*/;

		CopyBufferToDeviceMatrix(jobStream->StdInBuffer(), pInBuf, jobStream->Stream());

		//sprintf_s(str, "Job %d; Phase %d; Xfer time", temp->OrdinalNumber(), thePhase);
		//jobStream->_pGPUJobManager->DeltaTimeStamp(str, timestamp);

		// Copy coefficients to device constant memory
		cudaMemcpyToSymbolAsync(coeffs, dInvTrans, sizeof(dInvTrans[0])*3, 0, cudaMemcpyHostToDevice, *jobStream->Stream());

		//for (int j=0; j<1/*8*/; ++j)
		{

			// Setup the execution configuration
			dim3 threads(TILE_WIDTH, 12/*TILE_WIDTH*/);
			dim3 grid(((iOutROIWidth - 1) / threads.x) + 1, ((iOutROIHeight - 1) / threads.y) + 1);

  			// Launch the device computation threads!
			ImageMorphKernel<<< grid, threads, 0, *jobStream->Stream()>>>
				(jobStream->StdInBuffer().elements, jobStream->StdOutBuffer().elements, iInSpan,
				iInHeight, iInWidth,
				iOutROIWidth, iOutROIHeight, iOutSpan, iOutROIStartX, iOutROIStartY);
		}

		return CyberGPU::CGPUJob::GPUJobStatus::ACTIVE;

	case 1:
		//CopyDeviceMatrixToHost(context->B, context->Bd, jobStream->Stream());

		/*cudaError_t */cudaMemcpy2D  ( pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY,  
				iOutSpan,
				jobStream->StdOutBuffer().elements,
				iOutROIWidth,
				iOutROIWidth,
				iOutROIHeight,
				cudaMemcpyDeviceToHost );

		cudaEventCreate(jobStream->PhaseEvent());

		cudaEventRecord(*jobStream->PhaseEvent(), *jobStream->Stream());

		return CyberGPU::CGPUJob::GPUJobStatus::ACTIVE;

	case 2:
		//cudaError_t result = cudaEventQuery(context->phaseEvent);
		//if (result != cudaSuccess)
		//{
		//	if (result == cudaErrorNotReady)
		//	{
		//		//sprintf_s(str, "Job %d; Phase %d; cudaErrorNotReady", temp->OrdinalNumber(), thePhase);
		//		//jobStream->_pGPUJobManager->LogTimeStamp(str);
		//	}
		//	else
		//	{
		//		sprintf_s(str, "Job %d; Phase %d; cudaError %d;", temp->OrdinalNumber(), thePhase, result);
		//		//jobStream->_pGPUJobManager->LogTimeStamp(str);
		//	}

		//	// maintain current phase to continue to check CopyDeviceMatrixToHost event for completion
		//	jobStream->Phase(thePhase);

		//	return CGPUJob::GPUJobStatus::WAITING;
		//}

		cudaEventSynchronize(*jobStream->PhaseEvent()/*context->phaseEvent*/); // wait on CopyDeviceMatrixToHost event not used

		// copy morphed FOV image to panel image buffer
		//CopyHostMatrixToBuffer(pOutBuf + iOutROIStartX + iOutSpan * iOutROIStartY, context->B, iOutSpan);

		cudaEventDestroy(*jobStream->PhaseEvent()/*context->phaseEvent*/);

		return CyberGPU::CGPUJob::GPUJobStatus::COMPLETED;
	}

	return CyberGPU::CGPUJob::GPUJobStatus::COMPLETED;
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
