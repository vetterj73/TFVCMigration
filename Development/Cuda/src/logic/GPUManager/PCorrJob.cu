// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "GPUManager.h"
#include "GPUStream.h"

// includes, project
#include <cutil.h>
#include <cufft.h>

typedef struct {
   float r;
   float i;
} complexf;

//__constant__ double coeffs[3][3];
__constant__ float acurve[256];
__constant__ float bcurve[256];
__constant__ float dKernel[10];

// includes, kernels
#include <PCorrJob_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

//extern "C"
//void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

//void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer, cudaStream_t *stream);
//void CopyDeviceMatrixToHost(ByteMatrix MHost, ByteMatrix Mdevice, cudaStream_t *stream);
//
////ByteMatrix AllocateZeroDeviceMatrix(int width, int height);
////ByteMatrix AllocateByteMatrix(int width, int height);
//ByteMatrix AllocateDeviceMatrix(int width, int height);
//void ResizeDeviceMatrix(ByteMatrix *Mdevice, int width, int height);
//ByteMatrix AllocateHostMatrix(int width, int height);
//void ResizeHostMatrix(ByteMatrix *Mhost, int width, int height);
//void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer);
//void CopyDeviceMatrixToBuffer(ByteMatrix Mdevice, unsigned char* buffer, int hostSpan);
//void CopyDeviceMatrixToHost(ByteMatrix MHost, ByteMatrix Mdevice);
//void CopyHostMatrixToBuffer(unsigned char* buffer, ByteMatrix Hdevice, int hostSpan);
//void FreeDeviceMatrix(ByteMatrix* M);
//void FreeMatrix(ByteMatrix* M);
//void FreeHostMatrix(ByteMatrix* M);

//bool CanMapHostMemory()
//{
//	struct cudaDeviceProp prop;
//
//	cudaGetDeviceProperties(&prop, 0);
//	if (prop.canMapHostMemory) return true;
//	else return false;
//
//}
//bool CudaBufferRegister(unsigned char *ptr, size_t size)
//{
//	cudaError_t error = cudaHostRegister( ptr, size, cudaHostRegisterPortable);
//	//cudaError_t error = cudaHostRegister( ptr, size, cudaHostRegisterMapped);
// 
//	if (error != cudaSuccess)
//		return false;
//	
//	return true;
//}
//bool CudaBufferUnregister(unsigned char *ptr)
//{
//	cudaError_t error = cudaHostUnregister(ptr);   
//
//	if (error != cudaSuccess)
//		return false;
//	
//	return true;
//}
//
//


static const float kernel2coef[] = {
	+6.913469554E-01,   /*  0.5 */
	+3.963233672E-01,   /*  1.5 */
	+5.312047567E-02,   /*  2.5 */
	-1.083921551E-01,   /*  3.5 */
	-8.447095014E-02    /*  4.5 */
};
static const float kernel4coef[] = {
	+3.658160711E-01,   /*  0.5 */
	+3.215795429E-01,   /*  1.5 */
	+2.437565548E-01,   /*  2.5 */
	+1.502194757E-01,   /*  3.5 */
	+6.090930440E-02,   /*  4.5 */
	-8.232239494E-03,   /*  5.5 */
	-4.850434474E-02,   /*  6.5 */
	-6.070450677E-02,   /*  7.5 */
	-5.144587161E-02,   /*  8.5 */
	-3.450495203E-02    /*  9.5 */
};

void GPUPCorrExit()
{
	cudaError_t temp = cudaThreadExit();
}

CyberGPU::CGPUJob::GPUJobStatus GPUPCorr( CyberGPU::GPUStream *jobStream,
	int ncols,			/* Number of columns in images */
	int nrows,			/* Number of rows in images */
	unsigned char a[],	/* Pointer to first image  */
	unsigned char b[],	/* Pointer to second image */
	int astride,
	int bstride,
	float apal[], float bpal[],
	/*int columns, int rows,*/ int decimx, int decimy,
	int ncd, int nrd, complexf * z, float * work, int crosswindow, cufftHandle plan) 
{
	//cufftHandle plan;
	cufftResult results;

	unsigned int thePhase = jobStream->Phase();
	jobStream->Phase(thePhase+1);

	int size = ncd * nrd * sizeof(complexf);

	switch (thePhase)
	{
	case 0:
		//result = cufftPlan2d( jobStream->Plan(), nrd, ncd, CUFFT_C2C);
		results = cufftSetStream(plan, *jobStream->Stream());
		if (results != CUFFT_SUCCESS)
		{
			results = (cufftResult)0; // code to break on
			// log error
		}
		//CopyBufferToDeviceMatrix(jobStream->StdInBuffer(), pInBuf, jobStream->Stream());

		////sprintf_s(str, "Job %d; Phase %d; Xfer time", temp->OrdinalNumber(), thePhase);
		////jobStream->_pGPUJobManager->DeltaTimeStamp(str, timestamp);

		{

			// copy A to device stdin beginning
			//cudaMemcpyAsync(jobStream->StdInBuffer().elements, a/*(unsigned char*)z*/,
			//	imagesize, cudaMemcpyHostToDevice, *jobStream->Stream());

			cudaError_t error2D = cudaMemcpy2DAsync(
				jobStream->StdInBuffer().elements,  
				ncols,
				a,
				astride,
				ncols,
				nrows,
				cudaMemcpyHostToDevice,
				 *jobStream->Stream());
			if (error2D != cudaSuccess)
			{
				error2D = (cudaError_t)0; // code to break on
				// log error
			}
			

			// copy B to device stdin with offset
			int alignment = 0x20;
			int offset = ncols * nrows * sizeof(unsigned char);
			if (offset % alignment ) offset += alignment - (offset%alignment);
			//cudaMemcpyAsync(jobStream->StdInBuffer().elements+offset,
			//	b, imagesize, cudaMemcpyHostToDevice, *jobStream->Stream());
			error2D = cudaMemcpy2DAsync(
				jobStream->StdInBuffer().elements+offset,  
				ncols,
				b,
				bstride,
				ncols,
				nrows,
				cudaMemcpyHostToDevice,
				 *jobStream->Stream());
			if (error2D != cudaSuccess)
			{
				error2D = (cudaError_t)0; // code to break on
				// log error
			}

			// copy apal to device constant memory
			cudaMemcpyToSymbolAsync(acurve, apal, 256*sizeof(float), 0, cudaMemcpyHostToDevice, *jobStream->Stream());
			// copy bpal to device constant memory
			cudaMemcpyToSymbolAsync(bcurve, bpal, 256*sizeof(float), 0, cudaMemcpyHostToDevice, *jobStream->Stream());

			// Setup merge of a and b images into complex image
			dim3 threads(TILE_WIDTH, /*12*/TILE_WIDTH);
			dim3 grid(((ncols - 1) / threads.x) + 1, ((nrows - 1) / threads.y) + 1);

			ApplyEqualizationKernel<<< grid, threads, 0, *jobStream->Stream()>>>
				(jobStream->StdInBuffer().elements,
				jobStream->StdInBuffer().elements+offset,
				(complexf*)jobStream->StdOutBuffer().elements,
				ncols, nrows,
				ncols, ncols);

#ifdef __TEXTURE_COEF
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

			cudaArray* cuArray;
			cudaMallocArray(&cuArray, &channelDesc, 10, 1);
			
			// Set texture parameters
			CyberGPU::texKernel.addressMode[0] = cudaAddressModeClamp;
			CyberGPU::texKernel.addressMode[1] = cudaAddressModeClamp;
			CyberGPU::texKernel.filterMode = cudaFilterModePoint;
			CyberGPU::texKernel.normalized = false;
#endif

			bool decimate = true;
			unsigned int decimx_mask = 0;
			switch (decimx)
			{
			case 2:
				cudaMemcpyToSymbol(dKernel, kernel2coef, sizeof(kernel2coef), 0, cudaMemcpyHostToDevice);
				decimx_mask = 1;
#ifdef __TEXTURE_COEF
				cudaMemcpyToArray(cuArray, 0, 0, kernel2coef, sizeof(kernel2coef), cudaMemcpyHostToDevice);
#else
				cudaMemcpyToSymbol(dKernel, kernel2coef, sizeof(kernel2coef), 0, cudaMemcpyHostToDevice);
#endif
			
				break;
			case 4:
				decimx_mask = 3;
#ifdef __TEXTURE_COEF
				cudaMemcpyToArray(cuArray, 0, 0, kernel4coef, sizeof(kernel4coef), cudaMemcpyHostToDevice);
#else
				cudaMemcpyToSymbol(dKernel, kernel4coef, sizeof(kernel4coef), 0, cudaMemcpyHostToDevice);
#endif
				break;
			default:
				decimate = false;
				break;
			}

			if (decimate)
			{
#ifdef __TEXTURE_COEF
				// Bind the array to the texture reference
				cudaBindTextureToArray(CyberGPU::texKernel, cuArray, channelDesc);
#endif

				// !!! if ncol (the number of undecimated columns is greater than 500, shared memory is not big enough
				//     dynamically created shared memory can be passed as the third argument in a cuda call. This is
				//     one solution. Hard coded shared memory can be allocated inside the cuda call. This is the current
				//     solution to get 512*sizeof(complexf) bytes allocated for shared memory

				// Setup horizontal image decimation
				dim3 Hthreads(ncd, 1); // a block is a decimated row
				dim3 Hgrid(1, nrows); // grid is undecimated number of rows 

  				// Launch the device computation threads!
				DecimHorizontalKernel<<< Hgrid, Hthreads, (ncols+16)*sizeof(complexf), *jobStream->Stream()>>>
					((complexf*)jobStream->StdOutBuffer().elements,
					(complexf*)jobStream->StdInBuffer().elements,
					ncols, nrows, ncols, ncd, decimx, 2*decimx + decimx/2 ); // 2*decimx + decimx/2; // 2->5, 4->10

				//dim3 threads(TILE_16, TILE_16);
				//dim3 grid(((ncols - TILE_16 - 1) / threads.x) + 1, ((nrows - 1) / threads.y) + 1);

				//DecimHorizontalKernelInside<<< grid, threads, ncd, *jobStream->Stream()>>>
				//	((complexf*)jobStream->StdOutBuffer().elements,
				//	(complexf*)jobStream->StdInBuffer().elements,
				//	ncols, nrows, ncols, ncd, decimx, decimx_mask/*2*decimx + decimx/2*/ ); // 2*decimx + decimx/2; // 2->5, 4->10
			}

			if (decimate)
			{
				dim3 Tthreads(TILE_DIM, TILE_DIM);
				dim3 Tgrid(((ncd - 1) / Tthreads.x) + 1, ((nrows - 1) / Tthreads.y) + 1);

  				// Launch the device computation threads!
				TransposeMatrix<<< Tgrid, Tthreads, 0, *jobStream->Stream()>>>
					((complexf*)jobStream->StdInBuffer().elements,
					(complexf*)jobStream->StdOutBuffer().elements,
					ncd, nrows );
			}

			decimate = true;
			switch (decimy)
			{
			case 2:
				cudaMemcpyToSymbol(dKernel, kernel2coef, sizeof(kernel2coef), 0, cudaMemcpyHostToDevice);
				break;
			case 4:
				cudaMemcpyToSymbol(dKernel, kernel4coef, sizeof(kernel4coef), 0, cudaMemcpyHostToDevice);
				break;
			default:
				decimate = false;
				break;
			}

			if (decimate)
			{
				// Setup vertical image decimation
				dim3 Vthreads(1, nrd); // a block is a decimated column
				//dim3 Vgrid(ncols, 1); // grid is decimated number of columns 
				dim3 Vgrid(ncd, 1); // grid is decimated number of columns 

  				// Launch the device computation threads!
				DecimVerticalKernel<<< Vgrid, Vthreads, (nrows+16)*sizeof(complexf), *jobStream->Stream()>>>
					((complexf*)jobStream->StdOutBuffer().elements,
					(complexf*)jobStream->StdInBuffer().elements,
					//((complexf*)jobStream->StdInBuffer().elements,
					//(complexf*)jobStream->StdOutBuffer().elements,
					//nrows, ncols, ncols, decimy, 2*decimy + decimy/2 ); // 2*decimx + decimx/2; // 2->5, 4->10
					nrows, /*ncd*/nrows, ncd, decimy, 2*decimy + decimy/2 ); // 2*decimx + decimx/2; // 2->5, 4->10
			}
		}

		results = cufftExecC2C( plan, (cufftComplex*)jobStream->StdInBuffer().elements,
			(cufftComplex*)jobStream->StdInBuffer().elements, CUFFT_FORWARD);
		if (results != CUFFT_SUCCESS)
		{
			cudaError_t err = cudaGetLastError();
			// log error
		}

		{
			// Setup vertical circular convolution for CrossFilter
			dim3 Vthreads(1, nrd); // 
			dim3 Vgrid(crosswindow*2 + 1, 1);

  			// Launch the device computation threads!
			CrossFilterVerticalKernel<<< Vgrid, Vthreads, (nrd+2)*sizeof(complexf), *jobStream->Stream()>>>
				((complexf*)jobStream->StdInBuffer().elements,
				ncd, nrd, crosswindow, ncd);

			// Setup horizontal circular convolution for CrossFilter
			dim3 Hthreads(ncd, 1);
			dim3 Hgrid(1, crosswindow*2 + 1);

  			// Launch the device computation threads!
			CrossFilterHorizontalKernel<<< Hgrid, Hthreads, (ncd+2)*sizeof(complexf), *jobStream->Stream()>>>
				((complexf*)jobStream->StdInBuffer().elements,
				ncd, nrd, crosswindow, ncd);
		}

		{

			int alignment = 0x20;
			int offset = ncols * nrows * sizeof(complexf);
			if (offset % alignment ) offset += alignment - (offset%alignment);

			// Setup conjugate multiplication
			dim3 threads(TILE_WIDTH, /*12*/TILE_WIDTH);
			dim3 grid(((ncd - 1) / threads.x) + 1, ((nrd/2/* - 1*/) / threads.y) + 1);

  			// Launch the device computation threads!
			ConjugateMultKernel<<< grid, threads, 0, *jobStream->Stream()>>>
				((complexf*)jobStream->StdInBuffer().elements,
				(complexf*)jobStream->StdOutBuffer().elements,
				(float*)(jobStream->StdInBuffer().elements+offset),
				ncd, nrd, ncd);

		}

		results = cufftExecC2C( plan, (cufftComplex*)jobStream->StdOutBuffer().elements,
			(cufftComplex*)jobStream->StdOutBuffer().elements, CUFFT_INVERSE);
		if (results != CUFFT_SUCCESS)
		{
			cudaError_t err = cudaGetLastError();
			// log error
		}

		return CyberGPU::CGPUJob::GPUJobStatus::ACTIVE;

	case 1:
		{
			int alignment = 0x20;
			int offset = ncols * nrows * sizeof(complexf);
			if (offset % alignment ) offset += alignment - (offset%alignment);
			int total = (((ncd - 1) / TILE_WIDTH) + 1) * (((nrd/2/* - 1*/) / TILE_WIDTH) + 1);
			cudaError_t error = cudaMemcpyAsync(work, jobStream->StdInBuffer().elements+offset,
				total*sizeof(float), cudaMemcpyDeviceToHost, *jobStream->Stream());

			int imagesize = /*110*/ncd * /*378*/nrd * sizeof(complexf);
			//int imagesize = ncols * nrows * sizeof(complexf);
			//error = cudaMemcpyAsync(z, jobStream->StdInBuffer().elements,
			error = cudaMemcpyAsync(z, jobStream->StdOutBuffer().elements,
				imagesize, cudaMemcpyDeviceToHost, *jobStream->Stream());
		}

		//cudaEventDestroy(*jobStream->PhaseEvent());

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
//// Allocate a device matrix of same size as M.
//ByteMatrix AllocateDeviceMatrix(int width, int height)
//{
//	ByteMatrix Mdevice;
//
//    Mdevice.width = width;
//    Mdevice.height = height;
//    Mdevice.size = width * height * sizeof(unsigned char);
//    Mdevice.elements = NULL;
//
//    cudaMalloc((void**)&Mdevice.elements, Mdevice.size);
//
//	return Mdevice;
//}
//
//void ResizeDeviceMatrix(ByteMatrix *Mdevice, int width, int height)
//{
//	if (width*height > Mdevice->size)
//	{
//		if (Mdevice->elements != NULL) cudaFree(Mdevice->elements);
//		Mdevice->elements = NULL;
//		cudaMalloc( &Mdevice->elements, width*height*sizeof(unsigned char));
//		Mdevice->size = width*height;
//	}
//    Mdevice->width = width;
//    Mdevice->height = height;
//}
//
//// Allocate a host matrix of dimensions height*width
//ByteMatrix AllocateHostMatrix(int width, int height)
//{
//	ByteMatrix Mhost;
//
//    Mhost.width = width;
//    Mhost.height = height;
//    Mhost.size = width * height * sizeof(unsigned char);
//
//	Mhost.elements = NULL;
//	cudaMallocHost( &Mhost.elements, Mhost.size*sizeof(unsigned char));
//	//Mhost.elements = (unsigned char*) malloc(Mhost.size*sizeof(unsigned char));
//
//	return Mhost;
//}
//
//void ResizeHostMatrix(ByteMatrix *Mhost, int width, int height)
//{
//	if (width*height > Mhost->size)
//	{
//		if (Mhost->elements != NULL)
//		{
//			 cudaError_t error = cudaFreeHost(Mhost->elements);
//			 //delete Mhost.elements;
//		}
//		Mhost->elements = NULL;
//		cudaMallocHost( &Mhost->elements, width*height*sizeof(unsigned char));
//		//Mhost.elements = (unsigned char*) malloc(Mhost.size*sizeof(unsigned char));
//		Mhost->size = width*height;
//	}
//    Mhost->width = width;
//    Mhost->height = height;
//}
//
//// Copy a host matrix to a device matrix.
//void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer, cudaStream_t *stream)
//{
//    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
//    cudaMemcpyAsync(Mdevice.elements, buffer, size, cudaMemcpyHostToDevice, *stream);
//}
//// Copy a host matrix to a device matrix.
//void CopyBufferToDeviceMatrix(ByteMatrix Mdevice, unsigned char* buffer)
//{
//    int size = Mdevice.width * Mdevice.height * sizeof(unsigned char);
//    cudaMemcpy(Mdevice.elements, buffer, size, cudaMemcpyHostToDevice);
//}
//
//void CopyDeviceMatrixToHost(ByteMatrix Mhost, ByteMatrix Mdevice, cudaStream_t *stream)
//{
//    int Hsize = Mhost.width * Mhost.height * sizeof(unsigned char);
//    int Dsize = Mdevice.width * Mdevice.height * sizeof(unsigned char);
//	int size = (Dsize > Hsize) ? Hsize : Dsize ;
//
//	cudaMemcpyAsync(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost, *stream);
//}
//void CopyDeviceMatrixToHost(ByteMatrix Mhost, ByteMatrix Mdevice)
//{
//    int Hsize = Mhost.width * Mhost.height * sizeof(unsigned char);
//    int Dsize = Mdevice.width * Mdevice.height * sizeof(unsigned char);
//	int size = (Dsize > Hsize) ? Hsize : Dsize ;
//
//	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
//}
//
//void CopyHostMatrixToBuffer(unsigned char* buffer, ByteMatrix Hdevice, int hostSpan)
//{
//	for (int i=0; i<Hdevice.height; ++i)
//	{
//		memcpy(buffer+i*hostSpan, Hdevice.elements+i*Hdevice.width, Hdevice.width);
//	}
//}
//
//// Free a device matrix.
//void FreeDeviceMatrix(ByteMatrix* M)
//{
//    cudaFree(M->elements);
//    M->elements = NULL;
//}
//
//// Free a host ByteMatrix
//void FreeHostMatrix(ByteMatrix* M)
//{
//    cudaFreeHost(M->elements);
//    M->elements = NULL;
//}
