#pragma once
#include "GPUManager.h"
#include "rtypes.h"
#include <cutil.h>
#include <cufft.h>

using namespace CyberGPU;

class CyberGPU::GPUStream;

//typedef struct {
//   float r;
//   float i;
//} complexf;
//
//void ClearMorphJobStream(CyberGPU::GPUStream *jobStream);

class PCorrJob : public CyberGPU::CGPUJob
{
public:
	//PCorrJob(int ncd, int nrd, complexf *z, unsigned int ordinal);
	PCorrJob(
		int ncols,			/* Number of columns in images */
		int nrows,			/* Number of rows in images */
		unsigned char a[],	/* Pointer to first image  */
		unsigned char b[],	/* Pointer to second image */
		int astride, int bstride,
		float apal[], float bpal[],
		int decimx, int decimy,
		int ncd, int nrd, complexf * z, float *sum, int crosswindow);
	~PCorrJob();

	void Run();
	CyberGPU::CGPUJob::GPUJobStatus GPURun(CyberGPU::GPUStream *jobStream);

	//unsigned int NumberOfStreams() { return 3; }

	unsigned int OrdinalNumber() { return _ordinal; }

protected:
	int _ncols;
	int _nrows;
	unsigned char *_a;
	unsigned char *_b;
	int _astride;
	int _bstride;
	float *_apal;
	float *_bpal;
	int _decimx;
	int _decimy;
	int _cw;

	int _ncd;
	int _nrd;
	complexf *_z;

	float *_sum;
	float *_work;

	cufftHandle _plan;

	unsigned int _ordinal;
};

CyberGPU::CGPUJob::GPUJobStatus GPUPCorr( CyberGPU::GPUStream *jobStream,
	int ncols,			/* Number of columns in images */
	int nrows,			/* Number of rows in images */
	unsigned char a[],	/* Pointer to first image  */
	unsigned char b[],	/* Pointer to second image */
	int astride, int bstride,
	float apal[], float bpal[],
	/*int columns, int rows,*/ int decimx, int decimy,
	int ncd, int nrd, complexf *z, float *work, int crosswindow, cufftHandle plan); 

