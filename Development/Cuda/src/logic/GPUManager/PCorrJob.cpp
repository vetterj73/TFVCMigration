#include "PCorrJob.h"

bool CudaBufferRegister(unsigned char *ptr, size_t size);
bool CudaBufferUnregister(unsigned char *ptr);
//static bool bCSInitialized = false;
//static 	::CRITICAL_SECTION _cs;

PCorrJob::PCorrJob(
	int ncols,			/* Number of columns in images */
	int nrows,			/* Number of rows in images */
	unsigned char a[],	/* Pointer to first image  */
	unsigned char b[],	/* Pointer to second image */
	int astride, int bstride,
	float apal[], float bpal[],
	int decimx, int decimy,
	int ncd, int nrd, complexf * z, float *sum, int crosswindow)
{
	_ncols = ncols;
	_nrows = nrows;
	_a = a;
	_b = b;
	_astride = astride;
	_bstride = bstride;
	_apal = apal;
	_bpal = bpal;
	_decimx = decimx;
	_decimy = decimy;
	_ncd = ncd;
	_nrd = nrd;
	_z = z;
	_cw = crosswindow;
	_sum = sum; 
	_work = new float[1024]; 
	//_ordinal = ordinal;

	cufftResult results = cufftPlan2d( &_plan, _nrd, _ncd, CUFFT_C2C);
	if (results != CUFFT_SUCCESS)
	{
		results = (cufftResult)0; // code to break on
		// log error
	}
	if (_plan == 0xffffffff/* || _plan == 0*/)
	{
		results = (cufftResult)0; // code to break on
		// log error
		//cufftResult results = cufftPlan2d( &_plan, _nrd, _ncd, CUFFT_C2C);
		//if (results != CUFFT_SUCCESS)
		//{
		//	results = (cufftResult)0; // code to break on
		//	// log error
		//}
	}

	//if (!bCSInitialized)
	//{
	//	bCSInitialized = true;
	//	::InitializeCriticalSection(&_cs);
	//}
	//::EnterCriticalSection(&_cs);
	//bCSInitialized = true;
	//CudaBufferRegister(_a, _ncols*_nrows*sizeof(unsigned char));
	//CudaBufferRegister(_b, _ncols*_nrows*sizeof(unsigned char));
	//CudaBufferRegister((unsigned char*)_z, _ncd*_nrd*sizeof(complexf));
	//::LeaveCriticalSection(&_cs);
	////::DeleteCriticalSection(&cs);
}
//PCorrJob::PCorrJob(int ncd, int nrd, complexf *z, unsigned int ordinal) 
//{
//	_ncd = ncd;
//	_nrd = nrd;
//	_z = z;
//	_ordinal = ordinal;
//
//	CudaBufferRegister((unsigned char*)_z, _ncd*_nrd*sizeof(complexf));
//}

PCorrJob::~PCorrJob()
{
	delete _work;

	cufftResult results = cufftDestroy( _plan );
	if (results != CUFFT_SUCCESS)
	{
		results = (cufftResult)0; // code to break on
		// log error
	}
	//CudaBufferUnregister((unsigned char*)_a);
	//CudaBufferUnregister((unsigned char*)_b);
	//CudaBufferUnregister((unsigned char*)_z);
}

void PCorrJob::Run()
{
	////printf_s("Thread execution: oridinal - %ld;\n", _ordinal);

	//if(_pStitched !=NULL && _rect.IsValid())
	//	_pStitched->MorphFrom(_pFOV, _rect);
}

#define TILE_WIDTH 16

CyberGPU::CGPUJob::GPUJobStatus PCorrJob::GPURun(CyberGPU::GPUStream *jobStream)
{
	CyberGPU::CGPUJob::GPUJobStatus results = CyberGPU::CGPUJob::GPUJobStatus::COMPLETED; // true = conversion complete
	
	results = GPUPCorr( jobStream,
		_ncols,			/* Number of columns in images */
		_nrows,			/* Number of rows in images */
		_a,	/* Pointer to first image  */
		_b,	/* Pointer to second image */
		_astride,
		_bstride,
		_apal, _bpal,
		_decimx, _decimy,
		_ncd, _nrd, _z, _work, _cw, _plan);

	if (results == CyberGPU::CGPUJob::GPUJobStatus::COMPLETED)
	{
		float value = 0.0;
		int total = (((_ncd - 1) / TILE_WIDTH) + 1) * (((_nrd / 2) / TILE_WIDTH) + 1);
		for (int i=0; i<total; ++i)
		{
			value += _work[i];
		}
		*_sum = value;
	}

	return results;
}
