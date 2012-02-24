#include "VsNgcWrapper.h"
#include "VsNgcAlignment.h"


VsNgcWrapper::VsNgcWrapper(void)
{
	_pAlignNgc = new VsNgcAlignment();
}


VsNgcWrapper::~VsNgcWrapper(void)
{
	delete _pAlignNgc;
}

bool VsNgcWrapper::Align(NgcParams params, NgcResults* pResults)
{
	return(_pAlignNgc->Align(params, pResults));
}
