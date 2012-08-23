#pragma once

#include "SIMAPI.h"
#include "MosaicSet.h"
using namespace MosaicDM; 

class ConfigMosaicSet
{
public:
	static void MosaicSetDefaultConfiguration(MosaicSet* pSet, bool bMaskForDiffDevices);
	static int TranslateTrigger(SIMAPI::CSIMFrame* pFrame);

protected:
	static int AddDeviceToMosaic(MosaicSet* pSet, SIMAPI::ISIMDevice *pDevice, int iDeviceIndex);
	static void SetDefaultCorrelationFlags(MosaicSet* pSet, bool bMaskForDiffDevices);
};

