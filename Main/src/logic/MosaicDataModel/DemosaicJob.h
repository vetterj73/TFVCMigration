#pragma once
#include "jobthread.h"
#include "MosaicSet.h"

namespace MosaicDM 
{

class DemosaicJob :	public CyberJob::Job
{
public:
	DemosaicJob(
		MosaicSet* pSet, 
		unsigned char *pBuffer, 
		unsigned int iLayerIndex, 
		unsigned int iCameraIndex, 
		unsigned int iTriggerIndex);

	virtual void Run();

private:
	MosaicSet* _pSet; 
	unsigned char* _pBuffer; 
	unsigned int _iLayerIndex; 
	unsigned int _iCameraIndex; 
	unsigned int _iTriggerIndex;
};

}