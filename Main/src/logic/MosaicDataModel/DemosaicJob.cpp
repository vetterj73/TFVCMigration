#include "StdAfx.h"
#include "DemosaicJob.h"
#include "MosaicLayer.h"

namespace MosaicDM 
{

DemosaicJob::DemosaicJob(
	MosaicSet* pSet, 
	unsigned char *pBuffer, 
	unsigned int iLayerIndex, 
	unsigned int iCameraIndex, 
	unsigned int iTriggerIndex)
{
	_pSet = pSet; 
	_pBuffer = pBuffer; 
	_iLayerIndex = iLayerIndex; 
	_iCameraIndex = iCameraIndex; 
	_iTriggerIndex = iTriggerIndex;
}

void DemosaicJob::Run()
{
	MosaicLayer* pLayer = _pSet->GetLayer(_iLayerIndex);
	if(pLayer == NULL)
		return;

	if(!pLayer->AddRawImage(_pBuffer, _iCameraIndex, _iTriggerIndex))
		return;

	_pSet->FireImageAdded(_iLayerIndex, _iCameraIndex, _iTriggerIndex);
}

}