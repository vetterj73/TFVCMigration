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
	unsigned int iTriggerIndex,
	bool bTrigAlignment)
{
	_pSet = pSet; 
	_pBuffer = pBuffer; 
	_iLayerIndex = iLayerIndex; 
	_iCameraIndex = iCameraIndex; 
	_iTriggerIndex = iTriggerIndex;

	_bTrigAlignment = bTrigAlignment;
}

void DemosaicJob::Run()
{
	MosaicLayer* pLayer = _pSet->GetLayer(_iLayerIndex);
	if(pLayer == NULL)
		return;

	if(!pLayer->AddRawImage(_pBuffer, _iCameraIndex, _iTriggerIndex))
		return;
	/*
	Image* pImage = pLayer->GetImage(_iTriggerIndex, _iCameraIndex);
	ImgTransform trans1;
	Image tempImage1(
		pImage->Columns(), 
		pImage->Rows(), 
		pImage->PixelRowStride(),
		1,
		trans1,
		trans1,
		false,
		pImage->GetBuffer());

	memcpy( tempImage1.GetBuffer(), pImage->GetBuffer(), pImage->BufferSizeInBytes()/3);	

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "C:\\Temp\\YCrCb\\L%d_T%d_C%d.bmp", 
		pLayer->Index(), _iTriggerIndex, _iCameraIndex);
	s.append(cTemp);
	tempImage1.Save(s);*/

	if(_bTrigAlignment)
		_pSet->FireImageAdded(_iLayerIndex, _iCameraIndex, _iTriggerIndex);
}

}