#include "PanelAligner.h"

PanelAligner::PanelAligner(void)
{
	_pOverlapManager = NULL;
	_pMosaics = NULL;
}

PanelAligner::~PanelAligner(void)
{
	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pMosaics != NULL)
		delete [] _pMosaics;
}

// Set panel
bool PanelAligner::SetPanel(MosaicSet* pSet)
{
	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pMosaics != NULL)
		delete [] _pMosaics;

	_pSet = pSet;

	_iNumIllumination = _pSet->GetNumMosaicLayers();
	_pMosaics = new MosaicImage[_iNumIllumination];

	unsigned int iImWidth  = _pSet->GetImageWidthInPixels();
	unsigned int iImHeight = _pSet->GetImageHeightInPixels();
	unsigned int iImStride = _pSet->GetImageStrideInPixels();
	
	// Create moasic images
	unsigned int i, j, iCam, iTrig;
	for(i=0; i<_iNumIllumination; i++)
	{
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		unsigned int iNumTrigs = pLayer->GetNumberOfTriggers();
		unsigned int iNumCams = pLayer->GetNumberOfCameras();
		bool bUseCad = false; // Need work

		_pMosaics[i].Config(i, iNumCams, iNumTrigs, iImWidth, iImHeight, iImStride, bUseCad);

		for(iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(iCam=0; iCam<iNumCams; iCam++)
			{
				MosaicTile* pTile = pLayer->GetTile(iCam, iTrig);
				ImgTransform t; // Need work
				_pMosaics[i].SetImageTransforms(t, iCam, iTrig);
			}
		}
	}

	for(i=0; i<_iNumIllumination; i++)
		for(j=0; j<_iNumIllumination; j++)
			_pCorrelationFlags[i][j] = *_pSet->GetCorrelationFlags(i, j);

	// Create Overlap manager
	DRect rect;	// need work
	_pOverlapManager = new OverlapManager(_pMosaics, _pCorrelationFlags, _iNumIllumination, NULL, rect); // nee work

	return(true);
}

bool PanelAligner::AddImage(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iColIndex,
	unsigned char* pcBuf)
{
	_pMosaics[iLayerIndex].AddImageBuffer(pcBuf, iColIndex, iTrigIndex);
	_pOverlapManager->DoAlignmentForFov(iLayerIndex, iTrigIndex, iColIndex);

	return(true);
}