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
	unsigned int i, j, kx, ky;
	for(i=0; i<_iNumIllumination; i++)
	{
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		unsigned int iNumFovY = pLayer->GetNumberOfTriggers();
		unsigned int iNumFovX = pLayer->GetNumberOfCameras();
		bool bUseCad = false; // Need work

		_pMosaics[i].Config(i, iNumFovX, iNumFovY, iImWidth, iImHeight, iImStride, bUseCad);

		for(ky=0; ky<iNumFovY; ky++)
		{
			for(kx=0; kx<iNumFovX; kx++)
			{
				MosaicTile* pTile = pLayer->GetTile(kx, ky);
				ImgTransform t; // Need work
				_pMosaics[i].SetImageTransforms(t, kx, ky);
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
	unsigned int iRowIndex, 
	unsigned int iColIndex,
	unsigned char* pcBuf)
{
	_pMosaics[iLayerIndex].AddImageBuffer(pcBuf, iColIndex, iRowIndex);
	_pOverlapManager->DoAlignmentForFov(iLayerIndex, iRowIndex, iColIndex);

	return(true);
}