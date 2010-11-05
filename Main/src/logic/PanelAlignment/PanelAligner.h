#pragma once

#include "MosaicImage.h"
#include "OverlapManager.h"
#include "MosaicSet.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"

using namespace MosaicDM;

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool SetPanel(MosaicSet* pSet);
	bool AddImage(
		unsigned int iLayerIndex, 
		unsigned int iRowIndex, 
		unsigned int iColIndex,
		unsigned char* pcBuf);

private:
	MosaicSet* _pSet;
	OverlapManager* _pOverlapManager;
	MosaicImage* _pMosaics;
	unsigned int _iNumIllumination;
};

