#pragma once

#include "MosaicImage.h"
#include "OverlapManager.h"
#include "MosaicSet.h"

using namespace MosaicDM;

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool SetPanel(MosaicSet set);
	bool AddImage(unsigned int iLayerIndex, unsigned int iRowIndex, unsigned int iColIndex);
};

