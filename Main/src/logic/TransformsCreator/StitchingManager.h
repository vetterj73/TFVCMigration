#pragma once
#include "MosaicSet.h"
#include "Panel.h"
using namespace MosaicDM;

class Image;
class StitchingManager
{
public:
	StitchingManager(MosaicSet *pMosaicSet, Panel *pPanel);
	~StitchingManager(void);

	void CreateStitchingImage(unsigned int iIllumIndex, Image* pPanelImage) const;
	static void CreateStitchingImage(MosaicLayer* pMosaic, Image* pPanelImage);
	void SaveStitchingImages(string name, unsigned int iNum, bool bCreateColorImg=false);

private:
	MosaicSet * _pMosaicSet;
	Panel *_pPanel;
};

