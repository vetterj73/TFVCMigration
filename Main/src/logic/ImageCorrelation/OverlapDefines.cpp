#include "OverlapDefines.h"

#pragma region Overlap class

Overlap::Overlap()
{
	_bValid = false;
}

Overlap::Overlap(
		Image* pImg1, 
		Image* pImg2,
		DRect validRect,
		OverlapType type,
		Image* pMaskImg)
{
	config(pImg1, pImg2, validRect, type, pMaskImg);
}

Overlap::Overlap(const Overlap& overlap) 
{
	*this = overlap;
}

void Overlap::operator=(const Overlap& b)
{
	_pImg1 = b._pImg1; 
	_pImg2 = b._pImg2;
	_pMaskImg = b._pMaskImg;

	_validRect = b._validRect;

	_type = b._type;

	_bValid = b._bValid;
	_coarsePair = b._coarsePair;
}

void Overlap::config(
	Image* pImg1, 
	Image* pImg2,
	DRect validRect,
	OverlapType type,		
	Image* pMaskImg)
{
	_pImg1 = pImg1; 
	_pImg2 = pImg2;
	_pMaskImg = _pMaskImg;

	_validRect = validRect;

	_type = type;

	_bValid = CalCoarseCorrPair();
}

// Calculate the coarse correlation pair
bool Overlap::CalCoarseCorrPair()
{
	// Get image's bound box in world space
	DRect rectWorld1 = _pImg1->GetBoundBoxInWorld();
	DRect rectWorld2 = _pImg2->GetBoundBoxInWorld();

	// Calculate the overlap of image in world space
	DRect overlapWorld;
	overlapWorld.xMin = rectWorld1.xMin>rectWorld2.xMin ? rectWorld1.xMin : rectWorld2.xMin;
	overlapWorld.xMax = rectWorld1.xMax<rectWorld2.xMax ? rectWorld1.xMax : rectWorld2.xMax;
	
	overlapWorld.yMin = rectWorld1.yMin>rectWorld2.yMin ? rectWorld1.yMin : rectWorld2.yMin;
	overlapWorld.yMax = rectWorld1.yMax<rectWorld2.yMax ? rectWorld1.yMax : rectWorld2.yMax;

	if(overlapWorld.xMin < _validRect.xMin) overlapWorld.xMin = _validRect.xMin;
	if(overlapWorld.xMax > _validRect.xMax) overlapWorld.xMax = _validRect.xMax;
	if(overlapWorld.yMin < _validRect.yMin) overlapWorld.yMin = _validRect.yMin;
	if(overlapWorld.yMax > _validRect.yMax) overlapWorld.yMax = _validRect.yMax;

		// validation check
	if(overlapWorld.xMin>=overlapWorld.xMax ||
		overlapWorld.yMin>=overlapWorld.yMax)
		return(false);

	// Calculate Roi1 and Roi2
		// Rows and colums
	unsigned int iRows, iCols;
	if(_pImg1->PixelSizeX() > _pImg2->PixelSizeX())
		iRows = (unsigned int)((overlapWorld.xMax - overlapWorld.xMin)/_pImg1->PixelSizeX());
	else
		iRows = (unsigned int)((overlapWorld.xMax - overlapWorld.xMin)/_pImg2->PixelSizeX());

	if(_pImg1->PixelSizeY() > _pImg2->PixelSizeY())
		iCols = (unsigned int)((overlapWorld.yMax - overlapWorld.yMin)/_pImg1->PixelSizeY());
	else
		iCols = (unsigned int)((overlapWorld.yMax - overlapWorld.yMin)/_pImg2->PixelSizeY());

	UIRect roi1, roi2;
	double dFirstRow1, dFirstCol1, dFirstRow2, dFirstCol2;
	_pImg1->WorldToImage(overlapWorld.xMin, overlapWorld.yMin, &dFirstRow1, &dFirstCol1);
	_pImg2->WorldToImage(overlapWorld.xMin, overlapWorld.yMin, &dFirstRow2, &dFirstCol2);
		
		// Roi1 and Roi2
	roi1.FirstRow = (unsigned int)dFirstRow1;
	roi1.LastRow = roi1.FirstRow+iRows-1;
	roi1.FirstColumn = (unsigned int)dFirstCol1;
	roi1.LastColumn = roi1.FirstColumn+iCols-1;

	roi2.FirstRow = (unsigned int)dFirstRow2;
	roi2.LastRow = roi2.FirstRow+iRows-1;
	roi2.FirstColumn = (unsigned int)dFirstCol2;
	roi2.LastColumn = roi2.FirstColumn+iCols-1;

		// Validation check 
	if(roi1.FirstRow < 0 ||
		roi1.LastRow > _pImg1->Rows()-1 ||
		roi1.FirstColumn < 0 ||
		roi1.LastColumn > _pImg1->Columns()-1)
	{
		//G_LOG_0_ERROR("CameraOverlap failed BoundsCheck");
	}

	if(roi2.FirstRow < 0 ||
		roi2.LastRow > _pImg2->Rows()-1 ||
		roi2.FirstColumn < 0 ||
		roi2.LastColumn > _pImg2->Columns()-1)
	{
		//G_LOG_0_ERROR("CameraOverlap failed BoundsCheck");
	}

	unsigned int iDecim = 4;
	if(roi1.Columns()>1000 && roi1.Rows()>1000)
		iDecim = 8;

	unsigned int iColSearchExpansion = 50;
	unsigned int iRowSearchExpansion = 75;

	CorrelationPair coarsePair(
		_pImg1, _pImg2, 
		roi1, pair<unsigned int, unsigned int>(roi2.FirstRow, roi2.FirstColumn),
		iDecim, iColSearchExpansion, iRowSearchExpansion,
		_type, _pMaskImg);
	_coarsePair = coarsePair;

	return(true);
}

bool Overlap::DoIt()
{

	return(true);
}

#pragma endregion

#pragma region FovFovOverlap

FovFovOverlap::FovFovOverlap(
	MosaicImage*	pMosaic1,
	MosaicImage*	pMosaic2,
	pair<unsigned int, unsigned int> ImgPos1,
	pair<unsigned int, unsigned int> ImgPos2,
	DRect validRect,
	bool bHasMask)
{
	_pMosaic1 = pMosaic1;
	_pMosaic2 = pMosaic2;
	_ImgPos1 = ImgPos1;
	_ImgPos2 = ImgPos2;
	_bHasMask = bHasMask;

	Image* pImg1 = _pMosaic1->GetImagePtr(ImgPos1.first, ImgPos1.second);
	Image* pImg2 = _pMosaic1->GetImagePtr(ImgPos2.first, ImgPos2.second);

	config(pImg1, pImg2, validRect, Fov_To_Fov, NULL);
}

#pragma endregion 

#pragma region CadFovOverlap class

CadFovOverlap::CadFovOverlap(
	MosaicImage* pMosaic,
	pair<unsigned int, unsigned int> ImgPos,
	Image* pCadImg,
	DRect validRect)
{
	_pMosaic = pMosaic;
	_ImgPos = ImgPos;
	_pCadImg = pCadImg;

	Image* pImg1 = _pMosaic->GetImagePtr(ImgPos.first, ImgPos.second);

	config(pImg1, _pCadImg, validRect, Cad_To_Fov);
}

#pragma endregion

#pragma region FidFovOverlap class

FidFovOverlap::FidFovOverlap(
	MosaicImage*	pMosaic,
	pair<unsigned int, unsigned int> ImgPos,
	Image* pFidImg,
	double dCenterX,
	double dCenterY,
	DRect validRect)
{
	_pMosaic = pMosaic;
	_ImgPos = ImgPos;
	_pFidImg = pFidImg;

	_dCenterX = dCenterX;
	_dCenterY = dCenterY;

	Image* pImg1 = _pMosaic->GetImagePtr(ImgPos.first, ImgPos.second);

	config(pImg1, _pFidImg, validRect, Fid_To_Fov);
}

#pragma endregion



