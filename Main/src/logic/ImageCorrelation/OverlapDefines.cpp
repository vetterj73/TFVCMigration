#include "OverlapDefines.h"
#include "Logger.h"
#include "MosaicTile.h"
#include "EquationWeights.h"

#pragma region Overlap class

Overlap::Overlap()
{
	_bValid = false;
	_bProcessed = false;
	_bGood4Solver = true;
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

	_bApplyCorrSizeUpLimit = b._bApplyCorrSizeUpLimit;

	_bValid = b._bValid;
	_bProcessed = b._bProcessed;
	_bGood4Solver = b._bGood4Solver;

	//** Warning Coarse pair need to point to different parents,
	_coarsePair = b._coarsePair;
	// Override the parent point
	_coarsePair.SetOverlapPtr(this);
}

void Overlap::config(
	Image* pImg1, 
	Image* pImg2,
	DRect validRect,
	OverlapType type,
	bool bApplyCorrSizeUpLimit,
	Image* pMaskImg)
{
	_pImg1 = pImg1; 
	_pImg2 = pImg2;
	_pMaskImg = pMaskImg;

	_validRect = validRect;

	_type = type;

	_bApplyCorrSizeUpLimit = bApplyCorrSizeUpLimit;

	_bValid = CalCoarseCorrPair();
	_bProcessed = false;
	_bGood4Solver = true;

	if(_bValid)
	{
		_iColumns = _coarsePair.Columns();
		_iRows	  = _coarsePair.Rows();
	}
}

// Reset to status to before alignment
bool Overlap::Reset()
{
	// Clean fine correlation pair list
	_finePairList.clear();
	
	// Reset coarse correlation pair
	_coarsePair.Reset();

	// Reset processed flag 
	_bProcessed = false;
	_bGood4Solver = true;

	return(true);
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

	// Make sure the overlap is inside the panel area
	// For Fiducail and Fov overalp, this is not necessary.  
	if(_type != Fid_To_Fov) 
	{
		if(overlapWorld.xMin < _validRect.xMin) overlapWorld.xMin = _validRect.xMin;
		if(overlapWorld.xMax > _validRect.xMax) overlapWorld.xMax = _validRect.xMax;
		if(overlapWorld.yMin < _validRect.yMin) overlapWorld.yMin = _validRect.yMin;
		if(overlapWorld.yMax > _validRect.yMax) overlapWorld.yMax = _validRect.yMax;
	}
	
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

	if(dFirstRow1+0.5<0 || dFirstRow2+0.5<0 || dFirstCol1+0.5<0 || dFirstCol2+0.5<0)
	{
		LOG.FireLogEntry(LogTypeError, "Overlap::CalCoarseCorrPair(): ROI is invalid");
		return(false);
	}
		
		// Roi1 and Roi2
	roi1.FirstRow = (unsigned int)(dFirstRow1+0.5);
	roi1.LastRow = roi1.FirstRow+iRows-1;
	roi1.FirstColumn = (unsigned int)(dFirstCol1+0.5);
	roi1.LastColumn = roi1.FirstColumn+iCols-1;

	roi2.FirstRow = (unsigned int)dFirstRow2;
	roi2.LastRow = roi2.FirstRow+iRows-1;
	roi2.FirstColumn = (unsigned int)dFirstCol2;
	roi2.LastColumn = roi2.FirstColumn+iCols-1;

		// Validation check 
	if(	roi1.LastRow > _pImg1->Rows()-1 ||
		roi1.LastColumn > _pImg1->Columns()-1)
	{
		LOG.FireLogEntry(LogTypeError, "Overlap::CalCoarseCorrPair(): ROI is invalid");
		return(false);
	}

	if(	roi2.LastRow > _pImg2->Rows()-1 ||
		roi2.LastColumn > _pImg2->Columns()-1)
	{
		LOG.FireLogEntry(LogTypeError, "Overlap::CalCoarseCorrPair(): ROI is invalid");
	}

	// create coarse correlation pair
	unsigned int iDecim = CorrelationParametersInst.iCoarseMinDecim;
	unsigned int iColSearchExpansion = CorrelationParametersInst.iCoarseColSearchExpansion;
	unsigned int iRowSearchExpansion = CorrelationParametersInst.iCoarseRowSearchExpansion;
	if(_type != Fid_To_Fov)
	{
		if(roi1.Columns()>500 && roi1.Rows()>500)
			iDecim = 4;
	}

	CorrelationPair coarsePair(
		_pImg1, _pImg2, 
		roi1, pair<unsigned int, unsigned int>(roi2.FirstColumn, roi2.FirstRow), // (column row)
		iDecim, iColSearchExpansion, iRowSearchExpansion,
		this, _pMaskImg);

	if(!coarsePair.IsValid())
		return(false);

	_coarsePair = coarsePair;

	return(true);
}

void Overlap::Run()
{
	// Validation check
	if(!_bValid)
		return;

	// Special process for fiducial use vsfinder 
	if(_type == Fid_To_Fov)
	{
		FidFovOverlap* pTemp =  (FidFovOverlap*)this;
		if(pTemp->GetFiducialSearchMethod() == FIDVSFINDER)
		{
			pTemp->VsfinderAlign();
			_bProcessed = true;

			if(CorrelationParametersInst.bSaveOverlaps || CorrelationParametersInst.bSaveFiducialOverlaps)
			{
				DumpOvelapImages();
				DumpResultImages();
			}

			return;
		}
		else if(pTemp->GetFiducialSearchMethod() == FIDCYBERNGC)
		{
			pTemp->NgcFidAlign();
			_bProcessed = true;

			if(CorrelationParametersInst.bSaveOverlaps || CorrelationParametersInst.bSaveFiducialOverlaps)
			{
				DumpOvelapImages();
				DumpResultImages();
			}
			return;
		}
	}

	/*/ for debug
	if(_type == Fov_To_Fov)
	{
		FovFovOverlap* pTemp =  (FovFovOverlap*)this;
		int iIllum1, iIllum2, iCam1, iCam2, iTrig1, iTrig2;
		iIllum1 = pTemp->GetFirstMosaicImage()->Index();
		iIllum2 = pTemp->GetSecondMosaicImage()->Index();
		iCam1 = pTemp->GetFirstCameraIndex();
		iCam2 = pTemp->GetSecondCameraIndex();
		iTrig1 = pTemp->GetFirstTriggerIndex();
		iTrig2 = pTemp->GetSecondTriggerIndex();

		if(iIllum1 == 0 && iTrig1 == 7 && iCam1 == 0 &&
			iIllum2 == 2 && iTrig2 == 7 && iCam2 == 0)
			iIllum1 = 0;
	}//*/


	// Do coarse correlation
	bool bCorrSizeReduced = false;
	_coarsePair.DoAlignment(_bApplyCorrSizeUpLimit, &bCorrSizeReduced);

	// If the Roi size is reduced in correlation
	double dCoarseReliableScore = 0;
	if(_coarsePair.IsProcessed() && bCorrSizeReduced) 
	{	//If the correlation result is not good enough
		CorrelationResult result= _coarsePair.GetCorrelationResult();
		dCoarseReliableScore = fabs(result.CorrCoeff) * (1-result.AmbigScore);
		if(dCoarseReliableScore < CorrelationParametersInst.dCoarseResultReliableTh ||
			result.AmbigScore > CorrelationParametersInst.dCoarseResultAmbigTh )
		{
			// try again without ROI reduce
			_coarsePair.Reset();
			_coarsePair.DoAlignment();
		}
	}

	if(_type != Fov_To_Fov)
	{
		_bProcessed = true;

		if(CorrelationParametersInst.bSaveOverlaps || 
			(_type == Fid_To_Fov && CorrelationParametersInst.bSaveFiducialOverlaps))
		{
			DumpOvelapImages();
			DumpResultImages();
		}

		return;
	}

// Fine alignemt (only for Fov and Fov)
	// Clean fine correlation pair list
	_finePairList.clear();

	// Adjust ROI base on the coarse results
	bool bAdjusted = false;
	CorrelationPair tempPair = _coarsePair;
	dCoarseReliableScore = 0;
	if(_coarsePair.IsProcessed())
	{
		CorrelationResult result= _coarsePair.GetCorrelationResult();
		dCoarseReliableScore = fabs(result.CorrCoeff) * (1-result.AmbigScore);
		if(dCoarseReliableScore > CorrelationParametersInst.dCoarseResultReliableTh)
			bAdjusted = _coarsePair.AdjustRoiBaseOnResult(&tempPair);	
	}

	// Create fine correlation pair list
	unsigned int iBlockWidth = CorrelationParametersInst.iFineBlockWidth;
	unsigned int iNumBlockX = (tempPair.Columns()/iBlockWidth);
	if(iNumBlockX > CorrelationParametersInst.iFineMaxBlocksInCol) iNumBlockX = CorrelationParametersInst.iFineMaxBlocksInCol;
	if(iNumBlockX < 1) iNumBlockX = 1;
	if(iBlockWidth > tempPair.Columns()/iNumBlockX) iBlockWidth = tempPair.Columns()/iNumBlockX;

	unsigned int iBlockHeight = CorrelationParametersInst.iFineBlockHeight;
	unsigned int iNumBlockY = (tempPair.Rows()/iBlockHeight);
	if(iNumBlockY > CorrelationParametersInst.iFineMaxBlocksInRow) iNumBlockY = CorrelationParametersInst.iFineMaxBlocksInRow;
	if(iNumBlockY < 1) iNumBlockY = 1;
	if(iBlockHeight > tempPair.Rows()/iNumBlockY) iBlockHeight = tempPair.Rows()/iNumBlockY;

	unsigned int iBlockDecim = CorrelationParametersInst.iFineDecim;
	unsigned int iBlockColSearchExpansion = CorrelationParametersInst.iFineColSearchExpansion;
	unsigned int iBlockRowSearchExpansion = CorrelationParametersInst.iFineRowSearchExpansion;
	if(!bAdjusted)
	{
		iBlockColSearchExpansion = CorrelationParametersInst.iCoarseColSearchExpansion;
		iBlockRowSearchExpansion = CorrelationParametersInst.iCoarseRowSearchExpansion;
	}

	tempPair.ChopCorrPair(
		iNumBlockX, iNumBlockY,
		iBlockWidth, iBlockHeight,
		iBlockDecim, iBlockColSearchExpansion, iBlockRowSearchExpansion,
		&_finePairList);

	// Do fine correlation
	for(list<CorrelationPair>::iterator i=_finePairList.begin(); i!=_finePairList.end(); i++)
	{
		i->DoAlignment();

		// Validation check
		// To prevent wrong correlation results between different illuminaitons to be used
		if(bAdjusted) 
		{
			bool bValid = true;
			CorrelationResult result = i->GetCorrelationResult();;
			if(fabs(result.ColOffset) > 1.5 * CorrelationParametersInst.iFineColSearchExpansion ||
				fabs(result.RowOffset) > 1.5 * CorrelationParametersInst.iFineColSearchExpansion)
			{ // If the offset is too big
				bValid = false;
			}
			else if(fabs(result.ColOffset) > CorrelationParametersInst.iFineColSearchExpansion ||
				fabs(result.RowOffset) > CorrelationParametersInst.iFineColSearchExpansion)
			{	// If offset is big
				if(fabs(result.CorrCoeff)*(1-result.AmbigScore) < dCoarseReliableScore)
				{	// If fine result is less reliable than coarse one 
					bValid = false;
				}
			}

			// If fine result is not valid, ignore this result by set a faiure one
			if(bValid == false)
			{
				CorrelationResult failureResult;			// Default result is a failure result;
				i->SetCorrlelationResult(failureResult);	
			}
		}
	}	
	
	_bProcessed = true;


	/*/ for debug
	if(_type == Fov_To_Fov)
	{
		FovFovOverlap* pTemp =  (FovFovOverlap*)this;
		int iIllum1, iIllum2, iCam1, iCam2, iTrig1, iTrig2;
		iIllum1 = pTemp->GetFirstMosaicImage()->Index();
		iIllum2 = pTemp->GetSecondMosaicImage()->Index();
		iCam1 = pTemp->GetFirstCameraIndex();
		iCam2 = pTemp->GetSecondCameraIndex();
		iTrig1 = pTemp->GetFirstTriggerIndex();
		iTrig2 = pTemp->GetSecondTriggerIndex();

		if(iIllum1 == 0 && iTrig1 == 4 && iCam1 == 2 &&
			iIllum2 == 1 && iTrig2 == 4 && iCam2 == 2)
		{
			DumpOvelapImages();
			DumpResultImages();
		}
	}//*/

	if(CorrelationParametersInst.bSaveOverlaps)
	{
		DumpOvelapImages();
		DumpResultImages();
	}
}

#pragma endregion

#pragma region FovFovOverlap

FovFovOverlap::FovFovOverlap(
	MosaicLayer*	pMosaic1,
	MosaicLayer*	pMosaic2,
	pair<unsigned int, unsigned int> ImgPos1,
	pair<unsigned int, unsigned int> ImgPos2,
	DRect validRect,
	bool bApplyCorrSizeUpLimit,
	bool bHasMask)
{
	_pMosaic1 = pMosaic1;
	_pMosaic2 = pMosaic2;
	_imgPos1 = ImgPos1;
	_imgPos2 = ImgPos2;
	_bHasMask = bHasMask;

	Image* pImg1 = _pMosaic1->GetImage(ImgPos1.first, ImgPos1.second);
	Image* pImg2 = _pMosaic2->GetImage(ImgPos2.first, ImgPos2.second);
	
	Image* pMaskImg = NULL;
	if(bHasMask)
		pMaskImg = _pMosaic1->GetMaskImage(ImgPos1.first, ImgPos1.second);
	config(pImg1, pImg2, validRect, Fov_To_Fov, bApplyCorrSizeUpLimit, pMaskImg);
}

bool FovFovOverlap::IsReadyToProcess() const
{
	bool bFlag =
		_pMosaic1->GetTile(_imgPos1.first, _imgPos1.second)->ContainsImage() &&
		_pMosaic2->GetTile(_imgPos2.first, _imgPos2.second)->ContainsImage() &&
		_bValid;

	return(bFlag);
}

// Return ture if overlap is from (illum1, iTrig1) and (illum2, iTrig2)
// reusults for (illum1, iTrig1) and (illum2, iTrig2) in different order are same
bool FovFovOverlap::IsFromIllumTrigs(
	unsigned int iLlum1,
	unsigned int iTrig1,
	unsigned int iLlum2,
	unsigned int iTrig2) const
{
	if((_pMosaic1->Index() == iLlum1 && GetFirstTriggerIndex() == iTrig1 &&
		_pMosaic2->Index() == iLlum2 && GetSecondTriggerIndex() == iTrig2) 
		||
		(_pMosaic1->Index() == iLlum2 && GetFirstTriggerIndex() == iTrig2 &&
		_pMosaic2->Index() == iLlum1 && GetSecondTriggerIndex() == iTrig1))
	{
		return(true);
	}
	else
	{
		return(false);
	}
}

bool FovFovOverlap::IsFromSameDevice() const
{
	return(_pMosaic1->DeviceIndex() == _pMosaic2->DeviceIndex());	
}

// For Debug 
bool FovFovOverlap::DumpOvelapImages()
{
	if(!IsReadyToProcess())
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sFovFov_coarse_I%dT%dC%d_I%dT%dC%d.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic1->Index(), _imgPos1.second, _imgPos1.first,
		_pMosaic2->Index(), _imgPos2.second, _imgPos2.first);
		
	s.append(cTemp);
	_coarsePair.DumpImg(s);

	int iCount = 0;
	for(list<CorrelationPair>::iterator i=_finePairList.begin(); i!=_finePairList.end(); i++)
	{
		sprintf_s(cTemp, 100, "%sFovFov_Fine_I%dT%dC%d_I%dT%dC%d_%d.bmp",  
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic1->Index(), _imgPos1.second, _imgPos1.first,
		_pMosaic2->Index(), _imgPos2.second, _imgPos2.first, i->GetIndex());

		s.clear();
		s.append(cTemp);
		i->DumpImg(s);

		iCount++;
	}

	return(true);
}

bool FovFovOverlap::DumpResultImages()
{
	if(!_bProcessed)
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sResult_FovFov_coarse_I%dT%dC%d_I%dT%dC%d_Score%dAmbig%d.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic1->Index(), _imgPos1.second, _imgPos1.first,
		_pMosaic2->Index(), _imgPos2.second, _imgPos2.first,
		(int)(_coarsePair.GetCorrelationResult().CorrCoeff*100),
		(int)(_coarsePair.GetCorrelationResult().AmbigScore*100));
		
	s.append(cTemp);
	_coarsePair.DumpImgWithResult(s);

	int iCount = 0;
	for(list<CorrelationPair>::iterator i=_finePairList.begin(); i!=_finePairList.end(); i++)
	{
		sprintf_s(cTemp, 100, "%sResult_FovFov_Fine_I%dT%dC%d_I%dT%dC%d_%d_Score%dAmbig%d.bmp",
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic1->Index(), _imgPos1.second, _imgPos1.first,
		_pMosaic2->Index(), _imgPos2.second, _imgPos2.first, i->GetIndex(),
		(int)(i->GetCorrelationResult().CorrCoeff*100),
		(int)(i->GetCorrelationResult().AmbigScore*100));

		s.clear();
		s.append(cTemp);
		i->DumpImgWithResult(s);

		iCount++;
	}

	return true;
}

#pragma endregion 

#pragma region CadFovOverlap class

CadFovOverlap::CadFovOverlap(
	MosaicLayer* pMosaic,
	pair<unsigned int, unsigned int> ImgPos,
	Image* pCadImg,
	DRect validRect)
{
	_pMosaic = pMosaic;
	_imgPos = ImgPos;
	_pCadImg = pCadImg;

	Image* pImg1 = _pMosaic->GetImage(ImgPos.first, ImgPos.second);

	config(pImg1, _pCadImg, validRect, Cad_To_Fov, false);
}

bool CadFovOverlap::IsReadyToProcess() const
{
	bool bFlag =
		_pMosaic->GetTile(_imgPos.first, _imgPos.second)->ContainsImage() &&
		(_pCadImg != NULL) && (_pCadImg->GetBuffer() != NULL) &&
		_bValid;

	return(bFlag);
}

// For Debug 
bool CadFovOverlap::DumpOvelapImages()
{
	if(!IsReadyToProcess())
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sCadFov_I%dT%dC%d.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic->Index(), _imgPos.second, _imgPos.first);
		
	s.append(cTemp);
	_coarsePair.DumpImg(s);

	return(true);
}

bool CadFovOverlap::DumpResultImages()
{
	if(!_bProcessed)
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sResult_CadFov_I%dT%dC%d_Score%dAmbig%d.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic->Index(), _imgPos.second, _imgPos.first,
		(int)(_coarsePair.GetCorrelationResult().CorrCoeff*100),
		(int)(_coarsePair.GetCorrelationResult().AmbigScore*100));
		
	s.append(cTemp);
	_coarsePair.DumpImgWithResult(s);

	return(true);
}

#pragma endregion

#pragma region FidFovOverlap class

FidFovOverlap::FidFovOverlap(
	MosaicLayer*	pMosaic,
	pair<unsigned int, unsigned int> ImgPos,
	Image* pFidImg,
	double dCenterX,
	double dCenterY,
	DRect validRect)
{
	_pMosaic = pMosaic;
	_imgPos = ImgPos;
	_pFidImg = pFidImg;

	_dFidCenterX = dCenterX;
	_dFidCenterY = dCenterY;

	Image* pImg1 = _pMosaic->GetImage(ImgPos.first, ImgPos.second);

	_fidSearchMethod = FIDREGOFF;

	config(pImg1, _pFidImg, validRect, Fid_To_Fov, false);
}

void FidFovOverlap::SetVsFinder(unsigned int iTemplateID)
{
	_fidSearchMethod = FIDVSFINDER;
	_iTemplateID = iTemplateID;
}

void FidFovOverlap::SetNgcFid(unsigned int iTemplateID)
{
	_fidSearchMethod = FIDCYBERNGC;
	_iTemplateID = iTemplateID;
}

bool FidFovOverlap::IsReadyToProcess() const
{
	bool bFlag =
		_pMosaic->GetTile(_imgPos.first, _imgPos.second)->ContainsImage() &&
		(_pFidImg != NULL) && (_pFidImg->GetBuffer() != NULL) &&
		_bValid;

	return(bFlag);
}

// Get the weight for the solver
double FidFovOverlap::GetWeightForSolver()
{
	// Validation check for overlap
	if(!IsProcessed())
		return(-1);

	CorrelationPair* pPair = GetCoarsePair();

	// Validation check for correlation pair
	CorrelationResult result;
	if(!pPair->GetCorrelationResult(&result))
		return(-2);

	return(Weights.CalWeight(pPair));
}

// Calculate fiducial center based on transform
bool FidFovOverlap::CalFidCenterBasedOnTransform(ImgTransform trans, double* pdx, double* pdy)
{
	double w = GetWeightForSolver();
	if(w <= 0) 
		return(false);

	CorrelationPair* pPair = GetCoarsePair();
	CorrelationResult result;
	pPair->GetCorrelationResult(&result);

	// Get Centers of ROIs (fiducial image is always the second one)
	double rowImgA = (pPair->GetFirstRoi().FirstRow + pPair->GetFirstRoi().LastRow)/ 2.0;
	double colImgA = (pPair->GetFirstRoi().FirstColumn + pPair->GetFirstRoi().LastColumn)/ 2.0;

	// The center of ROI in fiducial image
	double rowImgB = (pPair->GetSecondRoi().FirstRow + pPair->GetSecondRoi().LastRow)/ 2.0;
	double colImgB = (pPair->GetSecondRoi().FirstColumn + pPair->GetSecondRoi().LastColumn)/ 2.0;

	// Fiducial center in fiducial image 	
	double rowFidinImgB, colFidinImgB;
	GetFidImage()->WorldToImage(_dFidCenterX, _dFidCenterY, &rowFidinImgB, &colFidinImgB);

	// Get offset
	double offsetRows = result.RowOffset;
	double offsetCols = result.ColOffset;

	// Fiducial location in FOV
	double rowFidInFov = rowImgA-offsetRows + (rowFidinImgB-rowImgB);
	double colFidInFov = colImgA-offsetCols + (colFidinImgB-colImgB);

	// Fiducial location based on transform
	trans.Map(rowFidInFov, colFidInFov, pdx, pdy);

	return(true);
}

// For Debug 
bool FidFovOverlap::DumpOvelapImages()
{
	if(!IsReadyToProcess())
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sFidFov_I%dT%dC%d_ID%d.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic->Index(), _imgPos.second, _imgPos.first,
		_iTemplateID);
		
	s.append(cTemp);
	_coarsePair.DumpImg(s);

	return(true);
}

bool FidFovOverlap::DumpResultImages()
{
	if(!_bProcessed)
		return(false);

	string s;
	char cTemp[100];
	sprintf_s(cTemp, 100, "%sResult_FidFov_I%dT%dC%d_ID%d_Score%dAmbig%d%s.bmp", 
		CorrelationParametersInst.GetOverlapPath().c_str(),
		_pMosaic->Index(), _imgPos.second, _imgPos.first, 
		_iTemplateID,
		(int)(_coarsePair.GetCorrelationResult().CorrCoeff*100),
		(int)(_coarsePair.GetCorrelationResult().AmbigScore*100),
		_fidSearchMethod==FIDCYBERNGC? "_Ngc" : "");
		
	s.append(cTemp);
	_coarsePair.DumpImgWithResult(s);

	return(true);
}

bool FidFovOverlap::VsfinderAlign()
{
	double x, y, corscore, ambig, ngc;
	double search_center_x = (_coarsePair.GetFirstRoi().FirstColumn + _coarsePair.GetFirstRoi().LastColumn)/2.0; 
	double search_center_y = (_coarsePair.GetFirstRoi().FirstRow + _coarsePair.GetFirstRoi().LastRow)/2.0; 
	double search_width = _coarsePair.GetFirstRoi().Columns();
	double search_height = _coarsePair.GetFirstRoi().Rows();
	double time_out = 1e5;			// MicroSeconds
	double dMinScore = CorrelationParametersInst.dVsFinderMinCorrScore;
	
	VsFinderCorrelation::Instance().Find(
		_iTemplateID,							// map ID of template  and finder
		_coarsePair.GetFirstImg()->GetBuffer(),	// buffer containing the image
		_coarsePair.GetFirstImg()->Columns(),   // width of the image in pixels
		_coarsePair.GetFirstImg()->Rows(),		// height of the image in pixels
		x,										// returned x location of the center of the template from the origin
		y,										// returned x location of the center of the template from the origin
		corscore,								// match score 0-1
		ambig,									// ratio of (second best/best match) score 0-1
		&ngc,									// Normalized Grayscale Correlation Score 0-1
		search_center_x,						// x center of the search region in pixels
		search_center_y,						// y center of the search region in pixels
		search_width,							// width of the search region in pixels
		search_height,							// height of the search region in pixels
		time_out,								// number of seconds to search maximum. If limit is reached before any results found, an error will be generated	
		0,										// image origin is top-left				
		dMinScore/3,							// If >0 minimum score to persue at min pyramid level to look for peak override
		dMinScore/3);							// If >0 minumum score to accept at max pyramid level to look for peak override
												// Use a lower minimum score for vsfinder so that we can get a reliable ambig score

	CorrelationResult result;
	if(corscore > dMinScore)	// Valid results
	{
		result.CorrCoeff = corscore;
		result.AmbigScore = ambig;
		
		// Unclipped image patch (first image of overlap) center 
		// matches drawed fiducial center (second image of overlap) in the overlap
		// alignment offset is the difference of unclipped image patch center 
		// and idea location of fiducial in first image of overlap
		// Offset of unclipped center and clipped center
		double dCenOffsetX = (_coarsePair.GetSecondImg()->Columns()-1)/2.0 - 
			(_coarsePair.GetSecondRoi().FirstColumn+_coarsePair.GetSecondRoi().LastColumn)/2.0;
		double dCenOffsetY = (_coarsePair.GetSecondImg()->Rows()-1)/2.0 - 
			(_coarsePair.GetSecondRoi().FirstRow+_coarsePair.GetSecondRoi().LastRow)/2.0;
		double dUnclipCenterX = search_center_x + dCenOffsetX;
		double dUnclipCenterY = search_center_y + dCenOffsetY;
		result.ColOffset = dUnclipCenterX - x;
		result.RowOffset = dUnclipCenterY - y;
	}
	else	// Invalid results
	{
		result.CorrCoeff = 0;
		result.AmbigScore = 1;
	}

	_coarsePair.SetCorrlelationResult(result);
	/*
	LOG.FireLogEntry(LogTypeDiagnostic, "VsFinder VSScore=%f; ambig=%f; xOffset=%f; yOffset=%f; NgcScore=%f", 
		result.CorrCoeff, 
		result.AmbigScore,
		result.ColOffset,
		result.RowOffset,
		ngc);
	*/
	return(true);
}

bool FidFovOverlap::NgcFidAlign()
{
	double x, y, corscore, ambig;
	double search_center_x = (_coarsePair.GetFirstRoi().FirstColumn + _coarsePair.GetFirstRoi().LastColumn)/2.0; 
	double search_center_y = (_coarsePair.GetFirstRoi().FirstRow + _coarsePair.GetFirstRoi().LastRow)/2.0; 
	
	bool bFlag = CyberNgcFiducialCorrelation::Instance().Find(
		_iTemplateID,					// Map ID of template  and finder
		_coarsePair.GetFirstImg(),		// Search image
		_coarsePair.GetFirstRoi(),		// Search ROI
		x,								// returned x location of the center of the template from the origin
		y,								// returned Y location of the center of the template from the origin
		corscore,						// match score [-1,1]
		ambig);							// ratio of (second best/best match) score 0-1

	CorrelationResult result;
	if(bFlag)	// Valid results
	{
		result.CorrCoeff = corscore;
		result.AmbigScore = ambig;
		
		// Unclipped image patch (first image of overlap) center 
		// matches drawed fiducial center (second image of overlap) in the overlap
		// alignment offset is the difference of unclipped image patch center 
		// and idea location of fiducial in first image of overlap
		// Offset of unclipped center and clipped center
		double dCenOffsetX = (_coarsePair.GetSecondImg()->Columns()-1)/2.0 - 
			(_coarsePair.GetSecondRoi().FirstColumn+_coarsePair.GetSecondRoi().LastColumn)/2.0;
		double dCenOffsetY = (_coarsePair.GetSecondImg()->Rows()-1)/2.0 - 
			(_coarsePair.GetSecondRoi().FirstRow+_coarsePair.GetSecondRoi().LastRow)/2.0;
		double dUnclipCenterX = search_center_x + dCenOffsetX;
		double dUnclipCenterY = search_center_y + dCenOffsetY;
		result.ColOffset = dUnclipCenterX - x;
		result.RowOffset = dUnclipCenterY - y;
	}
	else	// Invalid results
	{
		result.CorrCoeff = 0;
		result.AmbigScore = 1;
	}

	_coarsePair.SetCorrlelationResult(result);
	/*
	LOG.FireLogEntry(LogTypeDiagnostic, "CyberNGC Score=%f; ambig=%f; xOffset=%f; yOffset=%f", 
		result.CorrCoeff, 
		result.AmbigScore,
		result.ColOffset,
		result.RowOffset);
*/
	return(true);
}

#pragma endregion



