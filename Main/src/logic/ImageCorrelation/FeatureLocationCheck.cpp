#include "FeatureLocationCheck.h"
#include "Feature.h"
#include "Image.h"
#include "RenderShape.h"
#include "VsFinderCorrelation.h"
#include "Bitmap.h"
extern "C" {
#include "warpxy.h"
}

#include <map>
using std::map;
typedef map<int, Feature*> FeatureList;
typedef FeatureList::iterator FeatureListIterator;

#pragma region class FeatureLocationCheck
// fiducial is placed in the exact center of the resulting image
// this is important as SqRtCorrelation (used to build least squares table)
// measures the positional difference between the center of the fiducial 
// image and the center of the FOV image segement.  Any information
// about offset of the fiducial from the center of the FOV is lost
// img: output, fiducial image,
// fid: input, fiducial feature 
// resolution: the pixel size, in meters
// dScale: the fiducial area expansion scale
void RenderFiducial(
	Image* pImg, 
	Feature* pFid, 
	double resolution, 
	double dScale,
	double dExpansion)
{
	// Area in world space
	double dHalfWidth = pFid->GetBoundingBox().Width()/2;
	double dHalfHeight= pFid->GetBoundingBox().Height()/2;
	double xMinImg = pFid->GetBoundingBox().Min().x - dHalfWidth*(dScale-1)  - dExpansion; 
	double yMinImg = pFid->GetBoundingBox().Min().y - dHalfHeight*(dScale-1) - dExpansion; 
	double xMaxImg = pFid->GetBoundingBox().Max().x + dHalfWidth*(dScale-1)  + dExpansion; 
	double yMaxImg = pFid->GetBoundingBox().Max().y + dHalfHeight*(dScale-1) + dExpansion;
	
	// The size of image in pixels
	unsigned int nRowsImg
			(unsigned int((1.0/resolution)*(xMaxImg-xMinImg)));	
	unsigned int nColsImg
			(unsigned int((1.0/resolution)*(yMaxImg-yMinImg)));	
	
	// Define transform for new fiducial image
	double t[3][3];
	t[0][0] = resolution;
	t[0][1] = 0;
	t[0][2] = pFid->GetCadX() - resolution * (nRowsImg-1) / 2.;
	
	t[1][0] = 0;
	t[1][1] = resolution;
	t[1][2] = pFid->GetCadY() - resolution * (nColsImg-1) / 2.;
	
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;	
	
	ImgTransform trans(t);

	// Create and clean buffer
	pImg->Configure(nColsImg, nRowsImg, nColsImg, trans, trans, true);
	pImg->ZeroBuffer();

	// Fiducial drawing/render
	unsigned int grayValue = 255;
	int antiAlias = 1;
	RenderFeature(pImg, pFid, grayValue, antiAlias);
}


// Create Fiducial images
void CreateFiducialImage(
	Panel* pPanel, 
	Image* pImage, 
	Feature* pFeature,
	double dExpansion)
{
	double dScale = 1.4;
	
	RenderFiducial(
		pImage, 
		pFeature, 
		pPanel->GetPixelSizeX(), 
		dScale,
		dExpansion);	
}

bool VsfinderAlign(
	Image* pImg,
	int iTemplateID,
	double dSearchColCen,
	double dSearchRowCen,
	double dSearchWidth,
	double dSearchHeight,
	double* pdScore,
	double* pdAmbig,
	double* pdX,
	double* pdY
	)
{
	double x, y, corscore, ambig, ngc;
	double time_out = 1e5;			// MicroSeconds
	double dMinScore = 0.4;
	double dMaxAmbig = 0.7;
	
	VsFinderCorrelation::Instance().Find(
		iTemplateID,		// map ID of template  and finder
		pImg->GetBuffer(),	// buffer containing the image
		pImg->PixelRowStride(),	// width of the image in pixels
		pImg->Rows(),		// height of the image in pixels
		x,					// returned x location of the center of the template from the origin
		y,					// returned x location of the center of the template from the origin
		corscore,			// match score 0-1
		ambig,				// ratio of (second best/best match) score 0-1
		&ngc,				// Normalized Grayscale Correlation Score 0-1
		dSearchColCen,		// Col center of the search region in pixels
		dSearchRowCen,		// Row center of the search region in pixels
		dSearchWidth,		// width of the search region in pixels
		dSearchHeight,		// height of the search region in pixels
		time_out,			// number of seconds to search maximum. If limit is reached before any results found, an error will be generated	
		0,					// image origin is top-left				
		dMinScore/3,		// If >0 minimum score to persue at min pyramid level to look for peak override
		dMinScore/3);		// If >0 minumum score to accept at max pyramid level to look for peak override
							// Use a lower minimum score for vsfinder so that we can get a reliable ambig score

	if(corscore > dMinScore && ambig < dMaxAmbig)	// Valid results
	{
		*pdScore = corscore;
		*pdAmbig = ambig;
		*pdX = x;
		*pdY = y;
	}
	else	// Invalid results
	{
		*pdScore = 0;
		*pdAmbig = 1;
		*pdX = -1;
		*pdY = -1;
	}

	return(true);
}


// For Debug
void DumpImg(
	string sFileName, int iWidth, int iHeight,
	Image* pImg1, int iStartX1, int iStartY1,
	Image* pImg2, int iStartX2, int iStartY2)
{
	unsigned char* pcBuf1 = pImg1->GetBuffer() 
		+ pImg1->PixelRowStride()*iStartY1
		+ iStartX1;

	if (pImg2 != NULL)
	{
		unsigned char* pcBuf2 = pImg2->GetBuffer() 
			+ pImg2->PixelRowStride()*iStartY2
			+ iStartX2;

		Bitmap* rbg = Bitmap::New2ChannelBitmap( 
			iHeight, 
			iWidth,
			pcBuf1, 
			pcBuf2,
			pImg1->PixelRowStride(),
			pImg2->PixelRowStride() );

		rbg->write(sFileName);
		delete rbg;
	}
	else
	{
		Bitmap* rbg = Bitmap::NewBitmapFromBuffer( 
			iHeight, 
			iWidth,
			pImg1->PixelRowStride(),
			pcBuf1,
			pImg1->GetBitsPerPixel() );

		rbg->write(sFileName);
		delete rbg;
	}
}

// pPanel: the point for fidcial descrptions
FeatureLocationCheck::FeatureLocationCheck(Panel* pPanel)
{
	// Setttings
	_dSearchExpansion = 2e-3; // 2 mm, if more is needed, something bad happened.

	_pPanel = pPanel;
	int iNum = _pPanel->NumberOfFiducials();
	_piTemplateIds = new int[iNum];

	// For debug
	_iCycleCount = 0;
	_pFidImages = new Image[iNum];

	// Vsfinder initialize
	VsFinderCorrelation::Instance().Config(
		_pPanel->GetPixelSizeX(), _pPanel->GetNumPixelsInY(), _pPanel->GetNumPixelsInX());

	int iCount = 0;
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		// For Debug
		// Create a fiducial image
		CreateFiducialImage(_pPanel, &_pFidImages[iCount], iFid->second, _dSearchExpansion);

		bool bFidBrighter = true;
		bool bAllowNegMatch = false;

		// Create Fiducial template 
		int iVsFinderTemplateId = VsFinderCorrelation::Instance().CreateVsTemplate(iFid->second, bFidBrighter, bAllowNegMatch);
		if(iVsFinderTemplateId < 0) 
			return;

		_piTemplateIds[iCount] = iVsFinderTemplateId;
		iCount++;
	}
}


FeatureLocationCheck::~FeatureLocationCheck(void)
{
	delete [] _pFidImages;
	delete [] _piTemplateIds;
}

// Find locations of fiducials in a panel image
// pImage: input, panel image
// dResults: output, the fidcuial results,
// for each fiducial return(cad_x, cad_y, Loc_x, loc_y, corrScore, ambig)
bool FeatureLocationCheck::CheckFeatureLocation(Image* pImage, double dResults[])
{	
	// Setttings
	int iItems = 6;
	
	if(pImage == NULL) return(false);
	double dPixelSize = pImage->PixelSizeX();

	// For each fiducial
	int iCount = 0;
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		// Prepare search area
		Box box = iFid->second->GetBoundingBox();
		double dSearchRowCen, dSearchColCen;
		pImage->WorldToImage(box.Center().x, box.Center().y, &dSearchRowCen, &dSearchColCen);
		double dSearchHeight = (box.Width() + _dSearchExpansion*2)/dPixelSize; // The CAD and image has 90 degree rotation
		double dSearchWidth = (box.Height() + _dSearchExpansion*2)/dPixelSize;

		// Find fiducial
		double dCol, dRow, dScore, dAmbig; 
		bool bFlag= VsfinderAlign(
			pImage,
			_piTemplateIds[iCount],
			dSearchColCen,
			dSearchRowCen,
			dSearchWidth,
			dSearchHeight,
			&dScore,
			&dAmbig,
			&dCol,
			&dRow);

		double dX, dY;
		if(dScore==0 || dAmbig==1) // If feature/fiducial is not found
		{
			dX = -1;
			dY = -1;
		}
		else
			pImage->ImageToWorld(dRow, dCol, &dX, &dY);

		// Record results
		dResults[iCount*iItems] = box.Center().x;	// CAD x  
		dResults[iCount*iItems+1] = box.Center().y; // CAD y
		dResults[iCount*iItems+2] = dX;				// Loc x  
		dResults[iCount*iItems+3] = dY;				// Loc y
		dResults[iCount*iItems+4] = dScore;			// Loc x  
		dResults[iCount*iItems+5] = dAmbig;			// Loc y

		///* For debug image output */
		//string sFileName;
		//char cTemp[100];

		//// for debug image output
		//if(dScore!=0 && dAmbig!=1)
		//{
		//	sprintf_s(cTemp, 100, "C:\\Temp\\GoodResults_Cycle%d_Fid%d_c%dr%d_s%da%d.bmp", 
		//		_iCycleCount, iCount, 
		//		(int)((box.Center().y-dY)*1e6), (int)((box.Center().x-dX)*1e6),
		//		(int)(dScore*100), (int)(dAmbig*100) );
		//	sFileName.clear();
		//	sFileName.append(cTemp);

		//	DumpImg(
		//		sFileName, (int)dSearchWidth, (int)dSearchHeight,
		//		pImage, (int)(dCol-dSearchWidth/2), (int)(dRow-dSearchHeight/2),
		//		NULL/*&_pFidImages[iCount]*/, (int)(_pFidImages[iCount].Columns()/2.-dSearchWidth/2) , (int)(_pFidImages[iCount].Rows()/2.-dSearchHeight/2) );
		//}
		//else
		//{
		//	sprintf_s(cTemp, 100, "C:\\Temp\\BadResults_Cycle%d_Fid%d_c%dr%d_s%da%d.bmp", 
		//		_iCycleCount, iCount, 
		//		(int)((box.Center().y/*-dY*/)*1e6), (int)((box.Center().x/*-dX*/)*1e6),
		//		(int)(dScore*100), (int)(dAmbig*100) );
		//	sFileName.clear();
		//	sFileName.append(cTemp);

		//	DumpImg(
		//		sFileName, (int)dSearchWidth, (int)dSearchHeight,
		//		pImage, (int)(dSearchColCen-dSearchWidth/2), (int)(dSearchRowCen-dSearchHeight/2),
		//		NULL/*&_pFidImages[iCount]*/, (int)(_pFidImages[iCount].Columns()/2.-dSearchWidth/2) , (int)(_pFidImages[iCount].Rows()/2.-dSearchHeight/2) );
		//}

		iCount++;
	}

	_iCycleCount++;

	return(true);
}

#pragma endregion

#pragma region class ImageFidAligner

ImageFidAligner::ImageFidAligner(Panel* pPanel)
{
	_pPanel = pPanel;

	// create fiducial finder
	_pFidFinder = new FeatureLocationCheck(pPanel);

	// Create image for process
	ImgTransform inputTransform;
	inputTransform.Config(pPanel->GetPixelSizeX(), 
		pPanel->GetPixelSizeX(), 0, 0, 0);

	_pMorphedImage = new Image
			(pPanel->GetNumPixelsInY(),		// Columns
			pPanel->GetNumPixelsInX(),		// Rows	
			pPanel->GetNumPixelsInY(),		// stride In pixels
			1,								// Bytes per pixel
			inputTransform,					
			inputTransform,		
			true);							// Falg for whether create own buffer
}

ImageFidAligner::~ImageFidAligner()
{
	delete _pFidFinder;
	delete _pMorphedImage;
}

// Calculate transform based on fiducial location
// pImage: input, stitched image which is flatterned
// t: output, calculated transform based on fiducial location
bool ImageFidAligner::CalculateTransform(Image* pImage, double t[3][3])
{
	// Settings
	int iItems = 6;
	double dMinScore = 0.3;
	int NUMBER_Z_BASIS_FUNCTIONS = 4;
	double wFidFlatBoardScale   = 1e3;
	double wFidFlatBoardRotation = 1e2;
	double wFidFlatFiducialLateralShift = 1e3;
	double wFidFlatFlattenFiducial = 1e5;
	
// Find the fiducial and process them
	// Do fiducial search
	int iNumFids = _pPanel->NumberOfFiducials();
	double* pdResults = new double[iItems*iNumFids]; 
	_pFidFinder->CheckFeatureLocation(pImage, pdResults);

	// Collect valid results
	int iCount = 0;
	POINT2D* pCadLoc = new POINT2D[iNumFids];
	POINT2D* pPanelLoc = new POINT2D[iNumFids];
	for(int i=0; i<iNumFids; i++)
	{
		double dScore = pdResults[i*6+4] * (1-pdResults[i*6+5]);
		if(dScore < dMinScore)
			continue;

		pCadLoc[iCount].x = pdResults[i*6];
		pCadLoc[iCount].y = pdResults[i*6+1];
		pPanelLoc[iCount].x = pdResults[i*6+2];
		pPanelLoc[iCount].y = pdResults[i*6+3];
		iCount++;
	}
	int nGoodFids = iCount;

// Fill the solver
	// Total 8 constraints
	int nContraints = 8;
		
	// Create A and b matrices, these are small so we can create/delete them without using big blocks of memory
	double		*FidFitA;
	double		*FidFitX;
	double		*FidFitb;
	double		*resid;
	unsigned int FidFitCols = 6;
	unsigned int FidFitRows = nGoodFids*2 + 8;
	int SizeofFidFitA = FidFitCols * FidFitRows;
	FidFitA = new double[SizeofFidFitA];
	FidFitX = new double[FidFitCols];
	FidFitb = new double[FidFitRows];
	resid   = new double[FidFitRows];
	// zeros the system
	for(size_t s(0); s<SizeofFidFitA; ++s)
		FidFitA[s] = 0.0;
	for (size_t s(0); s<FidFitRows; ++s)
		FidFitb[s] = 0.0;
	for (size_t s(0); s<FidFitCols; ++s)
		FidFitX[s] = 0.0;
	// add constraints
	FidFitb[0] = 1.0 * wFidFlatBoardScale;
	FidFitA[0*FidFitCols + 0] =  1.0 * wFidFlatBoardScale;
	FidFitb[1] = 1.0 * wFidFlatBoardScale;
	FidFitA[1*FidFitCols + 4] =  1.0 * wFidFlatBoardScale;
	FidFitb[2] = 0.0 * wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 0] =  1.0 * wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 4] = -1.0 * wFidFlatBoardScale;
	FidFitb[3] = 0.0 * wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 1] =  1.0 * wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 3] =  1.0 * wFidFlatBoardScale;
	FidFitb[4] = 0.0 * wFidFlatBoardRotation;
	FidFitA[4*FidFitCols + 1] =  1.0 * wFidFlatBoardRotation;
	FidFitb[5] = 0.0 * wFidFlatBoardRotation;
	FidFitA[5*FidFitCols + 3] =  1.0 * wFidFlatBoardRotation;
	FidFitb[6] = 0.0 * wFidFlatFiducialLateralShift;
	FidFitA[6*FidFitCols + 2] =  1.0 * wFidFlatFiducialLateralShift;
	FidFitb[7] = 0.0 * wFidFlatFiducialLateralShift;
	FidFitA[7*FidFitCols + 5] =  1.0 * wFidFlatFiducialLateralShift;
	// add the actual fiducial locations
	int AStart(0);
	for (int j(0); j< nGoodFids; j++)
	{
		FidFitb[8+j*2+0] = pCadLoc[j].x * wFidFlatFlattenFiducial;
		FidFitb[8+j*2+1] = pCadLoc[j].y * wFidFlatFlattenFiducial;
		AStart = (8+j*2+0)*FidFitCols;
		FidFitA[AStart + 0] = pPanelLoc[j].x * wFidFlatFlattenFiducial;
		FidFitA[AStart + 1] = pPanelLoc[j].y * wFidFlatFlattenFiducial;
		FidFitA[AStart + 2] = wFidFlatFlattenFiducial;
		AStart = (8+j*2+1)*FidFitCols;
		FidFitA[AStart + 3] = pPanelLoc[j].x * wFidFlatFlattenFiducial;
		FidFitA[AStart + 4] = pPanelLoc[j].y * wFidFlatFlattenFiducial;
		FidFitA[AStart + 5] = wFidFlatFlattenFiducial;
	}

	// Calculate transform
	LstSqFit(FidFitA, FidFitRows, FidFitCols, FidFitb, FidFitX, resid);

	t[0][0] = FidFitX[0];
	t[0][1] = FidFitX[1];
	t[0][2] = FidFitX[2];
	t[1][0] = FidFitX[3];
	t[1][1] = FidFitX[4];
	t[1][2] = FidFitX[5];
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;
	
	delete [] FidFitA;
	delete [] FidFitb;
	delete [] FidFitX;
	delete [] resid;

	return(true);
}


// Morph the input image based on fiducial and panel surface
// pImgIn: input, stitched image which is flatterned.
// return morphed image
Image* ImageFidAligner::MorphImage(Image* pImgIn)
{
	// Calculate transform based on fiucial
	double t[3][3];
	CalculateTransform(pImgIn, t);

	// Set output image transform
	// double check
	ImgTransform trans(t);
	trans = trans.Inverse()
		*_pMorphedImage->GetNominalTransform();
	_pMorphedImage->SetTransform(trans);

	// Fill the output image
	_pMorphedImage->ZeroBuffer();

	bool bIsYCrCb = false;
	UIRect roi(0, 0,_pMorphedImage->Columns()-1, _pMorphedImage->Rows()-1); 
	_pMorphedImage->MorphFrom(
		pImgIn, 
		bIsYCrCb,
		roi,
		NULL,
		0,
		0);

	return(_pMorphedImage);
}

#pragma endregion