#include "CorrelationParameters.h"

CorrelationParameters* CorrelationParameters::_pInst = 0;
CorrelationParameters& CorrelationParameters::Instance()
{
	if(_pInst == 0)
		_pInst = new CorrelationParameters();

	return(*_pInst);
}

CorrelationParameters::CorrelationParameters(void)
{
	// Correlation pair
	iCorrMaxColsToUse = 1024;			// Max cols will be used in alignment for a correlation pair
	iCorrMaxRowsToUse = 1024;			// Max rows will be used in alignment for a correlation pair
	iCorrPairMinRoiSize = 20;			// The Size of Roi of correlation pair need >= this value to process
	dMaskAreaRatioTh = 0.05;			// NGC will be used only if Mask area/Roi area > this value

	// Coarse correlation
	iCoarseMinDecim = 2;				// Minimum decimatin for coarse correlation
	iCoarseColSearchExpansion = 50;		// Search expansion in cols for coarse correlation
	iCoarseRowSearchExpansion = 75;		// Search expansion in rows for coarse correlation
	dCoarseResultReliableTh = 0.03;		// Threshold to decide whether coarse correlaiton result is reliable

	// Fine correlation
	iFineBlockWidth = 256;				// Default width of fine correlation block
	iFineBlockHeight = 384;				// Default height of fine correlation block 
	iFineMaxBlocksInCol = 3;			// Max number of blocks in colum direction for fine correlation
	iFineMaxBlocksInRow = 3;			// Max Number of blocks in row direction for fine correlation

	iFineDecim = 2;						// Decimatin for fine correlation
	iFineColSearchExpansion = 20;		// Search expansion in cols for fine correlation if coarse correlation is successed
	iFineRowSearchExpansion = 20;		// Search expansion in rows for fine correlation if coarse correlation is successed

	// Use Vsfinder
	bUseVsFinder = true;				// Use vsfinder or regoff for fiducial finding

	// Fiducial search expansion
	dFiducialSearchExpansionX = 6e-3;	// Fiducial search expansion in x and y of world space 
	dFiducialSearchExpansionY = 2e-3;

	// Overlap
	iMinOverlapSize =100;				// Minimum overlap size for FovFov and FovCad

	// Support Bayer(color image)
	bGrayScale = true;					// Grey scale image in default

	// Number of Threads to use for processing
	NumThreads = 6;

	// debug flags
	bSaveFiducialOverlaps = false;
	bSaveOverlaps = false;
	bSaveMaskVectors = false;
	sDiagnosticPath = "C:\\Temp\\";
	sOverlapPath = sDiagnosticPath + "Overlaps\\";
}

CorrelationParameters::~CorrelationParameters(void)
{
}

string CorrelationParameters::GetOverlapPath()
{
	return sOverlapPath;
}