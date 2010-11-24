#include "CorrelationParameters.h"
#include <direct.h> //_mkdir

CorrelationParameters* CorrelationParameters::_pInst = 0;
CorrelationParameters& CorrelationParameters::Instance()
{
	if(_pInst == 0)
		_pInst = new CorrelationParameters();

	return(*_pInst);
}

CorrelationParameters::CorrelationParameters(void)
{
	// Coarse correlation
	iCoarseMinDecim = 2;				// Minimum decimatin for coarse correlation
	iCoarseColSearchExpansion = 50;		// Search expansion in cols for coarse correlation
	iCoarseRowSearchExpansion = 75;		// Search expansion in rows for coarse correlation
	dCoarseResultReliableTh = 0.03;			// Threshold to decide whether coarse correlaiton result is reliable

	// Fine correlation
	iFineBlockWidth = 256;				// Default width of fine correlation block
	iFineBlockHeight = 384;				// Default height of fine correlation block 
	iFineMaxBlocksInCol = 3;			// Max number of blocks in colum direction for fine correlation
	iFineMaxBlocksInRow = 3;			// Max Number of blocks in row direction for fine correlation

	iFineDecim = 2;						// Decimatin for fine correlation
	iFineColSearchExpansion = 20;		// Search expansion in cols for fine correlation if coarse correlation is successed
	iFineRowSearchExpansion = 20;		// Search expansion in rows for fine correlation if coarse correlation is successed

	// Correlation pair
	iCorrPairMinRoiSize = 20;			// The Size of Roi of correlation pair need >= this value to process
	dMaskAreaRatioTh = 0.05;			// NGC will be used only if Mask area/Roi area > this value

	// Fiducial search expansion
	dFiducialSearchExpansionX = 6e-3;	// Fiducial search expansion in x and y of world space 
	dFiducialSearchExpansionY = 2e-3;

	// Overlap
	iMinOverlapSize =100;				// Minimum overlap size for FovFov and FovCad

	// debug flage
	bSaveOverlap = false;
	sOverlapPath = "C:\\Temp\\Overlaps\\";
	bSaveStitchedImage = true;
	sStitchPath = "C:\\Temp\\";

	// make sure the directory exists 
	_mkdir(sStitchPath.c_str());
	_mkdir(sOverlapPath.c_str());

}

CorrelationParameters::~CorrelationParameters(void)
{
}
