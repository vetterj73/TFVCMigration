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
	iCorrMaxColsToUse = 1024;			// Max cols will be used in alignment for a correlation pair if bApplyCorrSizeUpLimit==true
	iCorrMaxRowsToUse = 1024;			// Max rows will be used in alignment for a correlation pair if bApplyCorrSizeUpLimit==true
	iCorrPairMinRoiSize = 20;			// The Size of Roi of correlation pair need >= this value to process
	dMaskAreaRatioTh = 0.10;			// NGC will be used only if Mask area/Roi area > this value

	// Coarse correlation
	iCoarseMinDecim = 2;				// Minimum decimatin for coarse correlation
	iCoarseColSearchExpansion = 50;		// Search expansion in cols for coarse correlation
	iCoarseRowSearchExpansion = 300;	// Search expansion in rows for coarse correlation
	dCoarseResultReliableTh = 0.03;		// Threshold to decide whether coarse correlaiton result is reliable
	dCoarseResultAmbigTh = 0.8;			// Threshold to decide whehter coarse correlation ambiugous is too high

	// Fine correlation
	iFineBlockWidth = 256;				// Default width of fine correlation block
	iFineBlockHeight = 384;				// Default height of fine correlation block 
	iFineMaxBlocksInCol = 3;			// Max number of blocks in colum direction for fine correlation
	iFineMaxBlocksInRow = 3;			// Max Number of blocks in row direction for fine correlation

	iFineDecim = 2;						// Decimatin for fine correlation
	iFineColSearchExpansion = 30;		// Search expansion in cols for fine correlation if coarse correlation is successed
	iFineRowSearchExpansion = 30;		// Search expansion in rows for fine correlation if coarse correlation is successed

	// Fiducail search
	fidSearchMethod = FIDVSFINDER;		// Search method for fiducial
	//fidSearchMethod = FIDCYBERNGC;		// Search method for fiducial
	dVsFinderMinCorrScore = 0.5;		// The minimum correlation score for vsFinder
	dCyberNgcMinCorrScore = 0.5;		// The minimum correlation score for CyberNgc
	dFiducialSearchExpansionX = 6e-3;	// Fiducial search expansion in x and y of world space 
	dFiducialSearchExpansionY = 2e-3;
	dFidSearchExpXWithEdge = 2e-3;		// Fiducial search expansion in x and y of world space with panel edge information
	dFidSearchExpYWithEdge = 2e-3;

	// FovFov Alignment result check for each trigger
	bFovFovAlignCheck = true;			// Whether do FovFov alignment check for each trigger
	// Those values should be big enough to ignore small angle difference and small enought to catch exception
	dMaxColInconsistInPixel = 10;		// Max inconsist in pixel for columns and Rows for the same device
	dMaxRowInconsistInPixel = 15;
	dColAdjust4DiffDevice = 10;			// Adjust for the different device
	dRowAdjust4DiffDevice = 10;

	// Fiducial check
	bFiducialAlignCheck = true;			// Whether do fiducial check
	dMaxPanelCadScaleDiff = 5e-3;		// Max panel scale compared with calibration that can be tolerate
	dMaxFidDisScaleDiff = 2e-3;			// Max Fiducial distnace scale that can be tolerate
	dMaxSameFidInConsist = 5e-5;		// Max fiducial alignment result inconsist for the same fiducial (in meter)

	// board scale test (for RobustSolverCM::FlattenFiducial() )
	// these might be tied to the Fiducial check parameters above
	dAlignBoardStretchLimit = 0.00067 * 3;			// units meter / meter   maximum that the panel can shrink stretch
	// IPC-D-300G which allows a stretch of 200 um over 300 mm or 0.067%
	dAlignBoardSkewLimit = 0.00067 * 1.5;			// units meter / meter maximum skew of board (in affine transform)


	// Expansion from Cad image to create mask image in pixels
	iMaskExpansionFromCad = 10;			

	// Overlap
	iMinOverlapSize =100;				// Minimum overlap size for FovFov and FovCad

	// Whether use projective transform
	bUseProjectiveTransform = true;
	bUseCameraModelStitch = false;
	bUseCameraModelIterativeStitch = false;
	iSolverMaxIterations = 3;  // set to 2 or 3 once confident (large number now to catch convergence issues)

	// Adjust morph to create stitched image for component height
	bAdjustMorph4ComponentHeight = true;

	// Panel edge detection
	bDetectPanelEdge = false;				// Detection panel edge to reduce fiducial search area
	iLayerIndex4Edge = 0;					// Index of mosaic layer for panel edge detection	
		// Conveyor reference
	bConveyorLeft2Right = true;				// Conveyor moving direction
	bConveyorFixedFrontRail = true;			// conveyor fixed rail

			// Control parameters
	dMinLeadingEdgeGap = 2e-3;				// Minimum gap between image edge and nominal panel leading edge
	dLeadingEdgeSearchRange = 6e-3;			// Search range for panel leading edge
	dConveyorBeltAreaSize = 4e-3;			// The size of conveyor belt area that need to be ignored in leading edge detection

	// Number of Threads to use for processing
	NumThreads = 8;

	// debug flags
	bSaveFiducialOverlaps = false;
	bSaveOverlaps = false;
	bSaveTransformVectors = false;
	bSavePanelEdgeDebugImages = false; 
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