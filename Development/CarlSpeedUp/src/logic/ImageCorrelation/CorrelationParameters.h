/* 
	Singleton Class for correlation parameters
*/

#include <string>
using std::string;

typedef enum {                           /* Don't change order of enums */
   FIDREGOFF,
   FIDVSFINDER,
   FIDCYBERNGC
}FiducialSearchMethod;

#pragma once

#define CorrelationParametersInst CorrelationParameters::Instance() 

class CorrelationParameters
{
public:
	static CorrelationParameters& Instance();

protected:
	CorrelationParameters(void);
	~CorrelationParameters(void);

	static CorrelationParameters* _pInst;

public:
	// Correlation pair
	unsigned int iCorrMaxColsToUse;			// Max cols will be used in alignment for a correlation pair if bApplyCorrSizeUpLimit==true
	unsigned int iCorrMaxRowsToUse;			// Max rows will be used in alignment for a correlation pair if bApplyCorrSizeUpLimit==true
	unsigned int iCorrPairMinRoiSize;		// The Size of Roi of correlation pair need >= this value to process
	double dMaskAreaRatioTh;				// NGC will be used only if Mask area/Roi area > this value

	// Coarse correlation
	unsigned int iCoarseMinDecim;			// Minimum decimatin for coarse correlation
	unsigned int iCoarseColSearchExpansion; // Search expansion in cols for coarse correlation
	unsigned int iCoarseRowSearchExpansion; // Search expansion in rows for coarse correlation
	double dCoarseResultReliableTh;			// Threshold to decide whether coarse correlaiton result is reliable
	double dCoarseResultAmbigTh;			// Threshold to decide whehter coarse correlation ambiugous is too high

	// Fine correaltion
	unsigned int iFineBlockWidth;			// Default width of fine correlation block
	unsigned int iFineBlockHeight;			// Default height of fine correlation block 
	unsigned int iFineMaxBlocksInCol;		// Max number of blocks in colum direction for fine correlation 
	unsigned int iFineMaxBlocksInRow;		// Max Number of blocks in row direction for fine correlation
	
	unsigned int iFineDecim;				// Decimatin for fine correlation
	unsigned int iFineColSearchExpansion;	// Search expansion in cols for fine correlation if coarse correlation is successed
	unsigned int iFineRowSearchExpansion;	// Search expansion in rows for fine correlation if coarse correlation is successed

	// Fiducail search
	FiducialSearchMethod fidSearchMethod;	// Search method for fiducial
	double dVsFinderMinCorrScore;			// The minimum correlation score for vsFinder
	double dCyberNgcMinCorrScore;			// The minimum correlation score for vsFinder
	double dFiducialSearchExpansionX;		// Fiducial search expansion in x and y of world space 
	double dFiducialSearchExpansionY;
	double dFidSearchExpXWithEdge;			// Fiducial search expansion in x and y of world space with panel edge information
	double dFidSearchExpYWithEdge;
	double dFidSearchExpXWithEdge1Fid;		// Fiducial search expansion in x and y of world space with panel edge information and one fiducial information
	double dFidSearchExpYWithEdge1Fid;

	// FovFov Alignment result check for each trigger
	bool bFovFovAlignCheck;					// Whether do FovFov alignment check for each trigger
	// Those values should be big enough to ignore small angle difference and small enought to catch exception
	double dMaxColInconsistInPixel;			// Max inconsist in pixel for columns and Rows for the same device
	double dMaxRowInconsistInPixel;
	double dColAdjust4DiffDevice;			// Adjust for the different device
	double dRowAdjust4DiffDevice;

	// Fiducial check
	bool bFiducialAlignCheck;				// Whether do fiducial alignment check
	double dMaxPanelCadScaleDiff;			// Max panel scale compared with calibration that can be tolerate
	double dMaxFidDisScaleDiff;				// Max Fiducial distnace scale that can be tolerate
	double dMaxSameFidInConsist;			// Max fiducial alignment result inconsist for the same fiducial (in meter)

	// board scale test (for RobustSolverCM::FlattenFiducial() )
	// these might be tied to the Fiducial check parameters above
	double dAlignBoardStretchLimit;			// units meter / meter   maximum that the panel can shrink stretch
	// IPC-D-300G which allows a stretch of 200 um over 300 mm or 0.067%
	double dAlignBoardSkewLimit;			// units meter / meter maximum skew of board (in affine transform)

	// Whether use projective transform
	bool bUseProjectiveTransform;
	
	// Whether use camera model image stitching
	bool bUseCameraModelStitch;
	bool bUseCameraModelIterativeStitch;
	bool bUseTwoPassStitch;					// do complete alignement based on coarse overlaps then chop fine overlaps
	bool bCoarsePassDone;
	bool bCoarseOnly;

	unsigned int iSolverMaxIterations;
	
	// Overlap
	unsigned int iMinOverlapSize;			// Minimum overlap size for FovFov and FovCad

	// Adjust morph to create stitched image for component height
	bool bAdjustMorph4ComponentHeight;

	// Panel edge detection
	bool bDetectPanelEdge;					// Detection panel edge to reduce fiducial search area
	int	iLayerIndex4Edge;					// Index of mosaic layer for panel edge detection
		// Conveyor reference
	bool bConveyorLeft2Right;				// Conveyor moving direction
	bool bConveyorFixedFrontRail;			// conveyor fixed rail
	
		// Control parameters
	double dMinLeadingEdgeGap;				// Minimum gap between image edge and nominal panel leading edge
	double dLeadingEdgeSearchRange;			// Search range for panel leading edge
	double dConveyorBeltAreaSize;			// The size of conveyor belt area that need to be ignored in leading edge detection

	// for supplement Overlaps	
	unsigned int iMaxSupOverlaps;			// Maximum supplememt overlaps to prevent add supplement overlaps for a messed up panel
	
	// Number of Threads to use for processing
	unsigned int NumThreads;

	// debug flage
	bool bSaveFiducialOverlaps;
	bool bSaveOverlaps;
	bool bSaveTransformVectors;
	bool bSavePanelEdgeDebugImages;
	string sDiagnosticPath;
	string GetOverlapPath();

	void SetCoarsePassDone(bool temp);
	bool GetCoarsePassDone();

private:
	string sOverlapPath;
};

