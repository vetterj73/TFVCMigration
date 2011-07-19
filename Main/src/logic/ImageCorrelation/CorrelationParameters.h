/* 
	Singleton Class for correlation parameters
*/

#include <string>
using std::string;

typedef enum {                           /* Don't change order of enums */
   BGGR,
   GBRG,
   GRBG,
   RGGB
}BayerPattern;

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

	// Expansion from Cad image to create mask image in pixels
	int iMaskExpansionFromCad;
	
	// Overlap
	unsigned int iMinOverlapSize;			// Minimum overlap size for FovFov and FovCad

	// Support Bayer(color image)
	bool bGrayScale;						// Grey scale image or Bayer(color) image

	// debug flage
	bool bSaveFiducialOverlaps;
	bool bSaveOverlaps;
	bool bSaveMaskVectors;
	unsigned int NumThreads;
	string sDiagnosticPath;
	string GetOverlapPath();

private:
	string sOverlapPath;
};

