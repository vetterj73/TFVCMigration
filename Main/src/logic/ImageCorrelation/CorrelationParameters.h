/* 
	Singleton Class for correlation parameters
*/

#include <string>
using std::string;

#pragma once

#define CorrParams CorrelationParameters::Instance() 

class CorrelationParameters
{
public:
	static CorrelationParameters& Instance();

protected:
	CorrelationParameters(void);
	~CorrelationParameters(void);

	static CorrelationParameters* _pInst;

public:
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

	// Correlation pair
	unsigned int iCorrPairMinRoiSize;		// The Size of Roi of correlation pair need >= this value to process
	double dMaskAreaRatioTh;				// NGC will be used only if Mask area/Roi area > this value

	// Fiducial search expansion
	double dFiducialSearchExpansionX;		// Fiducial search expansion in x and y of world space 
	double dFiducialSearchExpansionY;

	// Overlap
	unsigned int iMinOverlapSize;			// Minimum overlap size for FovFov and FovCad

	// debug flage
	bool bSaveOverlap;
	string sOverlapPath;
	bool bSaveStitchedImage;
	string sStitchPath;
};

