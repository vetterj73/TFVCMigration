#include "EquationWeights.h"

// For singleton pattern
EquationWeights* EquationWeights::ptr = 0;
EquationWeights& EquationWeights::Instance()
{
	if( ptr == NULL )
		ptr = new EquationWeights();

	return *ptr;
}

EquationWeights::EquationWeights(void)
{
// weights for calibration related constrains
	wRxy = 5e7;			// Rotation match (m1 = -m3)
	wMxy = 5e7;			// Magnification match (m0 = m4)
	wRcal = 5e6;		// Rotation m1/m3 match calibration
	wMcal = 2.5e6;		// Magnification m0/m4 (pixel size) match calibtation
	wYcent = 1e3;		// Fov center Y position match calibration
	wYdelta = 1e4;		// distance between cameras in Y match calibration
	wXdelta = 2.5e3;	// distance between cameras in X match calibration
	wXIndexwt = 0.01;	// Position of the FOV in X

	// for projective transform
	wPMEq = 1e12;		// m6 = m10 and m7 = m11
	wPM89 = 1e12;		// M8 = 0 and M9 = 0
	wPMNext = 2e11;		// M10 = Next camera/Triger M10, M11 = Next camera/triger M11

	// Parameters of weight for Fov and Fov overlap
	_dWeightFovFov = 2e5;
	_dMinFovFovLimit = 0.01;
	_dMaxFovFovAmbig = 0.9;

	// Parameters of weight for Cad and Fov overlap
	_dWeightCadFov = 2e4;
	_dMinCadFovLimit = 0.01;
	_dMaxCadFovAmbig = 0.9;

	// Parameters of weight for Fiducial and Fov overlap
	_dWeightFidFov = 1e6;
	_dMinFidFovLimit = 0.01;
	_dMaxFidFovAmbig = 0.9;
}

EquationWeights::~EquationWeights(void)
{
}

// Calculate weigtht for correlation pair
// pPair
double EquationWeights::CalWeight(CorrelationPair* pPair)
{
	if(!pPair->IsProcessed())
		return(0);

	CorrelationResult result = pPair->GetCorrelationResult();
	double dCorrScore = result.CorrCoeff;
	double dAmbig = result.AmbigScore;

	double dWeight = 0;

	switch(pPair->GetOverlapType())
	{
		case Fov_To_Fov:
		{
			dWeight = fabs(dCorrScore) *(_dMaxFovFovAmbig-dAmbig) - _dMinFovFovLimit;
			dWeight *= _dWeightFovFov;
		}
		break;

		case Cad_To_Fov:
		{
			dWeight = fabs(dCorrScore) *(_dMaxCadFovAmbig-dAmbig) - _dMinCadFovLimit;
			dWeight *= _dWeightCadFov;
		}
		break;

		case Fid_To_Fov:
		{
			dWeight = fabs(dCorrScore) *(_dMaxFidFovAmbig-dAmbig) - _dMinFidFovLimit;
			dWeight *= _dWeightFidFov;
		}
		break;
	}

	if(dWeight < 0) dWeight = 0;

	return(dWeight);
}


