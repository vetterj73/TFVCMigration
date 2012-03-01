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
	dCalScale = 1.0;	// Scale of calibration weight
	wRxy = 5e7;			// Rotation match (m1 = -m3)
	wMxy = 5e7;			// Magnification match (m0 = m4)
	wRcal = 5e6;		// Rotation m1/m3 match calibration
	wMcal = 2.5e6;		// Magnification m0/m4 (pixel size) match calibtation
	wYRdelta = 5e6;		// Angle different for adjacent cameras(Y) should match calibration
	wYcent = 1e3;		// Fov center Y position match calibration
	wXcent = 1e-1;		// Fov center X position match calibration
	wYdelta = 1e4;		// distance between cameras in Y match calibration
	wXdelta = 1e3;		// distance between cameras in X match calibration

		// for projective transform
	wPMEq = 1e12;		// m6 = m10 and m7 = m11
	wPM89 = 1e12;		// M8 = 0 and M9 = 0
	wPMNext = 2e11;	// M10 = Next camera/Triger M10, M11 = Next camera/triger M11

	// for solve without fiducial and panel edge
	wYcentNoFid = 1e5;	// Fov center Y position match calibration without fiducial equation (for single FOV only)
	wXcentNoFid = 1e5;	// Fov center X position match calibration without fiducial equation (for single FOV only)

	// for panel edge detection
	wXbyEdge = wXcent*1e5;	// x offset based on edge detection
	wRbyEdge = wRcal*1e3;	// FOV rotation based on edge detection

	// For Camera Model
	wZConstrain = 10;   // lightly constrain Z model to flat
	wZConstrainZero = 1e6;  // strongly constrain unused Z model terms to flat
	wXIndex = 0.01;		// constrain xTrig, yTrig, thetaTrig to match expected values
	wFidFlatBoardScale   = 1e3;
	wFidFlatBoardRotation = 1e2;
	wFidFlatFiducialLateralShift = 1e3;
	wFidFlatFlattenFiducial = 1e5;
	RelativeFidFovCamModWeight = 1e-3;

	// Parameters of weight for Fov and Fov overlap
	_dWeightFovFov = 2e5;
	_dMinFovFovLimit = 0.01;
	_dMaxFovFovAmbig = 0.9;

	// Parameters of weight for Cad and Fov overlap
	_dWeightCadFov = 2e4;
	_dMinCadFovLimit = 0.01;
	_dMaxCadFovAmbig = 0.9;

	// Parameters of weight for Fiducial and Fov overlap
	//_dWeightFidFov = 2e3;
	_dWeightFidFov = 2e6;
	_dMinFidFovLimit = 0.03;
	_dMaxFidFovAmbig = 0.8;
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


void EquationWeights::SetCalibrationScale(double dValue)
{

	double dOldValue = dCalScale;
	dCalScale = dValue;

	double dAdjustScale = dCalScale/dOldValue;

	wRxy *= dAdjustScale;		// Rotation match (m1 = -m3)
	wMxy *= dAdjustScale;		// Magnification match (m0 = m4)
	wRcal *= dAdjustScale;		// Rotation m1/m3 match calibration
	wMcal *= dAdjustScale;		// Magnification m0/m4 (pixel size) match calibtation
	wYRdelta *= dAdjustScale;	// Angle different for adjacent cameras(Y) should match calibration
	wYcent *= dAdjustScale;		// Fov center Y position match calibration
	wXcent *= dAdjustScale;		// Fov center X position match calibration
	wYdelta *= dAdjustScale;	// distance between cameras in Y match calibration
	wXdelta *= dAdjustScale;	// distance between cameras in X match calibration

		// for projective transform
	wPMEq *= dAdjustScale;		// m6 = m10 and m7 = m11
	wPM89 *= dAdjustScale;		// M8 = 0 and M9 = 0
	wPMNext *= dAdjustScale;	// M10 = Next camera/Triger M10, M11 = Next camera/triger M11
}


