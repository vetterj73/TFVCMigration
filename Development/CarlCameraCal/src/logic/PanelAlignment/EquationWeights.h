#pragma once

#include "CorrelationPair.h"

#define Weights EquationWeights::Instance()

class EquationWeights
{
public:
	static EquationWeights& Instance();

	double CalWeight(CorrelationPair* pPair);

protected:
	static EquationWeights* ptr;

	EquationWeights(void);
	~EquationWeights(void);

public:
// weights for calibration related constrains
	double wRxy;		// Rotation match (m1 = -m3)
	double wMxy;		// Magnification match (m0 = m4)
	double wRcal;		// Rotation m1/m3 match calibration
	double wMcal;		// Magnification m0/m4 (pixel size) match calibtation
	double wYRdelta;	// Angle different for adjacent cameras(Y) should match calibration
	double wYcent;		// Fov center Y position match calibration
	double wXcent;		// Fov center X position match calibration
	double wYdelta;		// distance between cameras in Y match calibration
	double wXdelta;		// distance between cameras in X match calibration

	double wYcentNoFid;	// Fov center Y position match calibration without fiducial equation (for single FOV only)
	double wXcentNoFid;	// Fov center X position match calibration without fiducial equation (for single FOV only)

	// for projective transform
	double wPMEq;		// m6 = m10 and m7 = m11
	double wPM89;		// M8 = 0 and M9 = 0
	double wPMNext;		// M10 = Next camera/Triger M10, M11 = Next camera/triger M11

	// For Camera Model
	double wZConstrain;
	double wZConstrainZero;
	double wXIndex;
	double wFidFlatBoardScale;
	double wFidFlatBoardRotation;
	double wFidFlatFiducialLateralShift;
	double wFidFlatFlattenFiducial;


private:
	// Parameters of weight for Fov and Fov overlap
	double _dWeightFovFov;
	double _dMinFovFovLimit;
	double _dMaxFovFovAmbig;

	// Parameters of weight for Cad and Fov overlap
	double _dWeightCadFov;
	double _dMinCadFovLimit;
	double _dMaxCadFovAmbig;

	// Parameters of weight for Fiducial and Fov overlap
	double _dWeightFidFov;
	double _dMinFidFovLimit;
	double _dMaxFidFovAmbig;
};

