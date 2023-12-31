#include "RobustSolverCM.h"

#include "EquationWeights.h"
#include "lsqrpoly.h"
#include "Logger.h"

#include "rtypes.h"
#include "dot2d.h"
extern "C" {
#include "ucxform.h"
#include "warpxy.h"
}


#define CorrelationParametersInst CorrelationParameters::Instance() 

/*
	Camera Model version of RobustSolver
	Solves for a smaller number of unknows than the FOV version
	
	Expectation is that the board is a rigid object which translates only in X, Y, Theta_Z as it goes under each SIM
	
	Each SIM may be at its own Z position and have its rotation around X, but the board will not bounce up and down randomly 
	during the scan

	Given these assumptions we solve for
	per trigger X, Y, Theta (3 terms per trigger)
	per board Z shape (10 terms per board)
	per SIM Z and theta_X (first SIM has these included in Z shape, following SIMs have 2 additional terms per SIM)
*/
/// <summary>
/// Constructor for RobustSolverCM
/// </summary>
/// <param name="pFovOrderMap"></param>
/// <param name="iMaxNumCorrelations"></param>
/// <param name="pSet"></param>
/// create new arrays for solver, initialize paramters, zero out the arrays
RobustSolverCM::RobustSolverCM(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet): 	RobustSolver( pFovOrderMap)
{

	// Constant parameter
	_iNumBasisFunctions = 4;
	_iNumZTerms = 10;		// Z terms -- 3rd order bi-variate polynomial
	_iNumParamsPerIndex = 3;  // per trigger X, Y, theta_Z values

	// Input 
	_pSet = pSet;
	_iMaxNumCorrelations = iMaxNumCorrelations;

	// Max Camera number per device, device number and trigger number
	_iNumCameras = CountCameras();  // number per SIM, must be the same for All SIMs
	_iTotalNumberOfSubTriggers = _pSet->GetMosaicTotalNumberOfSubTriggers();
		// have only implemented through 3 SIMs
	int numLayers = _pSet->GetNumMosaicLayers();
	_iNumDevices = _pSet->GetNumDevice();
	if (_iNumDevices > 3)
			LOG.FireLogEntry(LogTypeError, "More than 3 SIMs");	
	
	// Decide Matrics sizes
		// numbers of combinations of SIM pairs
	int nCombinations[6] = {0,0,1,3,6,10}; // zero or 1 SIM have 0 pairs, never expect more than 5 SIMs
	_iNumCalDriftTerms = _iNumDevices * _iNumCameras * 2;
		// adapt to SIM to SIM mounting differences 
		// width needed for CM fit
	_iStartColZTerms = _iTotalNumberOfSubTriggers * _iNumParamsPerIndex;
	_iMatrixWidth = _iStartColZTerms + _iNumZTerms + (_iNumDevices-1)*2;// additional width needed for Z mount differences
		// allocate larger block for iterative solver, 
		// later will merge the two methods and get rid of this step
	unsigned int iMatrixWidthAllocated = _iMatrixWidth  + _iNumCalDriftTerms;
	_iMatrixHeight = _iNumZTerms										// constrain Z
					+ _iTotalNumberOfSubTriggers * _iNumParamsPerIndex		// constrain Xtrig, Ytrig, theta_trig
					+ _iNumCalDriftTerms + _iNumDevices*2				// constrain cal drift to 0, sums per device and direction to 0
					+ 2 * nCombinations[_iNumDevices]					// constrain z mount terms
					+ 2*_iMaxNumCorrelations;
		// TODO add height for total number of constraint terms
	_iMatrixSize = iMatrixWidthAllocated * _iMatrixHeight;

	// Allocate matrics
	_dMatrixA = new double[_iMatrixSize];
	_dMatrixACopy = new double[_iMatrixSize];
	
	_dVectorB = new double[_iMatrixHeight];
	_dVectorBCopy = new double[_iMatrixHeight];

	_dVectorX = new double[iMatrixWidthAllocated];
	
	_iLengthNotes = 200;

	// For debug
	//if(true)  // TODO make this selectable
	//{
	_pdWeights = new double[_iMatrixHeight];
	_pcNotes	 = new char*[_iMatrixHeight];
	for(unsigned int i=0; i<_iMatrixHeight; i++)
		_pcNotes[i] = new char[_iLengthNotes];
	//}

	// Set to default values
	ZeroTheSystem();

}
RobustSolverCM::~RobustSolverCM()
{
	delete [] _pdWeights;
	for(unsigned int i=0; i<_iMatrixHeight; i++)
		delete [] _pcNotes[i] ;
	delete [] _pcNotes;
}

void RobustSolverCM::ZeroTheSystem()
{
	RobustSolver::ZeroTheSystem();

	for(unsigned int i=0; i<_iMatrixHeight; i++)
	{
		_pdWeights[i] = 0.0;
		sprintf_s(_pcNotes[i], _iLengthNotes, "");
	}

	for (unsigned int dev(0); dev < MAX_NUMBER_DEVICES; dev++)
		for (unsigned int i=0; i < NUMBER_Z_BASIS_FUNCTIONS; i++)
			for (unsigned int j(0); j < NUMBER_Z_BASIS_FUNCTIONS; j++)
				_zCoef[dev][i][j] = 0.0;
	double resetTransformValues[3][3] =  {{1.0,0,0},{0,1.0,0},{0,0,1}};
	_Board2CAD.SetMatrix( resetTransformValues );
}

void RobustSolverCM::ConstrainZTerms()
{
	// board warp is modeled by 10 terms, a bi-variate cubic equation
	// the last columns of A describe board shape in Z
	// We'll give these a mild weight toward 0
	// EXCEPT for those that describe high order terms in Y
	// if there is only a single camera we're in trouble -- must rely on old means of measuring board height
	// if _iNumCameras = 2 then we can only get a single range term (only y^0 terms)
	// if 3 cameras we get 2 range points and can est. y^0 and y^1 terms
	// if 4 cameras can add y^2 term
	// if 5 or more cameras (4 or more range points) then can add y^3 point
	unsigned int ZfitTerms; // number of active terms
	double temp(Weights.wZConstrain);
	//*AlignAlgParam& arg(AlignAlgParam::instance());
	if (_iNumCameras == 1)
		ZfitTerms = 4;  // Single camera case may be a problem, board could warp in X, but hard to measure
	else if (_iNumCameras == 2)
		ZfitTerms = 4;  // perhaps should be 5 (allow tilt in Y direction)
	else if (_iNumCameras == 3)
		ZfitTerms = 7;
	else if (_iNumCameras == 4)
		ZfitTerms = 9;
	else
		ZfitTerms = 10;

	double* pdRow; // pointer to location in A for setting Z
	for(unsigned int zIndex(0); zIndex< _iNumZTerms + (_iNumDevices-1)*2; ++zIndex)// must constrain for each SIM
	{
		// adding Z control terms to the top of the array (for now), so row num is equal to row in A
		unsigned int Acol(_iStartColZTerms + zIndex);
		pdRow = _dMatrixA + _iCurrentRow * _iMatrixWidth;
		if (zIndex < ZfitTerms)
			pdRow[Acol] = Weights.wZConstrain;
		else if (zIndex < _iNumZTerms)
			pdRow[Acol] = Weights.wZConstrainZero;
		else
			pdRow[Acol] = Weights.wZConstrain;

		// add log of weights and debug notes
		_pdWeights[_iCurrentRow] = pdRow[Acol];
		_iCurrentRow++; // track how many rows of MatrixA have been used
	}

	// Match between different devices
	if (_iNumDevices > 1)
	{	
		double wt(Weights.wSimMountZMatch); // TODO TODO !!! add parameter

		for(unsigned int iIndex1=0; iIndex1<_iNumDevices-1; iIndex1++)
		{
			for(unsigned int iIndex2 = iIndex1+1; iIndex2<_iNumDevices; iIndex2++)
			{	
				// term 0 should match
				pdRow = _dMatrixA + _iCurrentRow * _iMatrixWidth;	
				pdRow[ ColumnZTerm(0, iIndex1) ] = wt; 
				pdRow[ ColumnZTerm(0, iIndex2) ] = -wt;
				_pdWeights[_iCurrentRow] = wt;
				_iCurrentRow++; // 

				// term 4 should match
				pdRow = &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
				pdRow[ ColumnZTerm(4, iIndex1) ] = wt;
				pdRow[ ColumnZTerm(4, iIndex2) ] = -wt;
				_pdWeights[_iCurrentRow] = wt;
				_iCurrentRow++; 
			}
		}

		if (_iNumDevices > 3)
			LOG.FireLogEntry(LogTypeError, "More than 3 SIMs");	
	}
}

unsigned int RobustSolverCM::CountCameras()
{
	unsigned int iNumLayer = _pSet->GetNumMosaicLayers();
	unsigned int iMaxNumCam = 0;
	for(unsigned int i =0; i<iNumLayer; i++)
	{
		unsigned int iNumCam = _pSet->GetLayer(i)->GetNumberOfCameras();
		if(iMaxNumCam < iNumCam) 
			iMaxNumCam = iNumCam;
	}

	return(iMaxNumCam);
}

unsigned int RobustSolverCM::ColumnZTerm(unsigned int term, unsigned int deviceNum)
{
	// calculate which column a particular Z term is in based on term number and devcie number
	// order is.  Used to allow different SIMs to be mounted at different height and theta_X positions.
	// sim0 a0 - a9, sim1 a0, sim1 a4, sim2 a0, sim2 a4 etc.
	// 
	unsigned int col;
	if (deviceNum == 0 || term != 0 && term !=4)
	{
		col = _iStartColZTerms + term;
		return col;
	}
	else // device > 0 and terms 0 or 4
	{ 
		col = _iStartColZTerms + _iNumZTerms + (deviceNum-1)*2 + term/4;
		return col;
	}
}

/// <summary>
/// Add contraints for panel X, Y, theta for each trigger
/// </summary>
/// Each trigger is constrained toward 0 angle and X and Y locations that match the expected locations.
void RobustSolverCM::ConstrainPerTrig(bool bPinPanelWithCalibration)
{
	// add constraints for each trigger  
	double dFovOriginU = 0.0;
	double dFovOriginV = 0.0;
	// for each trigger we add a constrain the xTrig, yTrig, theataTrig values to
	// their expected values
	// need to visit each trigger just once in the same order as _pFovOrderMap
	
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		unsigned int iLayerIndex = k->first.LayerIndex;
		unsigned int iCamIndex = k->first.CameraIndex;
		if (_pSet->GetLayer(iLayerIndex)->IsFirstCamOfSubTrigger(iCamIndex))  // first camera of subtrig
		{
			unsigned int indexID( k->second );
			unsigned int iTrigIndex( k->first.TriggerIndex);
			TransformCamModel camCal = 
				 _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
			complexd xySensor;
			camCal.SPix2XY(dFovOriginU, dFovOriginV, &xySensor.r, &xySensor.i);

			double dWeight = Weights.wXIndex;
			// Pin the first sub-trig if it is necessary
			if(bPinPanelWithCalibration && iTrigIndex == 0 && iCamIndex == 0)
				dWeight = Weights.wXIndex_PinXY;
			
			// constrain x direction
			double* pdRow = _dMatrixA +_iCurrentRow*_iMatrixWidth;
			unsigned int beginIndex( indexID * _iNumParamsPerIndex);
			pdRow[beginIndex+0] = dWeight;	// b VALUE IS NON_ZERO!!
			pdRow[beginIndex+2] = -dWeight * xySensor.i;
			_dVectorB[_iCurrentRow] = dWeight * 
				(_pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(2)
				- xySensor.r);

			_pdWeights[_iCurrentRow] = dWeight;
			_iCurrentRow++;

			// constrain yTrig value
			pdRow = _dMatrixA +_iCurrentRow*_iMatrixWidth;
			pdRow[beginIndex+1] = dWeight;	// b VALUE IS NON_ZERO!!
			pdRow[beginIndex+2] = dWeight * xySensor.r;
			_dVectorB[_iCurrentRow] = dWeight * 
				(_pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(5)
				- xySensor.i);
			_pdWeights[_iCurrentRow] = dWeight;
			_iCurrentRow++;
			
			// constrain theata to zero
				// Pin the first sub-trig if it is necessary
			if(bPinPanelWithCalibration && iTrigIndex == 0 && iCamIndex == 0)
				dWeight = Weights.wXINdex_PinTheta;

			pdRow = _dMatrixA +_iCurrentRow*_iMatrixWidth;
			pdRow[beginIndex+2] = dWeight;
			_pdWeights[_iCurrentRow] = dWeight;
			_iCurrentRow++;
		}
	}
}

bool RobustSolverCM::AddAllLooseConstraints(
	bool bPinPanelWithCalibration, 
	bool bUseNominalTransform)
{
	ConstrainZTerms();
	ConstrainPerTrig(bPinPanelWithCalibration);

	return(true);
}

// Add constraint base on panel edge
bool RobustSolverCM::AddPanelEdgeContraints(
	MosaicLayer* pLayer, unsigned int iCamIndex, unsigned int iTrigIndex,
	double dXOffset, double dSlope, bool bSlopeOnly)
{
	// dXOffset is the X position of the 0,0 pixel of the imager,
	// dSlope is the rotation (estimate) of the imager
	// 
	// Position of equation in Matirix
	FovIndex index(pLayer->Index(), iTrigIndex, iCamIndex); 
	unsigned int indexID( (*_pFovOrderMap)[index] );
	unsigned int deviceNum = _pSet->GetLayer(index.LayerIndex)->DeviceIndex();
	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
	unsigned int iFOVPosA( indexID * _iNumParamsPerIndex);
	
	// Add a equataions 
	// model the equations after the fiducial alignment set
	// x_trig - s_y * theta_trig = dsx/dz * Z = x_fid - s_x
	double w = Weights.wXbyEdge;
	TransformCamModel camCalA = 
				_pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
	double xSensorA, ySensorA, dxSensordzA, dySensordzA;
	camCalA.SPix2XY(0, 0, &xSensorA, &ySensorA);
	camCalA.dSPix2XY(0, 0, &dxSensordzA, &dySensordzA);

	// approximate position of the 0,0 pixel on the surface of the board
	double boardX = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(2);
	double boardY = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(5);
	double* Zpoly;
	Zpoly = new double[_iNumZTerms];
	Zpoly[0] = pow(boardX,0) * pow(boardY,0);
	Zpoly[1] = pow(boardX,1) * pow(boardY,0);
	Zpoly[2] = pow(boardX,2) * pow(boardY,0);
	Zpoly[3] = pow(boardX,3) * pow(boardY,0);
	Zpoly[4] = pow(boardX,0) * pow(boardY,1);
	Zpoly[5] = pow(boardX,1) * pow(boardY,1);
	Zpoly[6] = pow(boardX,2) * pow(boardY,1);
	Zpoly[7] = pow(boardX,0) * pow(boardY,2);
	Zpoly[8] = pow(boardX,1) * pow(boardY,2);
	Zpoly[9] = pow(boardX,0) * pow(boardY,3);

	//  for X_trig
	if(!bSlopeOnly)
	{
		pdRow[iFOVPosA] += w;
		pdRow[iFOVPosA+2] += -ySensorA * w;
		for (unsigned int j(0); j < _iNumZTerms; j++)
				pdRow[ColumnZTerm(j, deviceNum)] = Zpoly[j] * dxSensordzA * w;
		_dVectorB[_iCurrentRow] = w * (dXOffset - xSensorA);
		_pdWeights[_iCurrentRow] = w;
		_iCurrentRow++;
	}

	// for theta_trig
	w = Weights.wRbyEdge; 
	pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
	pdRow[iFOVPosA+2] += w;
	_dVectorB[_iCurrentRow] = w * dSlope;
	sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "BoardEdge:%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e", 
		index.LayerIndex,
		iTrigIndex, iCamIndex,
		xSensorA, ySensorA, dXOffset, dSlope );
	_pdWeights[_iCurrentRow] = w;
	_iCurrentRow++;
	return(true);
}

bool RobustSolverCM::AddFovFovOvelapResults(FovFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// First Fov's information
	unsigned int iLayerIndexA = pOverlap->GetFirstMosaicLayer()->Index();
	unsigned int iTrigIndexA = pOverlap->GetFirstTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetFirstCameraIndex();
	FovIndex index1(iLayerIndexA, iTrigIndexA, iCamIndexA); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index1] *_iNumParamsPerIndex;
	unsigned int deviceNumA = _pSet->GetLayer(index1.LayerIndex)->DeviceIndex();
	
	// Second Fov's information
	unsigned int iLayerIndexB = pOverlap->GetSecondMosaicLayer()->Index();
	unsigned int iTrigIndexB = pOverlap->GetSecondTriggerIndex();
	unsigned int iCamIndexB = pOverlap->GetSecondCameraIndex();
	FovIndex index2(iLayerIndexB, iTrigIndexB, iCamIndexB); 
	unsigned int iFOVPosB = (*_pFovOrderMap)[index2] *_iNumParamsPerIndex;
	unsigned int deviceNumB = _pSet->GetLayer(index2.LayerIndex)->DeviceIndex();
	
	double* pdRow;
	list<CorrelationPair> pCoarsePairList;
	list<CorrelationPair>* pPairList; 
	pCoarsePairList.clear();
	if (CorrelationParametersInst.bUseTwoPassStitch && !CorrelationParametersInst.bCoarsePassDone) //@todo AND first time through......
	{
		pCoarsePairList.push_back(*pOverlap->GetCoarsePair());
		pPairList = &pCoarsePairList;
	}
	else
		pPairList = pOverlap->GetFinePairListPtr();

	for(list<CorrelationPair>::iterator i= pPairList->begin(); i!=pPairList->end(); i++)
	{
		// Skip any fine that is not processed or not good
		if(!i->IsProcessed() || !i->IsGoodForSolver())
			continue;

		// validation check for correlation pair
		CorrelationResult result;
		bool bFlag = i->GetCorrelationResult(&result);
		if(!bFlag) 
			continue;

		double w = Weights.CalWeight(&(*i));
		// include 0 weight rows for now....
		//if(w <= 0) 
		//	continue;
		
		// Get Centers of ROIs
		double rowImgA = i->GetFirstRoi().RowCenter();
		double colImgA = i->GetFirstRoi().ColumnCenter();

		double rowImgB = i->GetSecondRoi().RowCenter();
		double colImgB = i->GetSecondRoi().ColumnCenter();

		// Get offset
		double offsetRows = result.RowOffset;
		double offsetCols = result.ColOffset;
		rowImgB += offsetRows;
		colImgB += offsetCols;

		// S and dS/dz in SIM space
		TransformCamModel camCalA = 
				 _pSet->GetLayer(iLayerIndexA)->GetImage(iTrigIndexA, iCamIndexA)->GetTransformCamCalibration();
		double xSensorA, ySensorA, dxSensordzA, dySensordzA;
		camCalA.SPix2XY(colImgA, rowImgA, &xSensorA, &ySensorA);
		camCalA.dSPix2XY(colImgA, rowImgA, &dxSensordzA, &dySensordzA);
		
		TransformCamModel camCalB = 
				 _pSet->GetLayer(iLayerIndexB)->GetImage(iTrigIndexB, iCamIndexB)->GetTransformCamCalibration();
		double xSensorB, ySensorB, dxSensordzB, dySensordzB;
		camCalB.SPix2XY(colImgB, rowImgB, &xSensorB, &ySensorB);
		camCalB.dSPix2XY(colImgB, rowImgB, &dxSensordzB, &dySensordzB);

		// Approximate X and y in board space for Z calculation
		pair<double, double> imgAOverlapCenter = i->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
		pair<double, double> imgBOverlapCenter = i->GetSecondImg()->ImageToWorld(rowImgB,colImgB);
		double boardX = (imgAOverlapCenter.first + imgBOverlapCenter.first ) / 2.0;
		double boardY = (imgAOverlapCenter.second + imgBOverlapCenter.second) / 2.0;
		
		// Generate the Z poly terms
		double* Zpoly;
		Zpoly = new double[_iNumZTerms];
		// the z poly can be thought of as a partially populated 3x3 matrix:
		// 				X**0 * Y**0, X**1 * Y**0, X**2 * Y**0, X**3 * Y**0,
		//              X**0 * Y**1, X**1 * Y**1, X**2 * Y**1,      0,
		//              X**0 * Y**2, X**1 * Y**2,      0,           0,
		//              X**0 * Y**3       0,           0,           0

		Zpoly[0] = pow(boardX,0) * pow(boardY,0);
		Zpoly[1] = pow(boardX,1) * pow(boardY,0);
		Zpoly[2] = pow(boardX,2) * pow(boardY,0);
		Zpoly[3] = pow(boardX,3) * pow(boardY,0);
		Zpoly[4] = pow(boardX,0) * pow(boardY,1);
		Zpoly[5] = pow(boardX,1) * pow(boardY,1);
		Zpoly[6] = pow(boardX,2) * pow(boardY,1);
		Zpoly[7] = pow(boardX,0) * pow(boardY,2);
		Zpoly[8] = pow(boardX,1) * pow(boardY,2);
		Zpoly[9] = pow(boardX,0) * pow(boardY,3);

		// X direction
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		pdRow[iFOVPosA] += w;
		pdRow[iFOVPosB] -= w;  // may be in same trigger number
		pdRow[iFOVPosA+2] += -ySensorA * w;
		pdRow[iFOVPosB+2] -= -ySensorB * w;
		// debug notes:
		// colImgA, rowImgA, colImgB, rowImgB, xSensorA, ySensorA, xSensorB, ySensorB
		// TODO TODO terms 0 and 4 based on SIM number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		for (unsigned int j(0); j < _iNumZTerms; j++)
		{
			pdRow[ColumnZTerm(j, deviceNumA)] = Zpoly[j] * ( dxSensordzA )* w;
			pdRow[ColumnZTerm(j, deviceNumB)] += Zpoly[j] * (-dxSensordzB)* w;
		}
		_dVectorB[_iCurrentRow] = -w * (xSensorA - xSensorB);
		sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FovFovCorr:L%d:T%d:C%d_L%d:T%d:C%d_%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
			pOverlap->GetFirstMosaicLayer()->Index(),
			pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
			pOverlap->GetSecondMosaicLayer()->Index(),
			pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
			i->GetIndex(),
			colImgA, rowImgA, colImgB, rowImgB, xSensorA, ySensorA, xSensorB, ySensorB);
		_pdWeights[_iCurrentRow] = w;
		_iCurrentRow++;

		// Y direction
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		pdRow[iFOVPosA+1] += w;
		pdRow[iFOVPosB+1] -= w;  // may be in same trigger number
		pdRow[iFOVPosA+2] += xSensorA * w;
		pdRow[iFOVPosB+2] -= xSensorB * w;
		// TODO TODO terms 0 and 4 based on SIM number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		for (unsigned int j(0); j < _iNumZTerms; j++)
		{
			pdRow[ColumnZTerm(j, deviceNumA)] = Zpoly[j] * ( dySensordzA )* w;
			pdRow[ColumnZTerm(j, deviceNumB)] += Zpoly[j] * (-dySensordzB)* w;
		}
		_dVectorB[_iCurrentRow] = -w * (ySensorA - ySensorB);
		sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FovFovCorr:L%d:T%d:C%d_L%d:T%d:C%d_%d,%.2e,%.2e,%.2e,%.2e,%.4e,%.4e", 
			pOverlap->GetFirstMosaicLayer()->Index(),
			pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
			pOverlap->GetSecondMosaicLayer()->Index(),
			pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
			i->GetIndex(),
			offsetRows, offsetCols,
			i->GetCorrelationResult().CorrCoeff, i->GetCorrelationResult().AmbigScore,
			boardX,boardY);
		_pdWeights[_iCurrentRow] = w;
	
		_iCurrentRow++;
		
		delete [] Zpoly;
	}
	return( true );
}

bool RobustSolverCM::AddCadFovOvelapResults(CadFovOverlap* pOverlap)
{
	// there are no CAD apperture fits in the camera model fit
	// TODO TODO   is this true????????
	return( true );
}

bool RobustSolverCM::AddFidFovOvelapResults(FidFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);
	
	// Fov's information
	unsigned int iLayerIndexA= pOverlap->GetMosaicLayer()->Index();
	unsigned int iTrigIndexA = pOverlap->GetTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetCameraIndex();
	FovIndex index(iLayerIndexA, iTrigIndexA, iCamIndexA); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index] *_iNumParamsPerIndex;
	unsigned int deviceNumA = _pSet->GetLayer(index.LayerIndex)->DeviceIndex();
	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	CorrelationPair* pPair = pOverlap->GetCoarsePair();

	// Validation check for correlation pair
	CorrelationResult result;
	bool bFlag = pPair->GetCorrelationResult(&result);
	if(!bFlag) 
		return(false);
	double w = Weights.CalWeight(pPair) * Weights.RelativeFidFovCamModWeight;
	if(w <= 0) 
		return(false);

	// Get Centers of ROIs (fiducial image is always the second one)
	double rowImgA = pPair->GetFirstRoi().RowCenter();
	double colImgA = pPair->GetFirstRoi().ColumnCenter();

	double rowImgB = pPair->GetSecondRoi().RowCenter();
	double colImgB = pPair->GetSecondRoi().ColumnCenter();
	double dFidRoiCenX, dFidRoiCenY; // ROI center of fiducial image (not fiducail center or image center) in world space
	pOverlap->GetFidImage()->ImageToWorld(rowImgB, colImgB, &dFidRoiCenX, &dFidRoiCenY);

	// Get offset
	double offsetRows = result.RowOffset;
	double offsetCols = result.ColOffset;
	rowImgA -= offsetRows;
	colImgA -= offsetCols;
	
	//  S and dS/dz in SIM space
	TransformCamModel camCalA = 
				_pSet->GetLayer(iLayerIndexA)->GetImage(iTrigIndexA, iCamIndexA)->GetTransformCamCalibration();
	double xSensorA, ySensorA, dxSensordzA, dySensordzA;
	camCalA.SPix2XY(colImgA, rowImgA, &xSensorA, &ySensorA);
	camCalA.dSPix2XY(colImgA, rowImgA, &dxSensordzA, &dySensordzA);
		
	// Approximate x and y in board space for Z calculation
	pair<double, double> imgAOverlapCenter = pPair->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
	double boardX = imgAOverlapCenter.first;
	double boardY = imgAOverlapCenter.second;

	double* Zpoly;
	Zpoly = new double[_iNumZTerms];
	// the z poly can be thought of as a partially populated 4x4 matrix:
	// 				X**0 * Y**0, X**1 * Y**0, X**2 * Y**0, X**3 * Y**0,
	//              X**0 * Y**1, X**1 * Y**1, X**2 * Y**1,      0,
	//              X**0 * Y**2, X**1 * Y**2,      0,           0,
	//              X**0 * Y**3       0,           0,           0

	Zpoly[0] = pow(boardX,0) * pow(boardY,0);
	Zpoly[1] = pow(boardX,1) * pow(boardY,0);
	Zpoly[2] = pow(boardX,2) * pow(boardY,0);
	Zpoly[3] = pow(boardX,3) * pow(boardY,0);
	Zpoly[4] = pow(boardX,0) * pow(boardY,1);
	Zpoly[5] = pow(boardX,1) * pow(boardY,1);
	Zpoly[6] = pow(boardX,2) * pow(boardY,1);
	Zpoly[7] = pow(boardX,0) * pow(boardY,2);
	Zpoly[8] = pow(boardX,1) * pow(boardY,2);
	Zpoly[9] = pow(boardX,0) * pow(boardY,3);

	// X direction equations
	pdRow[iFOVPosA] += w;
	pdRow[iFOVPosA+2] += -ySensorA * w;
	for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[ColumnZTerm(j, deviceNumA)] = Zpoly[j] * dxSensordzA * w;
	//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialXPos() - xSensorA);
	_dVectorB[_iCurrentRow] = w * (dFidRoiCenX - xSensorA);
	sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FidCorr:L%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
		pOverlap->GetMosaicLayer()->Index(),
		pOverlap->GetTriggerIndex(), pOverlap->GetCameraIndex(),
		colImgA, rowImgA, xSensorA, ySensorA, pOverlap->GetFiducialXPos(),pOverlap->GetFiducialYPos() );
	
	_pdWeights[_iCurrentRow] = w;
	_iCurrentRow++;
	
	// Y direction equations
	pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
	pdRow[iFOVPosA+1] += w;
	pdRow[iFOVPosA+2] += +xSensorA * w;
	for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[ColumnZTerm(j, deviceNumA)] = Zpoly[j] * dySensordzA * w;
	//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialYPos() - ySensorA);
	_dVectorB[_iCurrentRow] = w * (dFidRoiCenY - ySensorA);
	sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FidCorr:L%d:T%d:C%d,%.2e,%.2e,%.2e,%.2e,%.4e,%.4e", 
		pOverlap->GetMosaicLayer()->Index(),
		pOverlap->GetTriggerIndex(), pOverlap->GetCameraIndex(),
		offsetRows, offsetCols,
		pPair->GetCorrelationResult().CorrCoeff, pPair->GetCorrelationResult().AmbigScore,
		boardX,boardY);

	_pdWeights[_iCurrentRow] = w;
	_iCurrentRow++;

	delete [] Zpoly;
	return( true );
}

/// <summary>
/// return image to CAD space transform (projective 3x3 array) for selected FOV
///</summary>
/// Also calculates the distortion model (camera calibration model) mapping FOV to/from CAD
ImgTransform RobustSolverCM::GetResultTransform(
	unsigned int iLayerIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex)  
{
	// Fov transform parameter begin position in column
	FovIndex fovIndex(iLayerIndex, iTriggerIndex, iCameraIndex); 
	unsigned int index = (*_pFovOrderMap)[fovIndex] *_iNumParamsPerIndex;
	unsigned int iImageCols = _pSet->GetLayer(iLayerIndex)->GetImage(iTriggerIndex, iCameraIndex)->Columns();
	unsigned int iImageRows = _pSet->GetLayer(iLayerIndex)->GetImage(iTriggerIndex, iCameraIndex)->Rows();
		
	double t[3][3];
	MatchProjeciveTransform(iLayerIndex, iTriggerIndex, iCameraIndex, t);
	
	ImgTransform trans(t);

	// add camModel transform calculation and load....
	// take set of points in image, set of points in CAD
	// calc matching transform (pixel to flattenend board surface)
	TransformCamModel tCamModel = _pSet->GetLayer(iLayerIndex)->GetImage(iTriggerIndex, iCameraIndex)->GetTransformCamModel();
	tCamModel.Reset();
	int iNumX = 5;	// was 11 x 11, reduced size saves time in generating transforms
	int iNumY = 5;
	double startX = 0;
	double stopX = (double)(iImageRows - 1); // x direction is in + row direction
	double startY = 0;
	double stopY = (double)(iImageCols- 1); // y direction is in + col direction
	// use non-linear spacing to get better performance at edges
	double angleStepX = PI / (iNumX - 1.0);
	double angleStepY = PI / (iNumY - 1.0);
	
	POINTPIX* uv = new POINTPIX[iNumX*iNumY];
	POINT2D_C* xy =  new POINT2D_C[iNumX*iNumY];
	int i, j, k;
	double dScale;
	//LOG.FireLogEntry(LogTypeSystem, "ResultFOVCamModel m %d, %d, %d",
	//	iLayerIndex, iTriggerIndex, iCameraIndex);
	for (j=0, i=0; j<iNumY; j++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
	  for (k=0; k<iNumX; k++) {          /* uv.u value is image COL, parralel to CAD Y*/
			dScale = (cos(-PI + angleStepX*k) + 1.0 )/2.0; // scaled 0 to 1
			uv[i].v = (stopX - startX)* dScale + startX;
	  		dScale = (cos(-PI + angleStepY*j) + 1.0 )/2.0; // scaled 0 to 1
			uv[i].u = (stopY - startY)*dScale + startY;
		 POINT2D_C xyBoard;
		 Pix2Board(uv[i], fovIndex, &xyBoard); 
		// Now have position on flattened board,
		// need to transform to CAD postion using board2CAD
		// x position
		xy[i].x = _Board2CAD.GetItem(0)*xyBoard.x + _Board2CAD.GetItem(1)*xyBoard.y + _Board2CAD.GetItem(2);
		// y position
		xy[i].y = _Board2CAD.GetItem(3)*xyBoard.x + _Board2CAD.GetItem(4)*xyBoard.y + _Board2CAD.GetItem(5);
		i++;
	  }
	}
	tCamModel.CalcTransform(uv, xy, iNumX*iNumY);
	delete [] uv;
	delete [] xy;
	return(trans);
}

/// <summary>
/// Calculate the FOV to CAD space transform (projective 3x3 array) for selected FOV
///</summary>
/// 
bool RobustSolverCM::MatchProjeciveTransform(	
	unsigned int iLayerIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex, double dTrans[3][3]) 
{
	FovIndex fovIndex(iLayerIndex, iTriggerIndex, iCameraIndex); 
	unsigned int iImageCols = _pSet->GetLayer(iLayerIndex)->GetImage(iTriggerIndex, iCameraIndex)->Columns();
	unsigned int iImageRows = _pSet->GetLayer(iLayerIndex)->GetImage(iTriggerIndex, iCameraIndex)->Rows();
	unsigned int deviceNum = _pSet->GetLayer(fovIndex.LayerIndex)->DeviceIndex();
	unsigned int iNumX = 3;  // set to 3 uses corner and centers of edges (along with image center)
	unsigned int iNumY = 3;  // (the resulting fit is more accurate on the edges)
	double startX = 0;
	double stopX = (double)(iImageRows - 1); // x direction is in + row direction
	double startY = 0;
	double stopY = (double)(iImageCols- 1); // y direction is in + col direction
	// use non-linear spacing to get better performance at edges
	double angleStepX = PI / (iNumX - 1.0);
	double angleStepY = PI / (iNumY - 1.0);
	
	unsigned int m = iNumX*iNumY;
	complexd* p = new complexd[m];  // lsqrproj() uses complexd data type
	complexd* q = new complexd[m];
	complexd* resid = new complexd[m];
	POINT2D xyBoard;
	POINT2D_C xyBoard_C;
	POINTPIX pix;  // data type needed by Pix2Board()
	// Create match pairs for projective transform
	// p will contain p.r = pixel row locations, p.i = pixel col locations
	// q.r will contain matching X (cad direction)
	// q.i will contain matching Y
	double XSampleFraction(0.80);  // we want to avoid the actual corners as they aren't used
	double YSampleFraction(0.85);  // this sets the fraction of the image size where the fit occurs
	int i(0);
	double dScale;
	for(unsigned int j=0; j<iNumY; j++)
	{
		for(unsigned int k=0; k<iNumX; k++)
		{
			dScale = (XSampleFraction * cos(-PI + angleStepX*j) + 1.0 )/2.0; // scaled 0 to 1
			p[i].r = (stopX - startX)*dScale + startX;
			dScale = (YSampleFraction * cos(-PI + angleStepY*k) + 1.0 )/2.0; // scaled 0 to 1
			p[i].i = (stopY - startY)*dScale + startY;
			// now find the matching point in CAD space
			// order is:
			// pixel -> xySensor -> xyBoard -> xyCAD
			// pixel to xyBoard
			pix.u = p[i].i;  // uv vs ri got ugly here
			pix.v = p[i].r;
			Pix2Board(pix, fovIndex, &xyBoard_C);
			xyBoard.x = xyBoard_C.x;
			xyBoard.y = xyBoard_C.y;
			//xyBoard to CAD
			POINT2D temp = warpxy(NUMBER_Z_BASIS_FUNCTIONS-1, 2*NUMBER_Z_BASIS_FUNCTIONS, NUMBER_Z_BASIS_FUNCTIONS,
								(double*)_zCoef[deviceNum], xyBoard, LAB_TO_CAD);
			// Now have position on flattened board,
			// need to transform to CAD postion using board2CAD
			// x position
			q[i].r = _Board2CAD.GetItem(0)*temp.x + _Board2CAD.GetItem(1)*temp.y + _Board2CAD.GetItem(2);
			// y position
			q[i].i = _Board2CAD.GetItem(3)*temp.x + _Board2CAD.GetItem(4)*temp.y + _Board2CAD.GetItem(5);
			i++;
		}
	}
	double RMS, dRcond;
	int iFlag = lsqrproj(m, p, q, 1, dTrans, &RMS, resid, &dRcond);
	/*if (_bVerboseLogging)
	{
		LOG.FireLogEntry(LogTypeSystem, "MatchProj index %d, %d, %d",iLayerIndex, iTriggerIndex, iCameraIndex); 
		LOG.FireLogEntry(LogTypeSystem, "MatchProj p %.4e, %.4e,    %.4e, %.4e,    %.4e, %.4e",
			p[0].i, p[0].r, p[1].i, p[1].r, p[2].i, p[2].r);
		LOG.FireLogEntry(LogTypeSystem, "MatchProj q %.4e, %.4e,    %.4e, %.4e,    %.4e, %.4e",
			q[0].i, q[0].r, q[1].i, q[1].r, q[2].i, q[2].r);
		LOG.FireLogEntry(LogTypeSystem, "dTrans \n %.4e, %.4e,%.4e, \n%.4e, %.4e,%.4e, \n%.4e, %.4e,%.4e",
			dTrans[0][0], dTrans[0][1], dTrans[0][2],
			dTrans[1][0], dTrans[1][1], dTrans[1][2],
			dTrans[2][0], dTrans[2][1], dTrans[2][2]);
		// NOTE: resid = q - fitValue = distorted pt - proj fit pt
	}*/
	if(iFlag != 0)
	{
		LOG.FireLogEntry(LogTypeError, "Failed to create matching projective transform");
	}
	
	delete [] p;
	delete [] q;
	delete [] resid;
	if(iFlag != 0)
		return(false);
	else
		return(true);
}

// Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolverCM::SolveXAlgH()
{

	::memcpy(_dMatrixACopy, _dMatrixA, _iMatrixSize*sizeof(double));
	::memcpy(_dVectorBCopy, _dVectorB, _iMatrixHeight*sizeof(double)); 

	// we built A in row order
	// the qr factorization method requires column order
	bool bRemoveEmptyRows = true;
	//bRemoveEmptyRows = false;
	//int* mb = new int[_iMatrixWidth];
	//unsigned int iEmptyRows;
	TransposeMatrixA(bRemoveEmptyRows);

	double*	resid = new double[_iMatrixHeight];
	double scaleparm = 0;
	double cond = 0;

	LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():BEGIN ALG_H");

	int algHRetVal = 
		alg_h(                // Robust regression by Huber's "Algorithm H".
			// Inputs //			

			//_iMatrixHeight,             /* Number of equations */
			_iMatrixALastRowUsed,             /* Number of equations */
			_iMatrixWidth,				/* Number of unknowns */
			_dMatrixA,				// System matrix
			//_iMatrixHeight,			//AStride
			_iMatrixALastRowUsed,
			_dVectorB,			// Constant vector (not overwritten); must not coincide with x[] or resid[]. 

		   // Outputs //

			_dVectorX,            // Coefficient vector (overwritten with solution) 
			resid,				// Residuals b - A*x (pass null if not wanted).  If non-null, must not coincide with x[] or b[].
			&scaleparm,				// Scale parameter.  This is a robust measure of
									//  dispersion that corresponds to RMS error.  (Pass NULL if not wanted.)
			&cond					// Approximate reciprocal condition number.  (Pass NULL if not wanted.) 

			
		   /* Return values

			>=0 - normal return; value is # of iterations.
			 -1 - Incompatible dimensions (m mustn't be less than n)
			 -2 - singular system matrix
			 -3 - malloc failed.
			 -4 - Iteration limit reached (results may be tolerable)*/	
				  );

	// Save resid
	if (_bSaveMatrixCSV) 
	{
		ofstream of("C:\\Temp\\Resid.csv");
		for(unsigned int k=0; k<_iMatrixHeight; k++)
		{ 
			of << resid[k] << std::endl;
		}
		of.close();
	}
	if(CorrelationParametersInst.bSaveTransformVectors)
	{
		_mkdir(CorrelationParametersInst.sDiagnosticPath.c_str());
		char cTemp[255];
		string s;
		sprintf_s(cTemp, 100, "%sTransformVectorX_%d.csv", CorrelationParametersInst.sDiagnosticPath.c_str(), iFileSaveIndex); 
		s.clear();
		s.assign(cTemp);
		OutputVectorXCSV(s);
	}
	
	if( algHRetVal<0 )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::SolveXAlgH():alg_h returned value of %d", algHRetVal);

	//LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():FINISHED ALG_H");
	LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():alg_h nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	delete [] resid;
	//delete [] mb;

	// copy board Z shape terms to the _zCoef array
	// First we'll put take VectorX values for Z and use it to populate Zpoly (a 4x4 array)
	// FORTRAN ORDER !!!!!
	// TODO TODO terms 0 and 4 based on SIM number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	for (unsigned int deviceNum(0); deviceNum < _iNumDevices; deviceNum++ )
	{
		_zCoef[deviceNum][0][0] = _dVectorX[ColumnZTerm(0, deviceNum)];
		_zCoef[deviceNum][1][0] = _dVectorX[ColumnZTerm(1, deviceNum)];
		_zCoef[deviceNum][2][0] = _dVectorX[ColumnZTerm(2, deviceNum)];
		_zCoef[deviceNum][3][0] = _dVectorX[ColumnZTerm(3, deviceNum)];
	
		_zCoef[deviceNum][0][1] = _dVectorX[ColumnZTerm(4, deviceNum)];
		_zCoef[deviceNum][1][1] = _dVectorX[ColumnZTerm(5, deviceNum)];
		_zCoef[deviceNum][2][1] = _dVectorX[ColumnZTerm(6, deviceNum)];
		_zCoef[deviceNum][3][1] = 0;		
											
		_zCoef[deviceNum][0][2] = _dVectorX[ColumnZTerm(7, deviceNum)];
		_zCoef[deviceNum][1][2] = _dVectorX[ColumnZTerm(8, deviceNum)];
		_zCoef[deviceNum][2][2] = 0;		
		_zCoef[deviceNum][3][2] = 0;		
											
		_zCoef[deviceNum][0][3] = _dVectorX[ColumnZTerm(9, deviceNum)];
		_zCoef[deviceNum][1][3] = 0;
		_zCoef[deviceNum][2][3] = 0;
		_zCoef[deviceNum][3][3] = 0;
	}
	iFileSaveIndex++;
}
/// <summary>
/// Calculates the xyBoard position (in meters) given a pixel location and FOV identifier
///</summary>
/// <param name="pix"></param>
/// <param name="fovindex"></param>
/// <param name="xyBoard">Board pos (meters)</param>

void  RobustSolverCM::Pix2Board(POINTPIX pix, FovIndex fovindex, POINT2D_C *xyBoard)
{
	// p2Board translates between a pixel location (for a given Layer, trigger and camera defined by index) to xyBoard location
	// remember that xyBoard is on a warped surface, so the xy distance between marks is likely to be less than the
	// CAD distance
	// Also remember that this board model has not been aligned to CAD
	unsigned int iLayerIndex( fovindex.LayerIndex );
	unsigned int iTrigIndex( fovindex.TriggerIndex );
	unsigned int iCamIndex( fovindex.CameraIndex );
	unsigned int iVectorXIndex = (*_pFovOrderMap)[fovindex] *_iNumParamsPerIndex;
	unsigned int deviceNum = _pSet->GetLayer(fovindex.LayerIndex)->DeviceIndex();
	TransformCamModel camCal = 
				 _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
	complexd xySensor;
	camCal.SPix2XY(pix.u, pix.v, &xySensor.r, &xySensor.i);

	// use board transform to go to board x,y
	//xyBoard->r = cos(VectorX[index+2]) * xySensor.r  - sin(VectorX[index+2]) * xySensor.i  + VectorX[index+0];
	//xyBoard->i = sin(VectorX[index+2]) * xySensor.r  + cos(VectorX[index+2]) * xySensor.i  + VectorX[index+1];
	// try WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
	xyBoard->x =        1         * xySensor.r  - _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
	xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +        1         * xySensor.i  + _dVectorX[iVectorXIndex+1];

	// can now estimate Z   zCoef HAS FORTRAN ORDERING
	double Zestimate(0);
	for (unsigned int k(0); k < NUMBER_Z_BASIS_FUNCTIONS; k++)
		for (unsigned int j(0); j < NUMBER_Z_BASIS_FUNCTIONS; j++)
			Zestimate += _zCoef[deviceNum][j][k] * pow(xyBoard->x,(double)j) * pow(xyBoard->y,(double)k);

	// accurate Z value allows accurate calc of xySensor
	double dx, dy;
	camCal.dSPix2XY(pix.u, pix.v, &dx, &dy);

	xySensor.r +=  Zestimate * dx;
	xySensor.i += Zestimate * dy;

	// accurate xySensor allows accurate xyBrd calc
	//xyBoard->r = cos(VectorX[index+2]) * xySensor.r  - sin(VectorX[index+2]) * xySensor.i  + VectorX[index+0];
	//xyBoard->i = sin(VectorX[index+2]) * xySensor.r  + cos(VectorX[index+2]) * xySensor.i  + VectorX[index+1];
	// try WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
	xyBoard->x =               1 * xySensor.r             -  _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
	xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +                1 * xySensor.i             + _dVectorX[iVectorXIndex+1];

}
/// <summary>
/// Determines affine transform to map surface (of a warped panel) to CAD space
///</summary>
/// The fiducials found on the warped panel are first flattened to a plane,
/// these planer locations are then mapped to CAD locations by an affine transform
void RobustSolverCM::FlattenFiducials(PanelFiducialResultsSet* fiducialSet)
{
	// translate the fiducial locations on the board down to a nominal height plane
	// uses warpxy.c to do this translation
	// then do a least sq. fit to do an affine xform between board and CAD 

	if(fiducialSet->Size() == 0)
	{
		LOG.FireLogEntry(LogTypeSystem, "Flatten Fiducial: No fiducial information, skip!"); 
		return;
	}

	// count the number fiducial overlaps in the data set 
	unsigned int nFidOverlaps(0);
	int iNumPhyFid = fiducialSet->Size();
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = fiducialSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
			nFidOverlaps++;
	}
	POINT2D* fidFlat2D = new POINT2D[nFidOverlaps]; // fiducial location on flattened board
	POINT2D* fidCAD2D = new	POINT2D[nFidOverlaps];  // true CAD location of fiducial
	double resetTransformValues[3][3] =  {{1.0,0,0},{0,1.0,0},{0,0,1}};
	_Board2CAD.SetMatrix( resetTransformValues );
	unsigned int nGoodFids(0);  // <= total number of fids
	// Now add the fiducials to the two arrays created above
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = fiducialSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
		{
			bool fidOK;
			CorrelationPair* pPair = (*j)->GetCoarsePair();
			double w = Weights.CalWeight(pPair) * Weights.RelativeFidFovCamModWeight;
			fidOK = (*j)->IsGoodForSolver() && (w > 0);
			if (fidOK)
			{
				// CAD locations
				double rowImgB = (*j)->GetCoarsePair()->GetSecondRoi().RowCenter();
				double colImgB = (*j)->GetCoarsePair()->GetSecondRoi().ColumnCenter();
				double dFidRoiCenX, dFidRoiCenY; // ROI center of fiducial image (not fiducail center or image center) in world space
				(*j)->GetFidImage()->ImageToWorld(rowImgB, colImgB, &dFidRoiCenX, &dFidRoiCenY);
				fidCAD2D [nGoodFids].x = dFidRoiCenX;
				fidCAD2D [nGoodFids].y = dFidRoiCenY;
				
				// Panel locations
				double rowImg = (*j)->GetCoarsePair()->GetFirstRoi().RowCenter();
				double colImg = (*j)->GetCoarsePair()->GetFirstRoi().ColumnCenter();
				double regoffCol = (*j)->GetCoarsePair()->GetCorrelationResult().ColOffset;
				double regoffRow = (*j)->GetCoarsePair()->GetCorrelationResult().RowOffset;
				double rowPeak( rowImg - regoffRow );
				double colPeak( colImg - regoffCol ); 
				POINT2D xyBoard;
				POINT2D_C xyBoard_C;
				POINTPIX pix;
				pix.u = colPeak;  // Pix2Board uses u -> i, v -> r
				pix.v = rowPeak;
				FovIndex fovIndex( (*j)->GetMosaicLayer()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex() );
				unsigned int deviceNum = _pSet->GetLayer(fovIndex.LayerIndex)->DeviceIndex();
				Pix2Board(pix, fovIndex, &xyBoard_C);
				xyBoard.x = xyBoard_C.x;
				xyBoard.y = xyBoard_C.y;

				// brdFlat from warpxy
				POINT2D temp;
				temp = warpxy(NUMBER_Z_BASIS_FUNCTIONS-1, 2*NUMBER_Z_BASIS_FUNCTIONS, NUMBER_Z_BASIS_FUNCTIONS,
									(double*)_zCoef[deviceNum], xyBoard, LAB_TO_CAD);
				fidFlat2D[nGoodFids].x = temp.x;
				fidFlat2D[nGoodFids].y = temp.y;

				nGoodFids++;
			}
		}
	}
	// Rows of FidFitA contain the following values:
	//  0		Constrain scale X ~ 1.0,  m0 ~ 1
	//  1		Constrain scale Y ~ 1.0,  m4 ~ 1
	//  2		Constrain scale x = scale y,   m0-m4 ~ 0
	//  3		Constrain m1 + m3  ~ 0  keep rectangular
	//  4		Constrain m1 ~ 0  small angle
	//  5		Constrain m3 ~ 0  small angle
	//  6		Constrain m2 ~ 0  small lateral shift
	//  7       Constrain m5 ~ 0  small lateral shift
	// 
	// Total 8 constraints
	int nContraints = 8;
		
	// Create A and b matrices, these are small so we can create/delete them without using big blocks of memory
	double		*FidFitA;
	double		*FidFitX;
	double		*FidFitb;
	double		*resid;
	unsigned int FidFitCols = 6;
	unsigned int FidFitRows = nGoodFids*2 + 8;
	unsigned int SizeofFidFitA = FidFitCols * FidFitRows;
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
	FidFitb[0] = 1.0 * Weights.wFidFlatBoardScale;
	FidFitA[0*FidFitCols + 0] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[1] = 1.0 * Weights.wFidFlatBoardScale;
	FidFitA[1*FidFitCols + 4] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[2] = 0.0 * Weights.wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 0] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 4] = -1.0 * Weights.wFidFlatBoardScale;
	FidFitb[3] = 0.0 * Weights.wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 1] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 3] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[4] = 0.0 * Weights.wFidFlatBoardRotation;
	FidFitA[4*FidFitCols + 1] =  1.0 * Weights.wFidFlatBoardRotation;
	FidFitb[5] = 0.0 * Weights.wFidFlatBoardRotation;
	FidFitA[5*FidFitCols + 3] =  1.0 * Weights.wFidFlatBoardRotation;
	FidFitb[6] = 0.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitA[6*FidFitCols + 2] =  1.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitb[7] = 0.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitA[7*FidFitCols + 5] =  1.0 * Weights.wFidFlatFiducialLateralShift;
	// add the actual fiducial locations
	int AStart(0);
	for (unsigned int j(0); j< nGoodFids; j++)
	{
		FidFitb[8+j*2+0] = fidCAD2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitb[8+j*2+1] = fidCAD2D[j].y * Weights.wFidFlatFlattenFiducial;
		AStart = (8+j*2+0)*FidFitCols;
		FidFitA[AStart + 0] = fidFlat2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 1] = fidFlat2D[j].y * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 2] = Weights.wFidFlatFlattenFiducial;
		AStart = (8+j*2+1)*FidFitCols;
		FidFitA[AStart + 3] = fidFlat2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 4] = fidFlat2D[j].y * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 5] = Weights.wFidFlatFlattenFiducial;
	}
	// Note: above equations do not weight the fiducial finds by corrcoef or ambig
	// ('bad' fids are already removed from the list)
	// As the number of fids is small each one one will have a great deal of leverage
	// therefore it's hard to de-weight poorer fits
	if (_bVerboseLogging)
	{
		for (unsigned int row(0); row < FidFitRows; row++)
			LOG.FireLogEntry(LogTypeSystem, "%d, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e,       \t %.6e,        ",row, 
				FidFitA[row*6], FidFitA[row*6+1], FidFitA[row*6+2], FidFitA[row*6+3], 
				FidFitA[row*6+4], FidFitA[row*6+5], FidFitb[row] );
	}
	// equations are done, now solve

	LstSqFit(FidFitA, FidFitRows, FidFitCols, FidFitb, FidFitX, resid);
	if (_bVerboseLogging)
		for(unsigned int row(0); row<FidFitRows; ++row)
			LOG.FireLogEntry(LogTypeSystem, "Resids %d, %.4f",row, resid[row]);
	double newBoard2CAD[3][3] = {{ FidFitX[0], FidFitX[1], FidFitX[2]}, {FidFitX[3], FidFitX[4], FidFitX[5]}, {0,0,1}};
	_Board2CAD.SetMatrix(newBoard2CAD);
	
	delete [] FidFitA;
	delete [] FidFitb;
	delete [] FidFitX;
	delete [] resid;
		
	// log results for debug
	//if (_bVerboseLogging)
		LOG.FireLogEntry(LogTypeSystem, "Flatten Fiducial _Board2CAD %.5e,%.5e,%.5e,   %.5e,%.5e,%.5e    ", 
			_Board2CAD.GetItem(0), _Board2CAD.GetItem(1), _Board2CAD.GetItem(2), _Board2CAD.GetItem(3), _Board2CAD.GetItem(4), _Board2CAD.GetItem(5));
		
	delete [] fidFlat2D;
	delete [] fidCAD2D;

	// Do some quality testing of the results
	if(nGoodFids  < 2 )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::FlattenFiducials(): Too few fiducials found, only found  = %d", nGoodFids);
	
	double skew;
	double xStretch;
	double yStretch;
	// test that the scaling of the matrix if very close to 1.000
	xStretch = sqrt(pow(_Board2CAD.GetItem(0), 2.0) + pow(_Board2CAD.GetItem(1), 2.0)) - 1;
	yStretch = sqrt(pow(_Board2CAD.GetItem(3), 2.0) + pow(_Board2CAD.GetItem(4), 2.0)) - 1;
	// the 1 and 3 terms should be equal and opposite, if not then the affine xform is skewing the board
	skew = _Board2CAD.GetItem(1) + _Board2CAD.GetItem(3);
	//fiducialSet->SetnGoodFids(nGoodFids);
	fiducialSet->SetPanelSkew(skew);
	fiducialSet->SetXscale(xStretch);
	fiducialSet->SetYscale(yStretch);
	// as all units are meter / meter we can compare the results to IPC limits
	// John Hoffman found  IPC-D-300G which allows a stretch of 200 um over 300 mm or 0.067%
	if( abs(xStretch) > CorrelationParametersInst.dAlignBoardStretchLimit )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::FlattenFiducials(): X stretch excessive = %f", xStretch);
	if( abs(yStretch) > CorrelationParametersInst.dAlignBoardStretchLimit )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::FlattenFiducials(): Y stretch excessive = %f", yStretch);
	if( abs(skew) > CorrelationParametersInst.dAlignBoardSkewLimit )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::FlattenFiducials(): skew excessive = %f", skew);

	// For debug
	if(CorrelationParametersInst.bSaveTransformVectors)
	{
		_mkdir(CorrelationParametersInst.sDiagnosticPath.c_str());
		char cTemp[255];
		string s;
		sprintf_s(cTemp, 100, "%sBoardToCAD_%d.csv", CorrelationParametersInst.sDiagnosticPath.c_str(), iFileSaveIndex); 
		s.clear();
		s.assign(cTemp);
		OutputBoardToCAD(s);
	}
}

//  transpose Matrix A for solver
// Don't reorder for this solver, only transpose to _iCurrentRow if bRemoveEmptyRows
///<summary>
///Transpose the A matrix, remove empty rows if requrested
///</summary>
/// <param name="bRemoveEmptyRows"></param>
///Name doesn't match function as this routine no longer reorders matrix (not needed in Camera Model solver)
/// Transpose is still needed for call to solver

void RobustSolverCM::TransposeMatrixA(bool bRemoveEmptyRows)
{
	string fileName;
	char cTemp[100];
	if (bRemoveEmptyRows)
		_iMatrixALastRowUsed = _iCurrentRow;
	else
		_iMatrixALastRowUsed = _iMatrixHeight;
	if (_bSaveMatrixCSV) 
	{
		// Save Matrix A
		//ofstream of("C:\\Temp\\MatrixA.csv");
		sprintf_s(cTemp, 100, "C:\\Temp\\MatrixA_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		ofstream of(fileName.c_str());
	
		for(unsigned int k=0; k<_iMatrixALastRowUsed; k++)
		{ 
			for(unsigned int j=0; j<_iMatrixWidth; j++)
			{
				of << _dMatrixA[j+k*_iMatrixWidth];
				if(j != _iMatrixWidth-1)
					of <<",";
			}
			of << std::endl;
		}
		of.close();

		// Save Matrix B
		sprintf_s(cTemp, 100, "C:\\Temp\\VectorB_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		of.open(fileName.c_str());
		for(unsigned int k=0; k<_iMatrixALastRowUsed; k++)
		{ 
			of << _dVectorB[k] << std::endl;
		}
		of.close();

		// Save weights
		if (_pdWeights !=NULL)
		{
			sprintf_s(cTemp, 100, "C:\\Temp\\Weights_%d.csv",iFileSaveIndex); 
			fileName.clear();
			fileName.assign(cTemp);
			of.open(fileName.c_str());
			for(unsigned int k=0; k<_iMatrixALastRowUsed; k++)
			{ 
				of << _pdWeights[k] << std::endl;
			}
			of.close();
		}

		// Save Notes
		if (_pcNotes!=NULL)
		{
			sprintf_s(cTemp, 100, "C:\\Temp\\Notes_%d.csv",iFileSaveIndex); 
			fileName.clear();
			fileName.assign(cTemp);
			of.open(fileName.c_str());
			for(unsigned int k=0; k<_iMatrixALastRowUsed; k++)
			{ 
				of << _pcNotes[k] << std::endl;
			}
			of.close();
		}  
	}
	
	
	// Transpose
	double* workspace = new double[_iMatrixSize];
	//double* dCopyB = new double[_iMatrixHeight];
	for (unsigned int i(0); i<_iMatrixSize; i++)
		workspace[i]=0;
	//for (unsigned int i(0); i<_iMatrixHeight; i++)
	//	dCopyB[i]=0;
	unsigned int iDestRow = 0;
	//list<LeftIndex>::const_iterator i;
	for (unsigned int row(0); row<_iMatrixALastRowUsed; row++)
	{
		for(unsigned int col(0); col<_iMatrixWidth; ++col)
			workspace[col*_iMatrixALastRowUsed+row] = _dMatrixA[row*_iMatrixWidth+col];

	}
	
	// include empty rows
	::memcpy(_dMatrixA, workspace, _iMatrixSize*sizeof(double));

	// for debug
	/*if (_bSaveMatrixCSV) // comment out this section, very large file, slow to write
	{
		// Save transposed Matrix A 
		sprintf_s(cTemp, 100, "C:\\Temp\\MatrixA_t_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		ofstream of(fileName.c_str());		
		of << std::scientific;

		unsigned int ilines = _iMatrixALastRowUsed;
		
		for(unsigned int j=0; j<_iMatrixWidth; j++)
		{
			for(unsigned int k=0; k<ilines; k++)
			{ 
	
				of << _dMatrixA[j*ilines+k];
				if(k != ilines-1)
					of <<",";
			}
			of << std::endl;
		}
		of.close();
	} */
	delete [] workspace;
}


bool RobustSolverCM::GetPanelHeight(unsigned int iDeviceIndex, double pZCoef[16])
{
	if(iDeviceIndex >= _iNumDevices)
		return(false);

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			pZCoef[i*4+j] = _zCoef[iDeviceIndex][i][j];

	return(true);
}

#pragma region Debug
// Debug
// Vector X output
void RobustSolverCM::OutputVectorXCSV(string filename) const
{
	ofstream of(filename.c_str());

	of << std::scientific;

	string line;
	unsigned int j;
	
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		unsigned int iCamIndex( k->first.CameraIndex);
		unsigned int iIndexID( k->second );
		if (_pSet->GetLayer(k->first.LayerIndex)->IsFirstCamOfSubTrigger(k->first.CameraIndex)) 
		//if (iCamIndex == 0)  // first camera (logical camera) is always numbered 0
		{
			of << "I_" << k->first.LayerIndex 
				<< "T_" << k->first.TriggerIndex 
				<< "C_" << k->first.CameraIndex
				<< ",";
			of << iIndexID;  // record index ID
			of << ",";
			for(j=0; j<_iNumParamsPerIndex; ++j)
			{
				if( j!=0 )
					of << ",";
				double d = _dVectorX[iIndexID * _iNumParamsPerIndex +j];
				of << d;
			}
			of <<  std::endl;
		}
	}
		// TODO TODO terms 0 and 4 based on SIM number !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	for(unsigned int j=0; j<_iNumZTerms + (_iNumDevices-1)*2; ++j) 
	{
		of << ",";

		double d = _dVectorX[_iStartColZTerms + j];
		of << d;
	}
	of <<  std::endl;
	of.close();
}

void RobustSolverCM::OutputBoardToCAD(string filename) const
{
	ofstream of(filename.c_str());

	of << std::scientific;
	
	for(unsigned int j=0; j<9; j++) 
	{
		of << _Board2CAD.GetItem(j) << ",";
	}
	of <<  std::endl;
	of.close();
}
#pragma endregion

