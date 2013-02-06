#include "RobustSolverIterative.h"
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
/*
Matrix A layout has changed

Columns 0 through  _iTotalNumberOfSubTriggers * _iNumParamsPerIndex -1
x_trig, y_trig, dtheta_trig
 note that theta_trig is now theta_est + dtheta_trig

Columns to _iCalDriftStartCol + iNumCalDriftTerms - 1
calibration drift terms
dev0_cam0_ds_x, dev0_cam0_ds_y, dev0_cam1_ds_x, dev0_cam1_ds_y, ....
  dev1_cam0_ds_x, ....to last dev, last camera

Last 10 terms
Z board warp map
*/

//
// Iterative Camera Model Solver
RobustSolverIterative::RobustSolverIterative(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet): 	RobustSolverCM( pFovOrderMap,  iMaxNumCorrelations, pSet)
{
	_iMaxIterations = CorrelationParametersInst.iSolverMaxIterations;
	_iCalDriftStartCol = _iTotalNumberOfSubTriggers * _iNumParamsPerIndex;
	_iStartColZTerms = _iCalDriftStartCol + _iNumCalDriftTerms;
	_iMatrixWidth = _iStartColZTerms + _iNumZTerms + (_iNumDevices-1)*2;
	
	_dThetaEst = new double[_iTotalNumberOfSubTriggers];
	_fitInfo = new fitInfo[_iMaxNumCorrelations];
	_bPinPanelWithCalibration = false;

	ZeroTheSystem();
}

RobustSolverIterative::~RobustSolverIterative()
{
	delete [] _dThetaEst;
	delete [] _fitInfo;
}

void RobustSolverIterative::ZeroTheSystem()
{

	RobustSolverCM::ZeroTheSystem();
	
	for(unsigned int i =0; i<_iMaxNumCorrelations; i++)
	{
		// reset the fit info struct
		_fitInfo[i].fitType=0;	
		_fitInfo[i].fovIndexA.CameraIndex=0;
		_fitInfo[i].fovIndexA.LayerIndex=0;
		_fitInfo[i].fovIndexA.TriggerIndex=0;
		_fitInfo[i].fovIndexB.CameraIndex=0;
		_fitInfo[i].fovIndexB.LayerIndex=0;
		_fitInfo[i].fovIndexB.TriggerIndex=0;
		_fitInfo[i].boardX=0;		
		_fitInfo[i].boardY=0;
		_fitInfo[i].w=0;			
		_fitInfo[i].rowInMatrix=0;
		_fitInfo[i].colInMatrixA=0;
		_fitInfo[i].colInMatrixB=0;
		_fitInfo[i].xSensorA=0;
		_fitInfo[i].ySensorA=0;
		_fitInfo[i].xSensorB=0;
		_fitInfo[i].ySensorB=0;
		_fitInfo[i].dxSensordzA=0;
		_fitInfo[i].dySensordzA=0;
		_fitInfo[i].dxSensordzB=0;
		_fitInfo[i].dySensordzB=0;
		_fitInfo[i].dFidRoiCenX=0;
		_fitInfo[i].dFidRoiCenY=0;
	}

	_iCorrelationNum = 0;
}


bool RobustSolverIterative::AddAllLooseConstraints(
		bool bPinPanelWithCalibration, 
		bool bUseNominalTransform)
{
	ConstrainZTerms();
	ConstrainPerTrig(bPinPanelWithCalibration);

	return(true);
}

void RobustSolverIterative::SolveXAlgH()
{
	// algH iterative wrapper
	// allowing the calibration of the sensor to vary creates a non-linear fit, hence iteration
	// is needed.
	//Steps:
	// 0. set est. values of cal error to 0
	// 1. zero out A, b, etc.
	// 2. fill matrix A , b
	// 3. run one iteration of solver
	// 4. extract est. of cal error (update estimates)
	// 5. decide if done, if not go back to step 1.
	// 

	// Column order for MatrixA and VectorX
	// first _iTotalNumberOfSubTriggers * _iNumParamsPerIndex columns
	// contain Xtrig, Ytrig, dTheta
	// next _iNumCalDriftTerms = _iNumDevices * _iNumCameras * 2
	// contain ds_x, ds_y [device][camera] 
	// last _iNumZTerms = 10;	are Z terms
	unsigned int i;
	_iIterationNumber = 0;
	for(i =0; i<_iTotalNumberOfSubTriggers; i++)
		_dThetaEst[i]=0;
	//for(i =0; i<_iNumCalDriftTerms; i++)
	//	_dCalDriftEst[i]=0;
	int iMaxIter = _iMaxIterations; // local copy
	if (CorrelationParametersInst.bUseTwoPassStitch && !CorrelationParametersInst.bCoarsePassDone)
		iMaxIter = 1;			// first time through two pass align (the coarse pass), only one iteration
	for (_iIterationNumber=0; _iIterationNumber< (unsigned int)iMaxIter; _iIterationNumber++)
	{
		// zero out A, b, x, and notes
		RobustSolver::ZeroTheSystem();

		for(i=0; i<_iMatrixHeight; i++)
		{
			_pdWeights[i] = 0.0;
			sprintf_s(_pcNotes[i], _iLengthNotes, "");
		}

		FillMatrixA();  
		
		SolveXOneIteration();
		
		// extract deltas of cal and theta, update total estimates
		// Update theta_estimate
		unsigned int indexX;
		for(i=0; i<_iTotalNumberOfSubTriggers; i++)
		{
			// index of dThetaEst matches order of triggers in A matrix x,y,theta section
			indexX = i * _iNumParamsPerIndex + 2;
			_dThetaEst[i] += _dVectorX[indexX];
		}

		// update cal drift estimates
		//for(i=0; i<_iNumCalDriftTerms; i++)
		//{
		//	_dCalDriftEst[i] += _dVectorX[_iCalDriftStartCol + i];
		//}
		iFileSaveIndex++;  
	}	
	// spoof some later code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//_iIterationNumber++;  
}

void RobustSolverIterative::FillMatrixA()
{
	// fill in Matrix A,b
	_iCurrentRow = 0;
	AddAllLooseConstraints(_bPinPanelWithCalibration, true);
	
	double* Zpoly;
	Zpoly = new double[_iNumZTerms];
	// the z poly can be thought of as a partially populated 4x4 matrix:
	// 				X**0 * Y**0, X**1 * Y**0, X**2 * Y**0, X**3 * Y**0,
	//              X**0 * Y**1, X**1 * Y**1, X**2 * Y**1,      0,
	//              X**0 * Y**2, X**1 * Y**2,      0,           0,
	//              X**0 * Y**3       0,           0,           0
	unsigned int i;
	for(i =0; i<_iMaxNumCorrelations; i++)
	{
		// walk through stored correlation results and rebuild the array using updated values
		
		unsigned int iColInMatrixA = _fitInfo[i].colInMatrixA;
		unsigned int orderedTrigIndexA = iColInMatrixA / _iNumParamsPerIndex;
		unsigned int deviceNumA = _pSet->GetLayer(_fitInfo[i].fovIndexA.LayerIndex)->DeviceIndex();
		unsigned int iCamIndexA = _fitInfo[i].fovIndexA.CameraIndex;
		double	xSensorA = _fitInfo[i].xSensorA;
		double	ySensorA = _fitInfo[i].ySensorA;
		double	dxSensordzA = _fitInfo[i].dxSensordzA;
		double	dySensordzA = _fitInfo[i].dySensordzA;
		double  w = _fitInfo[i].w;
		double boardX = _fitInfo[i].boardX;
		double boardY = _fitInfo[i].boardY;
		
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
		if (_fitInfo[i].fitType == 1) // fiducial fit
		{
			_iCurrentRow = _fitInfo[i].rowInMatrix;
			double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			double	dFidRoiCenX  = _fitInfo[i].dFidRoiCenX;
			double	dFidRoiCenY  = _fitInfo[i].dFidRoiCenY;
			
			// X direction equations
			pdRow[iColInMatrixA] = w;  //X_trig
			// Ytrig = 0
			//pdRow[iColInMatrixA+2] += -ySensorA * w; // old
			pdRow[iColInMatrixA+2] = -w * (xSensorA * sin(_dThetaEst[orderedTrigIndexA])
				+ ySensorA * cos(_dThetaEst[orderedTrigIndexA]) ); //dtheta 
			// drift terms
			unsigned int calDriftCol = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
			pdRow[calDriftCol] = w * cos(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftCol+1] = -w * sin(_dThetaEst[orderedTrigIndexA]);
			// Board warp terms
			for (unsigned int j(0); j < _iNumZTerms; j++)
					pdRow[ColumnZTerm(j, deviceNumA)] = w * Zpoly[j] * 
						(dxSensordzA * cos(_dThetaEst[orderedTrigIndexA]) 
						 - dySensordzA * sin(_dThetaEst[orderedTrigIndexA]) );
			//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialXPos() - xSensorA);
			_dVectorB[_iCurrentRow] = w * 
				(dFidRoiCenX 
				- xSensorA * cos(_dThetaEst[orderedTrigIndexA])
				+ ySensorA * sin(_dThetaEst[orderedTrigIndexA]));
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FidCorr:L%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
				_fitInfo[i].fovIndexA.LayerIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				xSensorA, ySensorA,dxSensordzA,dySensordzA,dFidRoiCenX,dFidRoiCenY, _dThetaEst[orderedTrigIndexA] );
			_pdWeights[_iCurrentRow] = w;	
			_iCurrentRow++;
			
			// Y direction equations
			pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			pdRow[iColInMatrixA+1] = w;  //Y_trig
			pdRow[iColInMatrixA+2] = + w * (xSensorA * cos(_dThetaEst[orderedTrigIndexA]) 
				- ySensorA * sin(_dThetaEst[orderedTrigIndexA]) );  //dtheta_trig
			pdRow[calDriftCol] = w * sin(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftCol+1] = w * cos(_dThetaEst[orderedTrigIndexA]);
			
			for (unsigned int j(0); j < _iNumZTerms; j++)
					pdRow[ColumnZTerm(j, deviceNumA)] = w * Zpoly[j] * 
					(dxSensordzA * sin(_dThetaEst[orderedTrigIndexA]) 
						 + dySensordzA * cos(_dThetaEst[orderedTrigIndexA]) ) ;
			//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialYPos() - ySensorA);
			_dVectorB[_iCurrentRow] = w * 
				(dFidRoiCenY 
				- xSensorA * sin(_dThetaEst[orderedTrigIndexA]) 
				- ySensorA * cos(_dThetaEst[orderedTrigIndexA]) );
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FidCorr:L%d:T%d:C%d,%.2e,%.4e,%.4e, %d, %d", 
				_fitInfo[i].fovIndexA.LayerIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				w,
				boardX,boardY,  calDriftCol, deviceNumA);
			_pdWeights[_iCurrentRow] = w;
			_iCurrentRow++;
		}
		else if (_fitInfo[i].fitType == 2) // FOV to FOV fit
		{
			_iCurrentRow = _fitInfo[i].rowInMatrix;
			double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			unsigned int iColInMatrixB = _fitInfo[i].colInMatrixB;
			unsigned int orderedTrigIndexB = iColInMatrixB / _iNumParamsPerIndex;
			unsigned int deviceNumB = _pSet->GetLayer(_fitInfo[i].fovIndexB.LayerIndex)->DeviceIndex();
			unsigned int iCamIndexB = _fitInfo[i].fovIndexB.CameraIndex;
			double	xSensorB = _fitInfo[i].xSensorB;
			double	ySensorB = _fitInfo[i].ySensorB;
			double	dxSensordzB = _fitInfo[i].dxSensordzB;
			double	dySensordzB = _fitInfo[i].dySensordzB;

			// For easier debug of matrix, flip signs of equations so that lower column number 
			// has + terms, higher col number has - terms
			int eqSign;
			if (iColInMatrixA <= iColInMatrixB)
				eqSign = +1;
			else
				eqSign = -1;

			// X direction
			pdRow[iColInMatrixA] += w * eqSign;	// X_trig,   Y_trig wt = 0
			pdRow[iColInMatrixB] -= w * eqSign;  // may be in same trigger number
			pdRow[iColInMatrixA+2] += -w * eqSign * (xSensorA*sin(_dThetaEst[orderedTrigIndexA]) 
									  +ySensorA*cos(_dThetaEst[orderedTrigIndexA]) ) ;
			pdRow[iColInMatrixB+2] -= -w * eqSign * (xSensorB*sin(_dThetaEst[orderedTrigIndexB]) 
									  +ySensorB*cos(_dThetaEst[orderedTrigIndexB]) ) ;
			
			// drift terms
			unsigned int calDriftColA = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
			pdRow[calDriftColA] = w * eqSign * cos(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftColA+1] = -w * eqSign * sin(_dThetaEst[orderedTrigIndexA]);
			unsigned int calDriftColB = _iCalDriftStartCol + (deviceNumB * _iNumCameras + iCamIndexB)  * 2;
			pdRow[calDriftColB] -= w * eqSign * cos(_dThetaEst[orderedTrigIndexB]);
			pdRow[calDriftColB+1] -= -w * eqSign * sin(_dThetaEst[orderedTrigIndexB]);
			// Board warp terms
			for (unsigned int j(0); j < _iNumZTerms; j++)
			{
				pdRow[ColumnZTerm(j, deviceNumA)] = w * eqSign * Zpoly[j] 
					* (  dxSensordzA * cos(_dThetaEst[orderedTrigIndexA]) - dySensordzA * sin(_dThetaEst[orderedTrigIndexA]));
				pdRow[ColumnZTerm(j, deviceNumB)] += w * eqSign * Zpoly[j] 
					* ( - dxSensordzB * cos(_dThetaEst[orderedTrigIndexB]) + dySensordzB * sin(_dThetaEst[orderedTrigIndexB]));
			}
			_dVectorB[_iCurrentRow] = w * eqSign * 
				(- xSensorA * cos(_dThetaEst[orderedTrigIndexA]) + ySensorA * sin(_dThetaEst[orderedTrigIndexA])
				 + xSensorB * cos(_dThetaEst[orderedTrigIndexB]) - ySensorB * sin(_dThetaEst[orderedTrigIndexB]) );
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FovFovCorr:L%d:T%d:C%d_L%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e", 
				_fitInfo[i].fovIndexA.LayerIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				_fitInfo[i].fovIndexB.LayerIndex,
				_fitInfo[i].fovIndexB.TriggerIndex, _fitInfo[i].fovIndexB.CameraIndex,
				xSensorA, ySensorA, xSensorB, ySensorB);
			_pdWeights[_iCurrentRow] = w * eqSign;
			_iCurrentRow++;
			
			// Y direction
			pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			pdRow[iColInMatrixA+1] += w * eqSign;
			pdRow[iColInMatrixB+1] -= w * eqSign;  // may be in same trigger number
			pdRow[iColInMatrixA+2] += w * eqSign * (xSensorA * cos(_dThetaEst[orderedTrigIndexA]) 
									 -ySensorA * sin(_dThetaEst[orderedTrigIndexA]) ) ;
			pdRow[iColInMatrixB+2] -= w * eqSign * (xSensorB * cos(_dThetaEst[orderedTrigIndexB]) 
									 -ySensorB * sin(_dThetaEst[orderedTrigIndexB]) ) ;

			// drift terms
			//calDriftColA = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
			pdRow[calDriftColA] = w * eqSign * sin(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftColA+1] = w * eqSign * cos(_dThetaEst[orderedTrigIndexA]);
			//calDriftCol = _iCalDriftStartCol + (deviceNumB * _iNumCameras + iCamIndexB)  * 2;
			pdRow[calDriftColB] -= w * eqSign * sin(_dThetaEst[orderedTrigIndexB]);
			pdRow[calDriftColB+1] -= w * eqSign * cos(_dThetaEst[orderedTrigIndexB]);
			// Board warp terms
			for (unsigned int j(0); j < _iNumZTerms; j++)
			{
				pdRow[ColumnZTerm(j, deviceNumA)] =  w * eqSign * Zpoly[j] 
					* (  dxSensordzA * sin(_dThetaEst[orderedTrigIndexA]) + dySensordzA * cos(_dThetaEst[orderedTrigIndexA]));
				pdRow[ColumnZTerm(j, deviceNumB)] +=  w * eqSign * Zpoly[j] 
					* (- dxSensordzB * sin(_dThetaEst[orderedTrigIndexB]) - dySensordzB * cos(_dThetaEst[orderedTrigIndexB]));
			}
			_dVectorB[_iCurrentRow] = w * eqSign * 
				(- xSensorA * sin(_dThetaEst[orderedTrigIndexA]) - ySensorA * cos(_dThetaEst[orderedTrigIndexA])
				 + xSensorB * sin(_dThetaEst[orderedTrigIndexB]) + ySensorB * cos(_dThetaEst[orderedTrigIndexB]) );
			
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FovFovCorr:L%d:T%d:C%d_L%d:T%d:C%d,%.2e,%.4e,%.4e,%d,%d,%d,%d", 
				_fitInfo[i].fovIndexA.LayerIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				_fitInfo[i].fovIndexB.LayerIndex,
				_fitInfo[i].fovIndexB.TriggerIndex, _fitInfo[i].fovIndexB.CameraIndex,
				w,
				boardX,boardY, calDriftColA, calDriftColB, deviceNumA, deviceNumB);
			_pdWeights[_iCurrentRow] = w * eqSign;
			_iCurrentRow++;
		}
		else if (_fitInfo[i].fitType == 3 || _fitInfo[i].fitType == 4) // Constrain board edge
		{
			_iCurrentRow = _fitInfo[i].rowInMatrix;
			double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			double	dXOffset  = _fitInfo[i].dFidRoiCenX;
			double	dSlope    = _fitInfo[i].dFidRoiCenY;
			bool bSlopeOnly( _fitInfo[i].fitType == 4 );  // slope only fit
			
			// X direction equations
			if(!bSlopeOnly)
			{		
				pdRow[iColInMatrixA] += w;
				pdRow[iColInMatrixA+2] += -ySensorA * w;
				unsigned int calDriftCol = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
				pdRow[calDriftCol] = w * cos(_dThetaEst[orderedTrigIndexA]);
				pdRow[calDriftCol+1] = -w * sin(_dThetaEst[orderedTrigIndexA]);
				for (unsigned int j(0); j < _iNumZTerms; j++)
						pdRow[ColumnZTerm(j, deviceNumA)] = Zpoly[j] * dxSensordzA * w;
				_dVectorB[_iCurrentRow] = w * (dXOffset 
					- xSensorA * cos(_dThetaEst[orderedTrigIndexA])
					+ ySensorA * sin(_dThetaEst[orderedTrigIndexA]));
				_pdWeights[_iCurrentRow] = w;
				_iCurrentRow++;
			}

			// constrain theta_trig
			w = Weights.wRbyEdge; 
			pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			pdRow[iColInMatrixA+2] += w;
			_dVectorB[_iCurrentRow] = w * (dSlope - _dThetaEst[orderedTrigIndexA]);
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "BrdEdge:L%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
				_fitInfo[i].fovIndexA.LayerIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				xSensorA, ySensorA,dxSensordzA,dySensordzA,dXOffset, dSlope, _dThetaEst[orderedTrigIndexA] );
			_pdWeights[_iCurrentRow] = w;
			_iCurrentRow++;
			
		}
		else if (_fitInfo[i].fitType != 0)
			LOG.FireLogEntry(LogTypeError, "RobustSolverIterative::FillMatrixA():  Invalid fit type");
	}
	delete [] Zpoly;
}

// Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolverIterative::SolveXOneIteration()
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

	LOG.FireLogEntry(LogTypeSystem, "RobustSolverIterative::SolveXAlgH():BEGIN ALG_H");

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
		string fileName;
		char cTemp[100];
		sprintf_s(cTemp, 100, "C:\\Temp\\Resid_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		ofstream of(fileName.c_str());
		for(unsigned int k=0; k<_iCurrentRow; k++)
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
		LOG.FireLogEntry(LogTypeError, "RobustSolverIterative::SolveXAlgH():alg_h returned value of %d", algHRetVal);

	//LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():FINISHED ALG_H");
	LOG.FireLogEntry(LogTypeSystem, "RobustSolverIterative::SolveXAlgH():alg_h nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	delete [] resid;
	//delete [] mb;

	// copy board Z shape terms to the _zCoef array
	// First we'll put take VectorX values for Z and use it to populate Zpoly (a 4x4 array)
	// FORTRAN ORDER !!!!!
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
}

void RobustSolverIterative::ConstrainPerTrig(bool bPinPanelWithCalibration)
{
	// need local copy of this as we are solving for delta_Theta instead of theta
	// and to constrain cal drift terms
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
			unsigned int deviceNum = _pSet->GetLayer(iLayerIndex)->DeviceIndex();
			unsigned int iCols = _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->Columns();
			unsigned int iRows = _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->Rows();
			TransformCamModel camCal = 
				 _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
			complexd xySensor;
			camCal.SPix2XY(dFovOriginU, dFovOriginV, &xySensor.r, &xySensor.i);
			
			double dWeight = Weights.wXIndex;
				// Pin the first sub-trig if it is necessary
			if(bPinPanelWithCalibration && iTrigIndex == 0 && iCamIndex == 0)
				dWeight = Weights.wXIndex_PinXY;
			
			// constrain x direction
			double* begin_pin_in_image_center(&_dMatrixA[(_iCurrentRow) * _iMatrixWidth]);
			unsigned int beginIndex( indexID * _iNumParamsPerIndex);
			begin_pin_in_image_center[beginIndex+0] = dWeight;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = -dWeight * 
				(xySensor.r*sin(_dThetaEst[indexID])
				+ xySensor.i*cos(_dThetaEst[indexID])  );
			_dVectorB[_iCurrentRow] = dWeight * 
				(_pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(2)
				- xySensor.r*cos(_dThetaEst[indexID])
				+ xySensor.i*sin(_dThetaEst[indexID]));
			// ignoring the calibration drift terms, very small values for this lightly weighted equation

			_pdWeights[_iCurrentRow] = dWeight;
			// constrain yTrig value
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+1] = dWeight;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = dWeight * 
				(xySensor.r*cos(_dThetaEst[indexID])
				- xySensor.i*sin(_dThetaEst[indexID])  );
			_dVectorB[_iCurrentRow] = dWeight * 
				(_pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(5)
				- xySensor.r*sin(_dThetaEst[indexID])
				- xySensor.i*cos(_dThetaEst[indexID]));

			// constrain theata to zero
				// Pin the first sub-trig if it is necessary
			if(bPinPanelWithCalibration && iTrigIndex == 0 && iCamIndex == 0)
				dWeight = Weights.wXINdex_PinTheta;

			_pdWeights[_iCurrentRow] = dWeight;
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+2] = dWeight;
			_pdWeights[_iCurrentRow] = dWeight;
			_iCurrentRow++;
		}
	}
	// constrain each of the cal drift items to zero
	double* pdRow;
	for (unsigned int i(0); i < _iNumCalDriftTerms; i++)
	{
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		pdRow[_iCalDriftStartCol + i] = Weights.wCalDriftZero;
		_pdWeights[_iCurrentRow] = Weights.wCalDriftZero;
		_iCurrentRow++;
	}
	// constrain sum of cal drift per device to zero
	for  (unsigned int i(0); i < _iNumDevices; i++)
	{
		// x direction
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		for  (unsigned int j(0); j < _iNumCameras; j++)
		{
			pdRow[_iCalDriftStartCol + (i * _iNumCameras + j)*2] = Weights.wCalDriftZero;
		}
		_pdWeights[_iCurrentRow] = Weights.wCalDriftZero;
		_iCurrentRow++;
		// y direction
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		for  (unsigned int j(0); j < _iNumCameras; j++)
		{
			pdRow[_iCalDriftStartCol + (i * _iNumCameras + j)*2 + 1] = Weights.wCalDriftZero;
		}
		_pdWeights[_iCurrentRow] = Weights.wCalDriftZero;
		_iCurrentRow++;
	}
}

void  RobustSolverIterative::Pix2Board(POINTPIX pix, FovIndex fovindex, POINT2D_C *xyBoard)
{
	// Pix2Board in Iterative overloads the same function in the CM version
	// This function is called by the FlattenFiducial function, which is defined in CM
	// 
	// p2Board translates between a pixel location (for a given Layer, trigger and camera defined by index) to xyBoard location
	// remember that xyBoard is on a warped surface, so the xy distance between marks is likely to be less than the
	// CAD distance
	// Also remember that this board model has not been aligned to CAD
	unsigned int iLayerIndex( fovindex.LayerIndex );
	unsigned int iTrigIndex( fovindex.TriggerIndex );
	unsigned int iCamIndex( fovindex.CameraIndex );
	unsigned int iOrderedTrigIndex = (*_pFovOrderMap)[fovindex];
	unsigned int iVectorXIndex = iOrderedTrigIndex *_iNumParamsPerIndex;
	unsigned int deviceNum = _pSet->GetLayer(iLayerIndex)->DeviceIndex();
	
	TransformCamModel camCal = 
				 _pSet->GetLayer(iLayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
	complexd xySensor;
	double x, y;
	camCal.SPix2XY(pix.u, pix.v, &x, &y);
	xySensor.r = x + _dVectorX[_iCalDriftStartCol + (deviceNum * _iNumCameras + iCamIndex) * 2 ];
	xySensor.i = y + _dVectorX[_iCalDriftStartCol + (deviceNum * _iNumCameras + iCamIndex) * 2 + 1];

	// use board transform to go to board x,y
	if (_iIterationNumber==1)
	{
		// WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
		xyBoard->x =        1         * xySensor.r  - _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
		xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +        1         * xySensor.i  + _dVectorX[iVectorXIndex+1];
	}
	else
	{
		// iterative process used trig functions
		xyBoard->x = cos(_dThetaEst[iOrderedTrigIndex]) * xySensor.r  - sin(_dThetaEst[iOrderedTrigIndex]) * xySensor.i  + _dVectorX[iVectorXIndex+0];
		xyBoard->y = sin(_dThetaEst[iOrderedTrigIndex]) * xySensor.r  + cos(_dThetaEst[iOrderedTrigIndex]) * xySensor.i  + _dVectorX[iVectorXIndex+1];
	}
	
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
	//xyBoard->x =               1 * xySensor.r             -  _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
	//xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +                1 * xySensor.i             + _dVectorX[iVectorXIndex+1];
	if (_iIterationNumber==1)
	{
		// WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
		xyBoard->x =        1         * xySensor.r  - _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
		xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +        1         * xySensor.i  + _dVectorX[iVectorXIndex+1];
	}
	else
	{
		// iterative process used trig functions
		xyBoard->x = cos(_dThetaEst[iOrderedTrigIndex]) * xySensor.r  - sin(_dThetaEst[iOrderedTrigIndex]) * xySensor.i  + _dVectorX[iVectorXIndex+0];
		xyBoard->y = sin(_dThetaEst[iOrderedTrigIndex]) * xySensor.r  + cos(_dThetaEst[iOrderedTrigIndex]) * xySensor.i  + _dVectorX[iVectorXIndex+1];
	}
}

// don't actually fill in MatrixA, b  -- add info _fitInfo array
bool RobustSolverIterative::AddFovFovOvelapResults(FovFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// First Fov's information
	unsigned int iLayerIndexA = pOverlap->GetFirstMosaicLayer()->Index();
	unsigned int iTrigIndexA = pOverlap->GetFirstTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetFirstCameraIndex();
	FovIndex index1(iLayerIndexA, iTrigIndexA, iCamIndexA); 
	
	// Second Fov's information
	unsigned int iLayerIndexB = pOverlap->GetSecondMosaicLayer()->Index();
	unsigned int iTrigIndexB = pOverlap->GetSecondTriggerIndex();
	unsigned int iCamIndexB = pOverlap->GetSecondCameraIndex();
	FovIndex index2(iLayerIndexB, iTrigIndexB, iCamIndexB); 
	
	//double* pdRow;
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
		CorrelationResult result;bool bFlag = i->GetCorrelationResult(&result);
		if(!bFlag) 
			continue;
		_fitInfo[_iCorrelationNum].fitType=2;
		_fitInfo[_iCorrelationNum].fovIndexA.LayerIndex = iLayerIndexA;
		_fitInfo[_iCorrelationNum].fovIndexA.TriggerIndex = iTrigIndexA;
		_fitInfo[_iCorrelationNum].fovIndexA.CameraIndex = iCamIndexA;
		_fitInfo[_iCorrelationNum].colInMatrixA = (*_pFovOrderMap)[index1] *_iNumParamsPerIndex;
		_fitInfo[_iCorrelationNum].fovIndexB.LayerIndex = iLayerIndexB;
		_fitInfo[_iCorrelationNum].fovIndexB.TriggerIndex  = iTrigIndexB;
		_fitInfo[_iCorrelationNum].fovIndexB.CameraIndex = iCamIndexB;
		_fitInfo[_iCorrelationNum].colInMatrixB = (*_pFovOrderMap)[index2] *_iNumParamsPerIndex;

		_fitInfo[_iCorrelationNum].rowInMatrix = _iCurrentRow;

		double w = Weights.CalWeight(&(*i));
		// include 0 weight rows for now....
		//if(w <= 0) 
		//	continue;
		//pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		// Get Centers of ROIs
		double rowImgA = (i->GetFirstRoi().FirstRow + i->GetFirstRoi().LastRow)/ 2.0;
		double colImgA = (i->GetFirstRoi().FirstColumn + i->GetFirstRoi().LastColumn)/ 2.0;

		double rowImgB = (i->GetSecondRoi().FirstRow + i->GetSecondRoi().LastRow)/ 2.0;
		double colImgB = (i->GetSecondRoi().FirstColumn + i->GetSecondRoi().LastColumn)/ 2.0;

		// Get offset
		double offsetRows = result.RowOffset;
		double offsetCols = result.ColOffset;
		rowImgB += offsetRows;
		colImgB += offsetCols;
		// Add a equataions 
		TransformCamModel camCalA = 
			_pSet->GetLayer(iLayerIndexA)->GetImage(iTrigIndexA, iCamIndexA)->GetTransformCamCalibration();
		camCalA.SPix2XY(colImgA, rowImgA, &_fitInfo[_iCorrelationNum].xSensorA, &_fitInfo[_iCorrelationNum].ySensorA);
		camCalA.dSPix2XY(colImgA, rowImgA, &_fitInfo[_iCorrelationNum].dxSensordzA, &_fitInfo[_iCorrelationNum].dySensordzA);
		
		TransformCamModel camCalB = 
			_pSet->GetLayer(iLayerIndexB)->GetImage(iTrigIndexB, iCamIndexB)->GetTransformCamCalibration();
		camCalB.SPix2XY(colImgB, rowImgB, &_fitInfo[_iCorrelationNum].xSensorB, &_fitInfo[_iCorrelationNum].ySensorB);
		camCalB.dSPix2XY(colImgB, rowImgB, &_fitInfo[_iCorrelationNum].dxSensordzB, &_fitInfo[_iCorrelationNum].dySensordzB);

		pair<double, double> imgAOverlapCenter = i->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
		pair<double, double> imgBOverlapCenter = i->GetSecondImg()->ImageToWorld(rowImgB,colImgB);
		_fitInfo[_iCorrelationNum].boardX = (imgAOverlapCenter.first + imgBOverlapCenter.first ) / 2.0;
		_fitInfo[_iCorrelationNum].boardY = (imgAOverlapCenter.second + imgBOverlapCenter.second) / 2.0;
		_fitInfo[_iCorrelationNum].dFidRoiCenX = 0;  // not used shouldn't need to reset here...
		_fitInfo[_iCorrelationNum].dFidRoiCenY = 0;
		_fitInfo[_iCorrelationNum].w = w;
		
		_iCurrentRow++;
		_iCurrentRow++;
		_iCorrelationNum++;
	}
	return( true );
}

bool RobustSolverIterative::AddCadFovOvelapResults(CadFovOverlap* pOverlap)
{
	// there are no CAD apperture fits in the camera model fit
	// TODO TODO   is this true????????
	return( true );
}

bool RobustSolverIterative::AddFidFovOvelapResults(FidFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);
	
	// Fov's information
	unsigned int iLayerIndexA= pOverlap->GetMosaicLayer()->Index();
	unsigned int iTrigIndexA = pOverlap->GetTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetCameraIndex();
	FovIndex index(iLayerIndexA, iTrigIndexA, iCamIndexA); 
		
	CorrelationPair* pPair = pOverlap->GetCoarsePair();

	// Validation check for correlation pair
	CorrelationResult result;
	bool bFlag = pPair->GetCorrelationResult(&result);
	if(!bFlag) 
		return(false);
	double w = Weights.CalWeight(pPair) * Weights.RelativeFidFovCamModWeight;
	if(w <= 0) 
		return(false);
	_fitInfo[_iCorrelationNum].fitType=1;
	_fitInfo[_iCorrelationNum].fovIndexA.LayerIndex = iLayerIndexA;
	_fitInfo[_iCorrelationNum].fovIndexA.TriggerIndex = iTrigIndexA;
	_fitInfo[_iCorrelationNum].fovIndexA.CameraIndex = iCamIndexA;
	_fitInfo[_iCorrelationNum].colInMatrixA = (*_pFovOrderMap)[index] *_iNumParamsPerIndex;
	_fitInfo[_iCorrelationNum].rowInMatrix = _iCurrentRow;
	_fitInfo[_iCorrelationNum].w = w;
	
	// Get Centers of ROIs (fiducial image is always the second one)
	double rowImgA = (pPair->GetFirstRoi().FirstRow + pPair->GetFirstRoi().LastRow)/ 2.0;
	double colImgA = (pPair->GetFirstRoi().FirstColumn + pPair->GetFirstRoi().LastColumn)/ 2.0;

	double rowImgB = (pPair->GetSecondRoi().FirstRow + pPair->GetSecondRoi().LastRow)/ 2.0;
	double colImgB = (pPair->GetSecondRoi().FirstColumn + pPair->GetSecondRoi().LastColumn)/ 2.0;
	double dFidRoiCenX, dFidRoiCenY; // ROI center of fiducial image (not fiducail center or image center) in world space
	pOverlap->GetFidImage()->ImageToWorld(rowImgB, colImgB, &dFidRoiCenX, &dFidRoiCenY);
	_fitInfo[_iCorrelationNum].dFidRoiCenX = dFidRoiCenX;
	_fitInfo[_iCorrelationNum].dFidRoiCenY = dFidRoiCenY;
	// Get offset
	double offsetRows = result.RowOffset;
	double offsetCols = result.ColOffset;
	rowImgA -= offsetRows;
	colImgA -= offsetCols;
	// Add a equataions 
	TransformCamModel camCalA = 
				_pSet->GetLayer(iLayerIndexA)->GetImage(iTrigIndexA, iCamIndexA)->GetTransformCamCalibration();
	camCalA.SPix2XY(colImgA, rowImgA, &_fitInfo[_iCorrelationNum].xSensorA, &_fitInfo[_iCorrelationNum].ySensorA);
	camCalA.dSPix2XY(colImgA, rowImgA, &_fitInfo[_iCorrelationNum].dxSensordzA, &_fitInfo[_iCorrelationNum].dySensordzA);

	pair<double, double> imgAOverlapCenter = pPair->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
	_fitInfo[_iCorrelationNum].boardX = imgAOverlapCenter.first;
	_fitInfo[_iCorrelationNum].boardY = imgAOverlapCenter.second;
	
	_iCurrentRow++;
	_iCurrentRow++;
	_iCorrelationNum++;
	return( true );
}

// Add constraint base on panel edge
bool RobustSolverIterative::AddPanelEdgeContraints(
	MosaicLayer* pLayer, unsigned int iCamIndex, unsigned int iTrigIndex,
	double dXOffset, double dSlope, bool bSlopeOnly)
{
	// dXOffset is the X position of the 0,0 pixel of the imager,
	// dSlope is the rotation (estimate) of the imager
	// 
	// Position of equation in Matirix
	FovIndex index(pLayer->Index(), iTrigIndex, iCamIndex); 
	unsigned int indexID( (*_pFovOrderMap)[index] );
	unsigned int iCols = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->Columns();
	unsigned int iRows = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->Rows();
	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
	unsigned int iFOVPosA( indexID * _iNumParamsPerIndex);
	
	// Add a equataions 
	// model the equations after the fiducial alignment set
	// x_trig - s_y * theta_trig = dsx/dz * Z = x_fid - s_x
	double w = Weights.wXbyEdge;
	TransformCamModel camCalA = 
				_pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetTransformCamCalibration();
	camCalA.SPix2XY(0.0, 0.0, &_fitInfo[_iCorrelationNum].xSensorA, &_fitInfo[_iCorrelationNum].ySensorA);
	camCalA.dSPix2XY(0.0, 0.0, &_fitInfo[_iCorrelationNum].dxSensordzA, &_fitInfo[_iCorrelationNum].dySensordzA);

	// approximate position of the 0,0 pixel on the surface of the board
	_fitInfo[_iCorrelationNum].boardX = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(2);
	_fitInfo[_iCorrelationNum].boardY = _pSet->GetLayer(index.LayerIndex)->GetImage(iTrigIndex, iCamIndex)->GetNominalTransform().GetItem(5);
	
	if(!bSlopeOnly)
		_fitInfo[_iCorrelationNum].fitType=3;
	else
		_fitInfo[_iCorrelationNum].fitType=4;
	_fitInfo[_iCorrelationNum].fovIndexA.LayerIndex = index.LayerIndex;
	_fitInfo[_iCorrelationNum].fovIndexA.TriggerIndex = iTrigIndex;
	_fitInfo[_iCorrelationNum].fovIndexA.CameraIndex = iCamIndex;
	_fitInfo[_iCorrelationNum].colInMatrixA = (*_pFovOrderMap)[index] *_iNumParamsPerIndex;
	_fitInfo[_iCorrelationNum].rowInMatrix = _iCurrentRow;
	_fitInfo[_iCorrelationNum].w = w;
	_fitInfo[_iCorrelationNum].dFidRoiCenX = dXOffset;
	_fitInfo[_iCorrelationNum].dFidRoiCenY = dSlope;
	_iCurrentRow++;
	_iCurrentRow++;
	_iCorrelationNum++;
	
	return(true);
}

#pragma region Debug
// Debug
// Vector X output
void RobustSolverIterative::OutputVectorXCSV(string filename) const
{
	ofstream of(filename.c_str());

	of << std::scientific;

	string line;
	unsigned int i;
	unsigned int j;
	
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		unsigned int iCamIndex( k->first.CameraIndex);
		unsigned int iIndexID( k->second );
		if (_pSet->GetLayer(k->first.LayerIndex)->IsFirstCamOfSubTrigger(k->first.CameraIndex)) 
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
			of << "," << _dThetaEst[iIndexID]<<  std::endl;
		}
	}
	i=0;
	for( j=0; j<_iNumDevices; ++j)
	{
		of << "Device_" << j ;
		for (unsigned int k(0); k < _iNumCameras*2; k++)
		{
			double d = _dVectorX[_iCalDriftStartCol + i];
			i++;
			of << "," << d ;
		}
		of <<  std::endl;
	}
	
	of << "Z ";
	for(j=0; j<_iNumZTerms + (_iNumDevices-1)*2; ++j)
	{
		

		double d = _dVectorX[_iStartColZTerms + j];
		of << "," << d;
	}
	of <<  std::endl;
	of.close();
}
#pragma endregion


