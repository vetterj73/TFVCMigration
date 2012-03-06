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

Columns 0 through  _iTotalNumberOfTriggers * _iNumParamsPerIndex -1
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
	_iMaxIterations = 5;
	_iCalDriftStartCol = _iTotalNumberOfTriggers * _iNumParamsPerIndex;
	_iMatrixWidth = _iCalDriftStartCol + _iNumZTerms + _iNumCalDriftTerms;
	
	_dThetaEst = new double[_iTotalNumberOfTriggers];
	_fitInfo = new fitInfo[_iMaxNumCorrelations];
	ZeroTheSystem();
}
RobustSolverIterative::~RobustSolverIterative()
{
	delete [] _dThetaEst;
	delete [] _fitInfo;
	//delete [] _dMatrixA;
	//delete [] _dMatrixACopy;
	//delete [] _dVectorB;
	//delete [] _dVectorBCopy;
	//delete [] _dVectorX;
	//delete [] _pdWeights;
	//delete [] _pcNotes;
}

void RobustSolverIterative::ZeroTheSystem()
{
	_iCurrentRow = 0;

	unsigned int i;
	for(i=0; i<_iMatrixSize; i++)
		_dMatrixA[i] = 0.0;

	for(i=0; i<_iMatrixHeight; i++)
	{
		_dVectorB[i] = 0.0;
		_pdWeights[i] = 0.0;
		sprintf_s(_pcNotes[i], _iLengthNotes, "");
	}

	for(i =0; i<_iMatrixWidth; i++)
		_dVectorX[i] = 0.0;
	for(i =0; i<_iMaxNumCorrelations; i++)
	{
		// reset the fit info struct
		_fitInfo[i].fitType=0;	
		_fitInfo[i].fovIndexA.CameraIndex=0;
		_fitInfo[i].fovIndexA.IlluminationIndex=0;
		_fitInfo[i].fovIndexA.TriggerIndex=0;
		_fitInfo[i].fovIndexB.CameraIndex=0;
		_fitInfo[i].fovIndexB.IlluminationIndex=0;
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
	// first _iTotalNumberOfTriggers * _iNumParamsPerIndex columns
	// contain Xtrig, Ytrig, dTheta
	// next _iNumCalDriftTerms = _iNumDevices * _iNumCameras * 2
	// contain ds_x, ds_y [device][camera] 
	// last _iNumZTerms = 10;	are Z terms
	unsigned int i;
	_iIterationNumber = 0;
	for(i =0; i<_iTotalNumberOfTriggers; i++)
		_dThetaEst[i]=0;
	//for(i =0; i<_iNumCalDriftTerms; i++)
	//	_dCalDriftEst[i]=0;

	for (_iIterationNumber=0; _iIterationNumber< _iMaxIterations; _iIterationNumber++)
	{
		// zero out A, b, x, and notes
		_iCurrentRow = 0;
		iFileSaveIndex = _iIterationNumber;
		for(i=0; i<_iMatrixSize; i++)
			_dMatrixA[i] = 0.0;

		for(i=0; i<_iMatrixHeight; i++)
		{
			_dVectorB[i] = 0.0;
			_pdWeights[i] = 0.0;
			sprintf_s(_pcNotes[i], _iLengthNotes, "");
		}

		for(i =0; i<_iMatrixWidth; i++)
			_dVectorX[i] = 0.0;
		FillMatrixA();  
		SolveXOneIteration();
		// extract deltas of cal and theta, update total estimates
		// Update theta_estimate
		unsigned int indexX;
		for(i=0; i<_iTotalNumberOfTriggers; i++)
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
	}	
	// spoof some later code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	_iIterationNumber++;  
}

void RobustSolverIterative::FillMatrixA()
{
	// fill in Matrix A,b
	_iCurrentRow = 0;
	ConstrainZTerms();
	ConstrainPerTrig();
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
		_iCurrentRow = _fitInfo[i].rowInMatrix;
		double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		unsigned int iColInMatrixA = _fitInfo[i].colInMatrixA;
		unsigned int orderedTrigIndexA = iColInMatrixA / _iNumParamsPerIndex;
		unsigned int deviceNumA = _pSet->GetLayer(_fitInfo[i].fovIndexA.IlluminationIndex)->DeviceIndex();
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
					pdRow[_iMatrixWidth - _iNumZTerms + j] = w * Zpoly[j] * 
						(dxSensordzA * cos(_dThetaEst[orderedTrigIndexA]) 
						 - dySensordzA * sin(_dThetaEst[orderedTrigIndexA]) );
			//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialXPos() - xSensorA);
			_dVectorB[_iCurrentRow] = w * 
				(dFidRoiCenX 
				- xSensorA * cos(_dThetaEst[orderedTrigIndexA])
				+ ySensorA * sin(_dThetaEst[orderedTrigIndexA]));
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FidCorr:%d:I%d:C%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
				_fitInfo[i].fovIndexA.IlluminationIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				xSensorA, ySensorA,dxSensordzA,dySensordzA,dFidRoiCenX,dFidRoiCenY, _dThetaEst[orderedTrigIndexA] );
			_pdWeights[_iCurrentRow] = w;
			// Y direction equations
			_iCurrentRow++;
			pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			pdRow[iColInMatrixA+1] = w;  //Y_trig
			pdRow[iColInMatrixA+2] = + w * (xSensorA * cos(_dThetaEst[orderedTrigIndexA]) 
				- ySensorA * sin(_dThetaEst[orderedTrigIndexA]) );  //dtheta_trig
			pdRow[calDriftCol] = w * sin(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftCol+1] = w * cos(_dThetaEst[orderedTrigIndexA]);
			
			for (unsigned int j(0); j < _iNumZTerms; j++)
					pdRow[_iMatrixWidth - _iNumZTerms + j] = w * Zpoly[j] * 
					(dxSensordzA * sin(_dThetaEst[orderedTrigIndexA]) 
						 + dySensordzA * cos(_dThetaEst[orderedTrigIndexA]) ) ;
			//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialYPos() - ySensorA);
			_dVectorB[_iCurrentRow] = w * 
				(dFidRoiCenY 
				- xSensorA * sin(_dThetaEst[orderedTrigIndexA]) 
				- ySensorA * cos(_dThetaEst[orderedTrigIndexA]) );
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FidCorr:%d:I%d:C%d,%.2e,%.4e,%.4e, %d, %d", 
				_fitInfo[i].fovIndexA.IlluminationIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				w,
				boardX,boardY,  calDriftCol, deviceNumA);
			_pdWeights[_iCurrentRow] = w;
	
			_iCurrentRow++;

			
		}
		else if (_fitInfo[i].fitType == 2) // FOV to FOV fit
		{
			unsigned int iColInMatrixB = _fitInfo[i].colInMatrixB;
			unsigned int orderedTrigIndexB = iColInMatrixB / _iNumParamsPerIndex;
			unsigned int deviceNumB = _pSet->GetLayer(_fitInfo[i].fovIndexB.IlluminationIndex)->DeviceIndex();
			unsigned int iCamIndexB = _fitInfo[i].fovIndexB.CameraIndex;
			double	xSensorB = _fitInfo[i].xSensorB;
			double	ySensorB = _fitInfo[i].ySensorB;
			double	dxSensordzB = _fitInfo[i].dxSensordzB;
			double	dySensordzB = _fitInfo[i].dySensordzB;

			pdRow[iColInMatrixA] += w;	// X_trig,   Y_trig wt = 0
			pdRow[iColInMatrixB] -= w;  // may be in same trigger number
			pdRow[iColInMatrixA+2] += -w * (xSensorA*sin(_dThetaEst[orderedTrigIndexA]) 
									  +ySensorA*cos(_dThetaEst[orderedTrigIndexA]) ) ;
			pdRow[iColInMatrixB+2] -= -w * (xSensorB*sin(_dThetaEst[orderedTrigIndexB]) 
									  +ySensorB*cos(_dThetaEst[orderedTrigIndexB]) ) ;
			
			// drift terms
			unsigned int calDriftColA = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
			pdRow[calDriftColA] = w * cos(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftColA+1] = -w * sin(_dThetaEst[orderedTrigIndexA]);
			unsigned int calDriftColB = _iCalDriftStartCol + (deviceNumB * _iNumCameras + iCamIndexB)  * 2;
			pdRow[calDriftColB] -= w * cos(_dThetaEst[orderedTrigIndexB]);
			pdRow[calDriftColB+1] -= -w * sin(_dThetaEst[orderedTrigIndexB]);
			// Board warp terms
			for (unsigned int j(0); j < _iNumZTerms; j++)
				pdRow[_iMatrixWidth - _iNumZTerms + j] = w * Zpoly[j] 
					* (  dxSensordzA * cos(_dThetaEst[orderedTrigIndexA]) - dySensordzA * sin(_dThetaEst[orderedTrigIndexA])
					   - dxSensordzB * cos(_dThetaEst[orderedTrigIndexB]) + dySensordzB * sin(_dThetaEst[orderedTrigIndexB]));
			_dVectorB[_iCurrentRow] = w * 
				(- xSensorA * cos(_dThetaEst[orderedTrigIndexA]) + ySensorA * sin(_dThetaEst[orderedTrigIndexA])
				 + xSensorB * cos(_dThetaEst[orderedTrigIndexB]) - ySensorB * sin(_dThetaEst[orderedTrigIndexB]) );
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FovFovCorr:I%d:T%d:C%d_I%d:T%d:C%d,%.4e,%.4e,%.4e,%.4e", 
				_fitInfo[i].fovIndexA.IlluminationIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				_fitInfo[i].fovIndexB.IlluminationIndex,
				_fitInfo[i].fovIndexB.TriggerIndex, _fitInfo[i].fovIndexB.CameraIndex,
				xSensorA, ySensorA, xSensorB, ySensorB);
			_pdWeights[_iCurrentRow] = w;
	
			_iCurrentRow++;
			pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
			// Y direction
			pdRow[iColInMatrixA+1] += w;
			pdRow[iColInMatrixB+1] -= w;  // may be in same trigger number
			pdRow[iColInMatrixA+2] += w * (xSensorA * cos(_dThetaEst[orderedTrigIndexA]) 
									 -ySensorA * sin(_dThetaEst[orderedTrigIndexA]) ) ;
			pdRow[iColInMatrixB+2] -= w * (xSensorB * cos(_dThetaEst[orderedTrigIndexB]) 
									 -ySensorB * sin(_dThetaEst[orderedTrigIndexB]) ) ;

			// drift terms
			//calDriftColA = _iCalDriftStartCol + (deviceNumA * _iNumCameras + iCamIndexA)  * 2;
			pdRow[calDriftColA] = w * sin(_dThetaEst[orderedTrigIndexA]);
			pdRow[calDriftColA+1] = w * cos(_dThetaEst[orderedTrigIndexA]);
			//calDriftCol = _iCalDriftStartCol + (deviceNumB * _iNumCameras + iCamIndexB)  * 2;
			pdRow[calDriftColB] -= w * sin(_dThetaEst[orderedTrigIndexB]);
			pdRow[calDriftColB+1] -= w * cos(_dThetaEst[orderedTrigIndexB]);
			// Board warp terms
			for (unsigned int j(0); j < _iNumZTerms; j++)
				pdRow[_iMatrixWidth - _iNumZTerms + j] =  w * Zpoly[j] 
					* (  dxSensordzA * sin(_dThetaEst[orderedTrigIndexA]) + dySensordzA * cos(_dThetaEst[orderedTrigIndexA])
					   - dxSensordzB * sin(_dThetaEst[orderedTrigIndexB]) - dySensordzB * cos(_dThetaEst[orderedTrigIndexB]));

			_dVectorB[_iCurrentRow] = w * 
				(- xSensorA * sin(_dThetaEst[orderedTrigIndexA]) - ySensorA * cos(_dThetaEst[orderedTrigIndexA])
				 + xSensorB * sin(_dThetaEst[orderedTrigIndexB]) + ySensorB * cos(_dThetaEst[orderedTrigIndexB]) );
			
			sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FovFovCorr:%d:I%d:C%d_%d:I%d:C%d,%.2e,%.4e,%.4e,%d,%d,%d,%d", 
				_fitInfo[i].fovIndexA.IlluminationIndex,
				_fitInfo[i].fovIndexA.TriggerIndex, _fitInfo[i].fovIndexA.CameraIndex,
				_fitInfo[i].fovIndexB.IlluminationIndex,
				_fitInfo[i].fovIndexB.TriggerIndex, _fitInfo[i].fovIndexB.CameraIndex,
				w,
				boardX,boardY, calDriftColA, calDriftColB, deviceNumA, deviceNumB);
			_pdWeights[_iCurrentRow] = w;
	
			_iCurrentRow++;
		
		}
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
	bool bRemoveEmptyRows = false;
	//bRemoveEmptyRows = false;
	int* mb = new int[_iMatrixWidth];
	unsigned int iEmptyRows;
	unsigned int bw = ReorderAndTranspose(bRemoveEmptyRows, mb, &iEmptyRows);

	double*	resid = new double[_iMatrixHeight];
	double scaleparm = 0;
	double cond = 0;

	LOG.FireLogEntry(LogTypeSystem, "RobustSolverIterative::SolveXAlgH():BEGIN ALG_H");

	int algHRetVal = 
		alg_h(                // Robust regression by Huber's "Algorithm H".
			// Inputs //			

			_iMatrixHeight,             /* Number of equations */
			_iMatrixWidth,				/* Number of unknowns */
			_dMatrixA,				// System matrix
			_iMatrixHeight,			//AStride
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
		ofstream of(fileName);
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
		LOG.FireLogEntry(LogTypeError, "RobustSolverIterative::SolveXAlgH():alg_h returned value of %d", algHRetVal);

	//LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():FINISHED ALG_H");
	LOG.FireLogEntry(LogTypeSystem, "RobustSolverIterative::SolveXAlgH():alg_h nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	delete [] resid;
	delete [] mb;

	// copy board Z shape terms to the _zCoef array
	// First we'll put take VectorX values for Z and use it to populate Zpoly (a 4x4 array)
	// FORTRAN ORDER !!!!!
	_zCoef[0][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 0];
	_zCoef[1][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 1];
	_zCoef[2][0] = _dVectorX[_iMatrixWidth - _iNumZTerms + 2];
	_zCoef[3][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 3];
	
	_zCoef[0][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 4];
	_zCoef[1][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 5];
	_zCoef[2][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 6];
	_zCoef[3][1] = 0;
	
	_zCoef[0][2] = _dVectorX[_iMatrixWidth- _iNumZTerms + 7];
	_zCoef[1][2] = _dVectorX[_iMatrixWidth- _iNumZTerms + 8];
	_zCoef[2][2] = 0;
	_zCoef[3][2] = 0;

	_zCoef[0][3] = _dVectorX[_iMatrixWidth- _iNumZTerms + 9];
	_zCoef[1][3] = 0;
	_zCoef[2][3] = 0;
	_zCoef[3][3] = 0;
}


void RobustSolverIterative::ConstrainPerTrig()
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
		unsigned int iCamIndex( k->first.CameraIndex);
		if (iCamIndex == 0)  // first camera (logical camera) is always numbered 0
		{
			unsigned int indexID( k->second );
			unsigned int iIllumIndex( k->first.IlluminationIndex);
			unsigned int iTrigIndex( k->first.TriggerIndex);
			unsigned int deviceNum = _pSet->GetLayer(iIllumIndex)->DeviceIndex();
			unsigned int iCols = _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->Columns();
			unsigned int iRows = _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->Rows();
			TransformCamModel camCal = 
				 _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetTransformCamCalibration();
			complexd xySensor;
			xySensor.r = htcorrp(iRows, iCols,
				dFovOriginU, dFovOriginV,
				_iNumBasisFunctions, _iNumBasisFunctions,
				(float *)camCal.S[CAL_ARRAY_X],  _iNumBasisFunctions);
			xySensor.i = htcorrp(iRows, iCols,
				dFovOriginU, dFovOriginV,
				_iNumBasisFunctions, _iNumBasisFunctions,
				(float *)camCal.S[CAL_ARRAY_Y],  _iNumBasisFunctions);
			
			// constrain x direction
			double* begin_pin_in_image_center(&_dMatrixA[(_iCurrentRow) * _iMatrixWidth]);
			unsigned int beginIndex( indexID * _iNumParamsPerIndex);
			begin_pin_in_image_center[beginIndex+0] = Weights.wXIndex;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = -Weights.wXIndex * 
				(xySensor.r*sin(_dThetaEst[indexID])
				+ xySensor.i*cos(_dThetaEst[indexID])  );
			_dVectorB[_iCurrentRow] = Weights.wXIndex * 
				(_pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetNominalTransform().GetItem(2)
				- xySensor.r*cos(_dThetaEst[indexID])
				+ xySensor.i*sin(_dThetaEst[indexID]));
			// ignoring the calibration drift terms, very small values for this lightly weighted equation

			_pdWeights[_iCurrentRow] = Weights.wXIndex;
			// constrain yTrig value
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+1] = Weights.wXIndex;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = Weights.wXIndex * 
				(xySensor.r*cos(_dThetaEst[indexID])
				- xySensor.i*sin(_dThetaEst[indexID])  );
			_dVectorB[_iCurrentRow] = Weights.wXIndex * 
				(_pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetNominalTransform().GetItem(5)
				- xySensor.r*sin(_dThetaEst[indexID])
				- xySensor.i*cos(_dThetaEst[indexID]));

			// constrain theata to zero
			_pdWeights[_iCurrentRow] = Weights.wXIndex;
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+2] = Weights.wXIndex;
			_pdWeights[_iCurrentRow] = Weights.wXIndex;
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
void  RobustSolverIterative::Pix2Board(POINTPIX pix, FovIndex fovindex, POINT2D *xyBoard)
{
	// p2Board translates between a pixel location (for a given illumination, trigger and camera defined by index) to xyBoard location
	// remember that xyBoard is on a warped surface, so the xy distance between marks is likely to be less than the
	// CAD distance
	// Also remember that this board model has not been aligned to CAD
	unsigned int iIllumIndex( fovindex.IlluminationIndex );
	unsigned int iTrigIndex( fovindex.TriggerIndex );
	unsigned int iCamIndex( fovindex.CameraIndex );
	unsigned int iOrderedTrigIndex = (*_pFovOrderMap)[fovindex];
	unsigned int iVectorXIndex = iOrderedTrigIndex *_iNumParamsPerIndex;
	unsigned int deviceNum = _pSet->GetLayer(iIllumIndex)->DeviceIndex();
		
	TransformCamModel camCal = 
				 _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetTransformCamCalibration();
	complexd xySensor;
	xySensor.r = htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.S[CAL_ARRAY_X],  _iNumBasisFunctions)
					+ _dVectorX[_iCalDriftStartCol + (deviceNum * _iNumCameras + iCamIndex)  * 2 ];
	xySensor.i = htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.S[CAL_ARRAY_Y],  _iNumBasisFunctions)
					+ _dVectorX[_iCalDriftStartCol + (deviceNum * _iNumCameras + iCamIndex)  * 2 + 1];

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
	for (unsigned int k(0); k < _iNumBasisFunctions; k++)
		for (unsigned int j(0); j < _iNumBasisFunctions; j++)
			Zestimate += _zCoef[j][k] * pow(xyBoard->x,(double)j) * pow(xyBoard->y,(double)k);

	// accurate Z value allows accurate calc of xySensor
	xySensor.r +=  Zestimate *
			htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.dSdz[CAL_ARRAY_X],  _iNumBasisFunctions);
	xySensor.i += Zestimate *
			htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.dSdz[CAL_ARRAY_Y],  _iNumBasisFunctions);
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
	unsigned int iMosicIndexA = pOverlap->GetFirstMosaicImage()->Index();
	unsigned int iTrigIndexA = pOverlap->GetFirstTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetFirstCameraIndex();
	FovIndex index1(iMosicIndexA, iTrigIndexA, iCamIndexA); 
	
	// Second Fov's information
	unsigned int iMosicIndexB = pOverlap->GetSecondMosaicImage()->Index();
	unsigned int iTrigIndexB = pOverlap->GetSecondTriggerIndex();
	unsigned int iCamIndexB = pOverlap->GetSecondCameraIndex();
	FovIndex index2(iMosicIndexB, iTrigIndexB, iCamIndexB); 
	
	//double* pdRow;

	list<CorrelationPair>* pPairList = pOverlap->GetFinePairListPtr();
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
		_fitInfo[_iCorrelationNum].fovIndexA.IlluminationIndex = iMosicIndexA;
		_fitInfo[_iCorrelationNum].fovIndexA.TriggerIndex = iTrigIndexA;
		_fitInfo[_iCorrelationNum].fovIndexA.CameraIndex = iCamIndexA;
		_fitInfo[_iCorrelationNum].colInMatrixA = (*_pFovOrderMap)[index1] *_iNumParamsPerIndex;
		_fitInfo[_iCorrelationNum].fovIndexB.IlluminationIndex = iMosicIndexB;
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
				 _pSet->GetLayer(iMosicIndexA)->GetImage(iCamIndexA, iTrigIndexA)->GetTransformCamCalibration();
		TransformCamModel camCalB = 
				 _pSet->GetLayer(iMosicIndexB)->GetImage(iCamIndexB, iTrigIndexB)->GetTransformCamCalibration();
		_fitInfo[_iCorrelationNum].xSensorA = 
			htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.S[CAL_ARRAY_X],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].ySensorA = 
			htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.S[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].dxSensordzA = 
			htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.dSdz[CAL_ARRAY_X],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].dySensordzA = 
			htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.dSdz[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].xSensorB = 
			htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.S[CAL_ARRAY_X],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].ySensorB = 
			htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.S[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].dxSensordzB = 
			htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.dSdz[CAL_ARRAY_X],
						_iNumBasisFunctions);
		_fitInfo[_iCorrelationNum].dySensordzB = 
			htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.dSdz[CAL_ARRAY_Y],
						_iNumBasisFunctions);
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
	unsigned int iMosicIndexA= pOverlap->GetMosaicImage()->Index();
	unsigned int iTrigIndexA = pOverlap->GetTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetCameraIndex();
	FovIndex index(iMosicIndexA, iTrigIndexA, iCamIndexA); 
		
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
	_fitInfo[_iCorrelationNum].fovIndexA.IlluminationIndex = iMosicIndexA;
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
				_pSet->GetLayer(iMosicIndexA)->GetImage(iCamIndexA, iTrigIndexA)->GetTransformCamCalibration();
	_fitInfo[_iCorrelationNum].xSensorA = 
		htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.S[CAL_ARRAY_X],
					_iNumBasisFunctions);
	_fitInfo[_iCorrelationNum].ySensorA = 
		htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.S[CAL_ARRAY_Y],
					_iNumBasisFunctions);
	_fitInfo[_iCorrelationNum].dxSensordzA = 
		htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.dSdz[CAL_ARRAY_X],
					_iNumBasisFunctions);
	_fitInfo[_iCorrelationNum].dySensordzA = 
		htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.dSdz[CAL_ARRAY_Y],
					_iNumBasisFunctions);

	pair<double, double> imgAOverlapCenter = pPair->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
	_fitInfo[_iCorrelationNum].boardX = imgAOverlapCenter.first;
	_fitInfo[_iCorrelationNum].boardY = imgAOverlapCenter.second;
	
	_iCurrentRow++;
	_iCurrentRow++;
	_iCorrelationNum++;
	return( true );
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
		if (iCamIndex == 0)  // first camera (logical camera) is always numbered 0
		{
			of << "I_" << k->first.IlluminationIndex 
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
	for(j=0; j<_iNumZTerms; ++j)
	{
		

		double d = _dVectorX[_iMatrixWidth - _iNumZTerms + j];
		of << "," << d;
	}
	of <<  std::endl;
	of.close();
}
#pragma endregion
