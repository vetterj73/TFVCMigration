#include "RobustSolver.h"
#include "EquationWeights.h"
#include "lsqrpoly.h"
#include "Logger.h"
#include "CorrelationParameters.h"

#include "rtypes.h"
#include "dot2d.h"
extern "C" {
#include "ucxform.h"
}
#include "coord.h"

#pragma region Operateors
bool operator<(const FovIndex& a, const FovIndex& b)
{
	if(a.LayerIndex < b.LayerIndex)
		return (true);
	else if(a.LayerIndex > b.LayerIndex)
		return (false);
	else
	{
		if(a.TriggerIndex < b.TriggerIndex)
			return(true);
		else if(a.TriggerIndex > b.TriggerIndex)
			return(false);
		else
		{ 
			if(a.CameraIndex < b.CameraIndex)
				return(true);
			else if(a.CameraIndex > b.CameraIndex)
				return(false);
		}
	}

	return(false);
}

bool operator>(const FovIndex& a, const FovIndex& b)
{
	if(a.LayerIndex > b.LayerIndex)
		return (true);
	else if(a.LayerIndex < b.LayerIndex)
		return(false);
	else
	{
		if(a.TriggerIndex > b.TriggerIndex)
			return(true);
		else if(a.TriggerIndex < b.TriggerIndex)
			return(false);
		else
		{
			if(a.CameraIndex > b.CameraIndex)
				return(true);
			else if(a.CameraIndex < b.CameraIndex)
				return(false);
		}
	}
			
	return(false);
}
#pragma endregion

#pragma region constructor
/// <summary>
/// Constructor for RobustSolver -- a base class for FOV, CM, and Iterative solvers
/// </summary>
/// <param name="pFovOrderMap"></param>

RobustSolver::RobustSolver(		
	map<FovIndex, unsigned int>* pFovOrderMap)
{
	_pFovOrderMap = pFovOrderMap;
	_bSaveMatrixCSV = false;
	_bVerboseLogging = false;
	_iNumFovs = (unsigned int)pFovOrderMap->size();
	iFileSaveIndex = 0;	
}

RobustSolver::~RobustSolver(void)
{
	delete [] _dMatrixA;
	delete [] _dMatrixACopy;
	delete [] _dVectorB;
	delete [] _dVectorBCopy;
	delete [] _dVectorX;
}

//initialize all coefficients in the system of equations to zero 
void RobustSolver::ZeroTheSystem()
{
	_iCurrentRow = 0;

	unsigned int i;
	for(i=0; i<_iMatrixSize; i++)
		_dMatrixA[i] = 0.0;

	for(i=0; i<_iMatrixHeight; i++)
	{
		_dVectorB[i] = 0.0;
	}

	for(i =0; i<_iMatrixWidth; i++)
		_dVectorX[i] = 0.0;
}

#pragma endregion




