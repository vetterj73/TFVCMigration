#include "RobustSolver.h"

#include "EquationWeights.h"
#include "lsqrpoly.h"
#include "Logger.h"

#include "rtypes.h"
#include "dot2d.h"
extern "C" {
#include "ucxform.h"
}




#pragma region Operateors
bool operator<(const FovIndex& a, const FovIndex& b)
{
	if(a.IlluminationIndex < b.IlluminationIndex)
		return (true);
	else if(a.IlluminationIndex > b.IlluminationIndex)
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
	if(a.IlluminationIndex > b.IlluminationIndex)
		return (true);
	else if(a.IlluminationIndex < b.IlluminationIndex)
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

RobustSolver::RobustSolver(		
	map<FovIndex, unsigned int>* pFovOrderMap)
{
	_pFovOrderMap = pFovOrderMap;
	_bSaveMatrixCSV=true;
	_bVerboseLogging = true;
	_iNumFovs = (unsigned int)pFovOrderMap->size();
	iFileSaveIndex = 0;	
}


RobustSolver::~RobustSolver(void)
{
	/*delete [] _dMatrixA;
	delete [] _dMatrixACopy;
	delete [] _dVectorB;
	delete [] _dVectorBCopy;
	delete [] _dVectorX;*/
}



#pragma endregion

#pragma endregion

#pragma region Solver and transform
struct LeftIndex
{
	unsigned int iLeft;
	unsigned int iIndex;
	unsigned int iLength;
};

bool operator<(const LeftIndex& a, const LeftIndex& b)
{
	return(a.iLeft < b.iLeft);
};

// Reorder the matrix A and vector B, and transpose Matrix A for banded solver
// bRemoveEmptyRows, flag for removing empty equations from Matrix A
// piCounts: output, # rows in block i for banded solver
// piEmptyRows: output, number of empty equations in Matrix A
unsigned int RobustSolver::ReorderAndTranspose(bool bRemoveEmptyRows, int* piCounts, unsigned int* piEmptyRows)
{
	string fileName;
	char cTemp[100];
	if (_bSaveMatrixCSV) 
	{
		// Save Matrix A
		//ofstream of("C:\\Temp\\MatrixA.csv");
		sprintf_s(cTemp, 100, "C:\\Temp\\MatrixA_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		ofstream of(fileName);
	
		for(unsigned int k=0; k<_iMatrixHeight; k++)
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
		of.open(fileName);
		for(unsigned int k=0; k<_iMatrixHeight; k++)
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
			of.open(fileName);
			for(unsigned int k=0; k<_iMatrixHeight; k++)
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
			of.open(fileName);
			for(unsigned int k=0; k<_iMatrixHeight; k++)
			{ 
				of << _pcNotes[k] << std::endl;
			}
			of.close();
		}  
	}
	
	//Get map for reorder
	list<LeftIndex> leftIndexList;
	int iStart, iEnd;
	bool bFlag;
	int iMaxIndex;
	unsigned int iMaxLength = 0;	// Max unempty length in a row 
	*piEmptyRows = 0;				// Number of empty rows
	for(unsigned int row(0); row<_iMatrixHeight; ++row) // for each row
	{
		bFlag = false;  // False if the whole row is zero
		for(unsigned int col(0); col<_iMatrixWidth; ++col)
		{
			if(_dMatrixA[row*_iMatrixWidth+col]!=0)
			{
				if(!bFlag)
				{
					iStart = col; //record the first column
					bFlag = true;
				}

				iEnd = col; // record the last column
			}
		}

		LeftIndex node;				
		node.iIndex = row;
		if(bFlag) // Not a empty row
		{
			node.iLeft = iStart;
			node.iLength = iEnd-iStart+1;
		}
		else
		{	// Empty Rows
			node.iLeft = _iMatrixWidth+1; // Empty rows need put on the last 
			node.iLength = 0;

			(*piEmptyRows)++;
		}
		
		// Update iMaxLength
		if(node.iLength > iMaxLength)
		{
			iMaxLength = node.iLength;
			iMaxIndex = row;
		}
		
		leftIndexList.push_back(node);	
	}
	// Sort node
	leftIndexList.sort();

	// Clear piCount
	for(unsigned int j = 0; j<_iMatrixWidth; j++)
		piCounts[j] = 0;

	// Reorder and transpose
	double* workspace = new double[_iMatrixSize];
	double* dCopyB = new double[_iMatrixHeight];
	unsigned int iDestRow = 0;
	list<LeftIndex>::const_iterator i;
	for(i=leftIndexList.begin(); i!=leftIndexList.end(); i++)
	{
		for(unsigned int col(0); col<_iMatrixWidth; ++col)
			workspace[col*_iMatrixHeight+iDestRow] = _dMatrixA[i->iIndex*_iMatrixWidth+col];

		dCopyB[iDestRow] = _dVectorB[i->iIndex];

		// Skip the empty rows
		if(bRemoveEmptyRows && (i->iLength == 0)) continue;

		// Count Rows of each block 
		if(i->iLeft < _iMatrixWidth - iMaxLength)
			piCounts[i->iLeft]++; // All blocks except the last one
		else
			piCounts[_iMatrixWidth - iMaxLength]++; // The last block
		
		iDestRow++;
	}
	
	unsigned int temp=0;
	for(unsigned int k=0; k<_iMatrixWidth-iMaxLength+1; k++)
		temp += piCounts[k];

	if(bRemoveEmptyRows)
	{
		// Skip the empty rows in input matrix (columns in workspace)
		for(unsigned int k=0; k<_iMatrixWidth; k++)
			::memcpy(_dMatrixA+k*(_iMatrixHeight-*piEmptyRows), workspace+k*_iMatrixHeight, (_iMatrixHeight-*piEmptyRows)*sizeof(double));

	}
	else	// include empty rows
		::memcpy(_dMatrixA, workspace, _iMatrixSize*sizeof(double));

	for(unsigned int k=0; k<_iMatrixHeight; k++)
		_dVectorB[k] = dCopyB[k];

	// for debug
	if (_bSaveMatrixCSV) 
	{
		// Save transposed Matrix A 
		sprintf_s(cTemp, 100, "C:\\Temp\\MatrixA_t_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		ofstream of(fileName);		
		of << std::scientific;

		unsigned int ilines = _iMatrixHeight;
		if(bRemoveEmptyRows)
			ilines = _iMatrixHeight-*piEmptyRows;

		for(unsigned int j=0; j<_iMatrixWidth; j++)
		{
			for(unsigned int k=0; k<ilines; k++)
			{ 
	
				of << _dMatrixA[j*ilines+k];
				if(j != ilines-1)
					of <<",";
			}
			of << std::endl;
		}

		of.close();

		// Save Matrix A
		sprintf_s(cTemp, 100, "C:\\Temp\\MatrixAOrdered_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		of.open(fileName);		
		for(unsigned int k=0; k<ilines; k++)
		{ 
			for(unsigned int j=0; j<_iMatrixWidth; j++)
			{
				of << _dMatrixA[j*ilines+k];
				if(j != _iMatrixWidth-1)
					of <<",";
			}
			of << std::endl;
		}
		of.close();

		// Save Matrix B
		sprintf_s(cTemp, 100, "C:\\Temp\\VectorBOrdered_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		of.open(fileName);		
		for(unsigned int k=0; k<ilines; k++)
		{ 
			of << _dVectorB[k] << std::endl;
		}
		of.close();

		// Save blocklength
		sprintf_s(cTemp, 100, "C:\\Temp\\BlockLength_%d.csv",iFileSaveIndex); 
		fileName.clear();
		fileName.assign(cTemp);
		of.open(fileName);		
		of << _iMatrixWidth << std::endl;
		of << ilines <<std::endl;
		of << iMaxLength <<std::endl;
		for(unsigned int k=0; k<_iMatrixWidth; k++)
		{ 
			of <<  piCounts[k] << std::endl;
		}
		of.close();
	}


	delete [] workspace;
	delete [] dCopyB;

	return(iMaxLength);
}

