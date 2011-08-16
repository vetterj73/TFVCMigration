#include "FiducialResult.h"
#include "Logger.h"
#include "EquationWeights.h"

///////////////////////////////////////////////////////////////
//////		FiducialResults Class
///////////////////////////////////////////////////////////////
FiducialResults::FiducialResults(void)
{
	_pFeature = NULL;
	_fidFovOverlapPointList.clear();
}

FiducialResults::~FiducialResults(void)
{
}

void FiducialResults::LogResults()
{
	// Validation check
	if(_pFeature != NULL)
	{
		LOG.FireLogEntry(LogTypeDiagnostic, "Fiducial ID(%d); location(%d, %d)mm", 
			_pFeature->GetId(), (int)(_pFeature->GetCadX()*1000), (int)(_pFeature->GetCadY()*1000));
	}

	for(list<FidFovOverlap*>::iterator i = _fidFovOverlapPointList.begin(); i != _fidFovOverlapPointList.end(); i++)
	{
		if((*i)->GetCoarsePair()->IsProcessed())
		{
			CorrelationResult result = (*i)->GetCoarsePair()->GetCorrelationResult();
			int iWeight = (int)(Weights.CalWeight((*i)->GetCoarsePair()));
			LOG.FireLogEntry(LogTypeDiagnostic, "	Illumintion(%d), Trigger(%d), Camera(%d): CorrScore(%d), Ambig(%d), offset(%d,%d)pixels, weight(%d)",
				(*i)->GetMosaicImage()->Index(), (*i)->GetTriggerIndex(), (*i)->GetCameraIndex(),
				(int)(result.CorrCoeff*100), (int)(result.AmbigScore*100), (int)(result.ColOffset), (int)(result.RowOffset),
				(int)(Weights.CalWeight((*i)->GetCoarsePair())));
		}
	}

}

double FiducialResults::CalConfidence()
{
	double dConfidenceScore = 0;

	// Pick the higheset one for each physical fiducial
	for(list<FidFovOverlap*>::iterator i = _fidFovOverlapPointList.begin(); i != _fidFovOverlapPointList.end(); i++)
	{
		if((*i)->GetCoarsePair()->IsProcessed())
		{
			CorrelationResult result = (*i)->GetCoarsePair()->GetCorrelationResult();
			double dScore = result.CorrCoeff*(1-result.AmbigScore);
			if(Weights.CalWeight((*i)->GetCoarsePair())<0)
				dScore = 0;

			if(dConfidenceScore < dScore)
				dConfidenceScore = dScore;
		}
	}

	return(dConfidenceScore);
}

///////////////////////////////////////////////////////////////
//////		FiducialResultsSet Class
///////////////////////////////////////////////////////////////
FiducialResultsSet::FiducialResultsSet(unsigned int iSize)
{
	_pResultSet = NULL;

	_iSize = iSize;
	if(iSize=0) 
	{
		LOG.FireLogEntry(LogTypeError, "FiducialResultsSet: The input size should not be 0");
		return;
	}

	_pResultSet = new FiducialResults[_iSize];
}

FiducialResultsSet::~FiducialResultsSet()
{
	if(_pResultSet != NULL)
		delete [] _pResultSet;
}

void FiducialResultsSet::LogResults()
{
	for(int i=0; i<_iSize; i++)
		_pResultSet[i].LogResults();

	double dConfidence = CalConfidence();
	LOG.FireLogEntry(LogTypeDiagnostic, "Confidence = %d/100", (int)(dConfidence*100));
}

// Calculate confidence based on the fid
double FiducialResultsSet::CalConfidence()
{
	// Calculate confidence for each physical fiducial
	list<double> dConfidenceList;
	for(int i=0; i<_iSize; i++)
	{
		double dConfidence = _pResultSet[i].CalConfidence();
		dConfidenceList.push_back(dConfidence);
	}
	dConfidenceList.sort();

	// Decide overall confidence
	switch(_iSize)
	{
	case 0:
	case 1:
		return(0);	// No confidence at all
	break;
	
	case 2:
	case 3:
		return(*dConfidenceList.begin()); // The smallest one
	break;

	default:	// 4 or more
		int iCount = 0;
		for(list<double>::iterator i = dConfidenceList.begin(); i!=dConfidenceList.end(); i++)
		{
			if(iCount == _iSize-3) // The third biggest one
				return(*i);
			
			iCount++;
		}
	}

	return(0); // should never reach here
}
