/*
	Wrapper around regoff that allows the caller
	to specify an OverlapRegion to perform SRC
*/

#pragma once

#include "CorrelationPair.h"

bool SqRtCorrelation(CorrelationPair& pCorrPair, unsigned int decimation_factor, bool bEnableNegCorr);