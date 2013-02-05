#include "Logger.h"

// Singleton pattern
Logger* Logger::_pInst = NULL;
Logger& Logger::Instance()
{
	if(_pInst == NULL)
		_pInst = new Logger();

	return(*_pInst);
}

Logger::Logger(void)
{
}

Logger::~Logger(void)
{
}