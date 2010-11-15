/*
	This is a singleton wrapper up of LoggableObject
*/

#pragma once

#include "LoggableObject.h"
using namespace LOGGER;

class Logger : public LoggableObject
{
public:
	static Logger& Instance();

protected:
	Logger(void);
	~Logger(void);

private:
	static Logger* _pInst;
};

