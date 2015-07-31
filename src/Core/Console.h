#pragma once

namespace three {
	
enum VerbosityLevel {
	VERBOSE_ERROR = 0,
	VERBOSE_WARNING = 1,
	VERBOSE_INFO = 2,
	VERBOSE_DEBUG = 3,
	VERBOSE_ALWAYS = 4
};

void SetVerbosityLevel(VerbosityLevel verbosity_level);

VerbosityLevel GetVerbosityLevel();

void PrintError(const char *format, ...);

void PrintWarning(const char *format, ...);

void PrintInfo(const char *format, ...);

void PrintDebug(const char *format, ...);

void PrintAlways(const char *format, ...);

}	// namespace three
