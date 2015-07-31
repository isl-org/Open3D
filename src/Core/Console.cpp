#include "Console.h"

#include <cstdio>
#ifdef WIN32
#include <windows.h>
#endif

namespace three{

namespace {

enum {
	TEXT_COLOR_BLACK = 0,
	TEXT_COLOR_RED = 1,
	TEXT_COLOR_GREEN = 2,
	TEXT_COLOR_YELLOW = 3,
	TEXT_COLOR_BLUE = 4,
	TEXT_COLOR_MAGENTA = 5,
	TEXT_COLOR_CYAN = 6,
	TEXT_COLOR_WHITE = 7
};

static VerbosityLevel global_verbosity_level = VERBOSE_INFO;

/// Internal function to change text color for the console
/// Note there is no security check for parameters.
/// \param text_color, from 0 to 7, they are black, red, green, yellow, blue, 
/// magenta, cyan, white
/// \param emphasis_text is 0 or 1
void ChangeConsoleColor(int text_color, int highlight_text)
{
#ifdef WIN32
	const WORD EMPHASIS_MASK[2] = { 0, FOREGROUND_INTENSITY };
	const WORD COLOR_MASK[8] = {
		0,
		FOREGROUND_RED,
		FOREGROUND_GREEN,
		FOREGROUND_GREEN | FOREGROUND_RED,
		FOREGROUND_BLUE,
		FOREGROUND_RED | FOREGROUND_BLUE,
		FOREGROUND_GREEN | FOREGROUND_BLUE,
		FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED
	};
	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(h, 
			EMPHASIS_MASK[highlight_text] | COLOR_MASK[text_color]);
#else
	printf("%c[%d;%dm", 0x1B, highlight_text, text_color + 30);
#endif
}

void ResetConsoleColor()
{
#ifdef WIN32
	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(h, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
#else
	printf("%c[0;m", 0x1B);
#endif
}

}	// unnamed namespace

void SetVerbosityLevel(VerbosityLevel verbosity_level)
{
	global_verbosity_level = verbosity_level;
}

VerbosityLevel GetVerbosityLevel()
{
	return global_verbosity_level;
}

void PrintError(const char *format, ...)
{
	if (global_verbosity_level >= VERBOSE_ERROR) {
		ChangeConsoleColor(TEXT_COLOR_RED, 1);
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		ResetConsoleColor();
	}
}

void PrintWarning(const char *format, ...)
{
	if (global_verbosity_level >= VERBOSE_WARNING) {
		ChangeConsoleColor(TEXT_COLOR_YELLOW, 1);
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		ResetConsoleColor();
	}
}

void PrintInfo(const char *format, ...)
{
	if (global_verbosity_level >= VERBOSE_INFO) {
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
	}
}

void PrintDebug(const char *format, ...)
{
	if (global_verbosity_level >= VERBOSE_DEBUG) {
		ChangeConsoleColor(TEXT_COLOR_GREEN, 0);
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		ResetConsoleColor();
	}
}

void PrintAlways(const char *format, ...)
{
	if (global_verbosity_level >= VERBOSE_ALWAYS) {
		ChangeConsoleColor(TEXT_COLOR_BLUE, 0);
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		ResetConsoleColor();
	}
}

}	// namespace three
