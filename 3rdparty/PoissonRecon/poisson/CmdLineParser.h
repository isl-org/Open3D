/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef CMD_LINE_PARSER_INCLUDED
#define CMD_LINE_PARSER_INCLUDED

#include <stdarg.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "MyMiscellany.h"

#ifdef WIN32
int strcasecmp(const char* c1, const char* c2);
#endif  // WIN32

class cmdLineReadable {
public:
    bool set;
    char* name;
    cmdLineReadable(const char* name);
    virtual ~cmdLineReadable(void);
    virtual int read(char** argv, int argc);
    virtual void writeValue(char* str) const;
};

template <class Type>
void cmdLineWriteValue(Type t, char* str);
template <class Type>
void cmdLineCleanUp(Type* t);
template <class Type>
Type cmdLineInitialize(void);
template <class Type>
Type cmdLineCopy(Type t);
template <class Type>
Type cmdLineStringToType(const char* str);

template <class Type>
class cmdLineParameter : public cmdLineReadable {
public:
    Type value;
    cmdLineParameter(const char* name);
    cmdLineParameter(const char* name, Type v);
    ~cmdLineParameter(void);
    int read(char** argv, int argc);
    void writeValue(char* str) const;
    bool expectsArg(void) const { return true; }
};

template <class Type, int Dim>
class cmdLineParameterArray : public cmdLineReadable {
public:
    Type values[Dim];
    cmdLineParameterArray(const char* name, const Type* v = NULL);
    ~cmdLineParameterArray(void);
    int read(char** argv, int argc);
    void writeValue(char* str) const;
    bool expectsArg(void) const { return true; }
};

template <class Type>
class cmdLineParameters : public cmdLineReadable {
public:
    int count;
    Type* values;
    cmdLineParameters(const char* name);
    ~cmdLineParameters(void);
    int read(char** argv, int argc);
    void writeValue(char* str) const;
    bool expectsArg(void) const { return true; }
};

void cmdLineParse(int argc, char** argv, cmdLineReadable** params);
char* FileExtension(char* fileName);
char* LocalFileName(char* fileName);
char* DirectoryName(char* fileName);
char* GetFileExtension(const char* fileName);
char* GetLocalFileName(const char* fileName);
char** ReadWords(const char* fileName, int& cnt);

#include "CmdLineParser.inl"
#endif  // CMD_LINE_PARSER_INCLUDED
