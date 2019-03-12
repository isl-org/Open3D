
#pragma once

#include <string>

bool AlocateHstMemory(float** h, const int& numElements, const std::string& name);
bool AlocateDevMemory(float** d, const int& numElements, const std::string& name);
void RandInit(float* h, const int& numElements);
bool CopyHst2DevMemory(float* h, float* d, const int& numElements);
bool CopyDev2HstMemory(float* d, float* h, const int& numElements);
bool freeDev(float** d, const std::string& name);
