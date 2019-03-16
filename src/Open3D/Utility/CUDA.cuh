
#pragma once

#include <cuda_runtime.h>

#include <string>

void DeviceInfo(const int& devID);
bool AlocateHstMemory(double** h, const int& numElements, const std::string& name);
bool AlocateDevMemory(double** d, const int& numElements, const std::string& name);
void RandInit(double* h, const int& numElements);
bool CopyHst2DevMemory(double* h, double* d, const int& numElements);
bool CopyDev2HstMemory(double* d, double* h, const int& numElements);
bool freeDev(double** d, const std::string& name);
