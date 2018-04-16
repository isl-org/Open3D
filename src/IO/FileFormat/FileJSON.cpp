// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <IO/ClassIO/IJsonConvertibleIO.h>

#include <fstream>
#include <sstream>
#include <json/json.h>
#include <Core/Utility/Console.h>

namespace three{

namespace {

bool ReadIJsonConvertibleFromJSONStream(std::istream &json_stream,
		IJsonConvertible &object)
{
	Json::Value root_object;
	Json::CharReaderBuilder builder;
	builder["collectComments"] = false;
	JSONCPP_STRING errs;
	bool is_parse_successful = parseFromStream(
				builder, json_stream, &root_object, &errs);
	if (is_parse_successful == false) {
		PrintWarning("Read JSON failed: %s.\n", errs.c_str());
		return false;
	}
	return object.ConvertFromJsonValue(root_object);
}

bool WriteIJsonConvertibleToJSONStream(std::ostream &json_stream,
		const IJsonConvertible &object)
{
	Json::Value root_object;
	if (object.ConvertToJsonValue(root_object) == false) {
		return false;
	}
	Json::StreamWriterBuilder builder;
	builder["commentStyle"] = "None";
	builder["indentation"] = "\t";
	auto writer = builder.newStreamWriter();
	writer->write(root_object, &json_stream);
	return true;
}

}	// unnamed namespace

bool ReadIJsonConvertibleFromJSON(const std::string &filename,
		IJsonConvertible &object)
{
	std::ifstream file_in(filename);
	if (file_in.is_open() == false) {
		PrintWarning("Read JSON failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	bool success = ReadIJsonConvertibleFromJSONStream(file_in, object);
	file_in.close();
	return success;
}

bool WriteIJsonConvertibleToJSON(const std::string &filename,
		const IJsonConvertible &object)
{
	std::ofstream file_out(filename);
	if (file_out.is_open() == false) {
		PrintWarning("Write JSON failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	bool success = WriteIJsonConvertibleToJSONStream(file_out, object);
	file_out.close();
	return success;
}

bool ReadIJsonConvertibleFromJSONString(const std::string &json_string,
		IJsonConvertible &object)
{
	std::istringstream iss(json_string);
	return ReadIJsonConvertibleFromJSONStream(iss, object);
}

bool WriteIJsonConvertibleToJSONString(std::string &json_string,
		const IJsonConvertible &object)
{
	std::ostringstream oss;
	bool success = WriteIJsonConvertibleToJSONStream(oss, object);
	json_string = oss.str();
	return success;
}

}	// namespace three
