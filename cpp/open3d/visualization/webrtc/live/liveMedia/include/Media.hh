/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// "liveMedia"
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// Medium
// C++ header

#ifndef _MEDIA_HH
#define _MEDIA_HH

#ifndef _LIVEMEDIA_VERSION_HH
#include "liveMedia_version.hh"
#endif

#ifndef _HASH_TABLE_HH
#include "HashTable.hh"
#endif

#ifndef _USAGE_ENVIRONMENT_HH
#include "UsageEnvironment.hh"
#endif

// Lots of files end up needing the following, so just #include them here:
#ifndef _NET_COMMON_H
#include "NetCommon.h"
#endif
#include <stdio.h>

// The following makes the Borland compiler happy:
#ifdef __BORLANDC__
#define _strnicmp strnicmp
#define fabsf(x) fabs(x)
#endif

#define mediumNameMaxLen 30

class Medium {
public:
  static Boolean lookupByName(UsageEnvironment& env,
			      char const* mediumName,
			      Medium*& resultMedium);
  static void close(UsageEnvironment& env, char const* mediumName);
  static void close(Medium* medium); // alternative close() method using ptrs
      // (has no effect if medium == NULL)

  UsageEnvironment& envir() const {return fEnviron;}

  char const* name() const {return fMediumName;}

  // Test for specific types of media:
  virtual Boolean isSource() const;
  virtual Boolean isSink() const;
  virtual Boolean isRTCPInstance() const;
  virtual Boolean isRTSPClient() const;
  virtual Boolean isRTSPServer() const;
  virtual Boolean isMediaSession() const;
  virtual Boolean isServerMediaSession() const;

protected:
  friend class MediaLookupTable;
  Medium(UsageEnvironment& env); // abstract base class
  virtual ~Medium(); // instances are deleted using close() only

  TaskToken& nextTask() {
	return fNextTask;
  }

private:
  UsageEnvironment& fEnviron;
  char fMediumName[mediumNameMaxLen];
  TaskToken fNextTask;
};


// A data structure for looking up a Medium by its string name.
// (It is used only to implement "Medium", but we make it visible here, in case developers want to use it to iterate over
//  the whole set of "Medium" objects that we've created.)
class MediaLookupTable {
public:
  static MediaLookupTable* ourMedia(UsageEnvironment& env);
  HashTable const& getTable() { return *fTable; }

protected:
  MediaLookupTable(UsageEnvironment& env);
  virtual ~MediaLookupTable();

private:
  friend class Medium;

  Medium* lookup(char const* name) const;
  // Returns NULL if none already exists

  void addNew(Medium* medium, char* mediumName);
  void remove(char const* name);

  void generateNewName(char* mediumName, unsigned maxLen);

private:
  UsageEnvironment& fEnv;
  HashTable* fTable;
  unsigned fNameGenerator;
};


// The structure pointed to by the "liveMediaPriv" UsageEnvironment field:
class _Tables {
public:
  static _Tables* getOurTables(UsageEnvironment& env, Boolean createIfNotPresent = True);
      // returns a pointer to a "_Tables" structure (creating it if necessary)
  void reclaimIfPossible();
      // used to delete ourselves when we're no longer used

  MediaLookupTable* mediaTable;
  void* socketTable;

protected:
  _Tables(UsageEnvironment& env);
  virtual ~_Tables();

private:
  UsageEnvironment& fEnv;
};

#endif
