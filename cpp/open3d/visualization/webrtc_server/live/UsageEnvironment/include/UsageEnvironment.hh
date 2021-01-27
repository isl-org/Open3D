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
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// Usage Environment
// C++ header

#ifndef _USAGE_ENVIRONMENT_HH
#define _USAGE_ENVIRONMENT_HH

#ifndef _USAGEENVIRONMENT_VERSION_HH
#include "UsageEnvironment_version.hh"
#endif

#ifndef _NETCOMMON_H
#include "NetCommon.h"
#endif

#ifndef _BOOLEAN_HH
#include "Boolean.hh"
#endif

#ifndef _STRDUP_HH
// "strDup()" is used often, so include this here, so everyone gets it:
#include "strDup.hh"
#endif

#ifndef NULL
#define NULL 0
#endif

#ifdef __BORLANDC__
#define _setmode setmode
#define _O_BINARY O_BINARY
#endif

class TaskScheduler; // forward

// An abstract base class, subclassed for each use of the library

class UsageEnvironment {
public:
  Boolean reclaim();
      // returns True iff we were actually able to delete our object

  // task scheduler:
  TaskScheduler& taskScheduler() const {return fScheduler;}

  // result message handling:
  typedef char const* MsgString;
  virtual MsgString getResultMsg() const = 0;

  virtual void setResultMsg(MsgString msg) = 0;
  virtual void setResultMsg(MsgString msg1, MsgString msg2) = 0;
  virtual void setResultMsg(MsgString msg1, MsgString msg2, MsgString msg3) = 0;
  virtual void setResultErrMsg(MsgString msg, int err = 0) = 0;
	// like setResultMsg(), except that an 'errno' message is appended.  (If "err == 0", the "getErrno()" code is used instead.)

  virtual void appendToResultMsg(MsgString msg) = 0;

  virtual void reportBackgroundError() = 0;
	// used to report a (previously set) error message within
	// a background event

  virtual void internalError(); // used to 'handle' a 'should not occur'-type error condition within the library.

  // 'errno'
  virtual int getErrno() const = 0;

  // 'console' output:
  virtual UsageEnvironment& operator<<(char const* str) = 0;
  virtual UsageEnvironment& operator<<(int i) = 0;
  virtual UsageEnvironment& operator<<(unsigned u) = 0;
  virtual UsageEnvironment& operator<<(double d) = 0;
  virtual UsageEnvironment& operator<<(void* p) = 0;

  // a pointer to additional, optional, client-specific state
  void* liveMediaPriv;
  void* groupsockPriv;

protected:
  UsageEnvironment(TaskScheduler& scheduler); // abstract base class
  virtual ~UsageEnvironment(); // we are deleted only by reclaim()

private:
  TaskScheduler& fScheduler;
};


typedef void TaskFunc(void* clientData);
typedef void* TaskToken;
typedef u_int32_t EventTriggerId;

class TaskScheduler {
public:
  virtual ~TaskScheduler();

  virtual TaskToken scheduleDelayedTask(int64_t microseconds, TaskFunc* proc,
					void* clientData) = 0;
	// Schedules a task to occur (after a delay) when we next
	// reach a scheduling point.
	// (Does not delay if "microseconds" <= 0)
	// Returns a token that can be used in a subsequent call to
	// unscheduleDelayedTask() or rescheduleDelayedTask()
        // (but only if the task has not yet occurred).

  virtual void unscheduleDelayedTask(TaskToken& prevTask) = 0;
	// (Has no effect if "prevTask" == NULL)
        // Sets "prevTask" to NULL afterwards.
        // Note: This MUST NOT be called if the scheduled task has already occurred.

  virtual void rescheduleDelayedTask(TaskToken& task,
				     int64_t microseconds, TaskFunc* proc,
				     void* clientData);
        // Combines "unscheduleDelayedTask()" with "scheduleDelayedTask()"
        // (setting "task" to the new task token).
        // Note: This MUST NOT be called if the scheduled task has already occurred.

  // For handling socket operations in the background (from the event loop):
  typedef void BackgroundHandlerProc(void* clientData, int mask);
    // Possible bits to set in "mask".  (These are deliberately defined
    // the same as those in Tcl, to make a Tcl-based subclass easy.)
    #define SOCKET_READABLE    (1<<1)
    #define SOCKET_WRITABLE    (1<<2)
    #define SOCKET_EXCEPTION   (1<<3)
  virtual void setBackgroundHandling(int socketNum, int conditionSet, BackgroundHandlerProc* handlerProc, void* clientData) = 0;
  void disableBackgroundHandling(int socketNum) { setBackgroundHandling(socketNum, 0, NULL, NULL); }
  virtual void moveSocketHandling(int oldSocketNum, int newSocketNum) = 0;
        // Changes any socket handling for "oldSocketNum" so that occurs with "newSocketNum" instead.

  virtual void doEventLoop(char volatile* watchVariable = NULL) = 0;
      // Causes further execution to take place within the event loop.
      // Delayed tasks, background I/O handling, and other events are handled, sequentially (as a single thread of control).
      // (If "watchVariable" is not NULL, then we return from this routine when *watchVariable != 0)

  virtual EventTriggerId createEventTrigger(TaskFunc* eventHandlerProc) = 0;
      // Creates a 'trigger' for an event, which - if it occurs - will be handled (from the event loop) using "eventHandlerProc".
      // (Returns 0 iff no such trigger can be created (e.g., because of implementation limits on the number of triggers).)
  virtual void deleteEventTrigger(EventTriggerId eventTriggerId) = 0;

  virtual void triggerEvent(EventTriggerId eventTriggerId, void* clientData = NULL) = 0;
      // Causes the (previously-registered) handler function for the specified event to be handled (from the event loop).
      // The handler function is called with "clientData" as parameter.
      // Note: This function (unlike other library functions) may be called from an external thread
      // - to signal an external event.  (However, "triggerEvent()" should not be called with the
      // same 'event trigger id' from different threads.)

  // The following two functions are deprecated, and are provided for backwards-compatibility only:
  void turnOnBackgroundReadHandling(int socketNum, BackgroundHandlerProc* handlerProc, void* clientData) {
    setBackgroundHandling(socketNum, SOCKET_READABLE, handlerProc, clientData);
  }
  void turnOffBackgroundReadHandling(int socketNum) { disableBackgroundHandling(socketNum); }

  virtual void internalError(); // used to 'handle' a 'should not occur'-type error condition within the library.

protected:
  TaskScheduler(); // abstract base class
};

#endif
