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
// Basic Usage Environment: for a simple, non-scripted, console application
// Implementation

#include "BasicUsageEnvironment0.hh"
#include "HandlerSet.hh"

////////// A subclass of DelayQueueEntry,
//////////     used to implement BasicTaskScheduler0::scheduleDelayedTask()

class AlarmHandler : public DelayQueueEntry {
public:
    AlarmHandler(TaskFunc* proc, void* clientData, DelayInterval timeToDelay)
        : DelayQueueEntry(timeToDelay), fProc(proc), fClientData(clientData) {}

private:  // redefined virtual functions
    virtual void handleTimeout() {
        (*fProc)(fClientData);
        DelayQueueEntry::handleTimeout();
    }

private:
    TaskFunc* fProc;
    void* fClientData;
};

////////// BasicTaskScheduler0 //////////

BasicTaskScheduler0::BasicTaskScheduler0()
    : fLastHandledSocketNum(-1),
      fTriggersAwaitingHandling(0),
      fLastUsedTriggerMask(1),
      fLastUsedTriggerNum(MAX_NUM_EVENT_TRIGGERS - 1) {
    fHandlers = new HandlerSet;
    for (unsigned i = 0; i < MAX_NUM_EVENT_TRIGGERS; ++i) {
        fTriggeredEventHandlers[i] = NULL;
        fTriggeredEventClientDatas[i] = NULL;
    }
}

BasicTaskScheduler0::~BasicTaskScheduler0() { delete fHandlers; }

TaskToken BasicTaskScheduler0::scheduleDelayedTask(int64_t microseconds,
                                                   TaskFunc* proc,
                                                   void* clientData) {
    if (microseconds < 0) microseconds = 0;
    DelayInterval timeToDelay((long)(microseconds / 1000000),
                              (long)(microseconds % 1000000));
    AlarmHandler* alarmHandler =
            new AlarmHandler(proc, clientData, timeToDelay);
    fDelayQueue.addEntry(alarmHandler);

    return (void*)(alarmHandler->token());
}

void BasicTaskScheduler0::unscheduleDelayedTask(TaskToken& prevTask) {
    DelayQueueEntry* alarmHandler = fDelayQueue.removeEntry((intptr_t)prevTask);
    prevTask = NULL;
    delete alarmHandler;
}

void BasicTaskScheduler0::doEventLoop(char volatile* watchVariable) {
    // Repeatedly loop, handling readble sockets and timed events:
    while (1) {
        if (watchVariable != NULL && *watchVariable != 0) break;
        SingleStep();
    }
}

EventTriggerId BasicTaskScheduler0::createEventTrigger(
        TaskFunc* eventHandlerProc) {
    unsigned i = fLastUsedTriggerNum;
    EventTriggerId mask = fLastUsedTriggerMask;

    do {
        i = (i + 1) % MAX_NUM_EVENT_TRIGGERS;
        mask >>= 1;
        if (mask == 0) mask = 0x80000000;

        if (fTriggeredEventHandlers[i] == NULL) {
            // This trigger number is free; use it:
            fTriggeredEventHandlers[i] = eventHandlerProc;
            fTriggeredEventClientDatas[i] = NULL;  // sanity

            fLastUsedTriggerMask = mask;
            fLastUsedTriggerNum = i;

            return mask;
        }
    } while (i != fLastUsedTriggerNum);

    // All available event triggers are allocated; return 0 instead:
    return 0;
}

void BasicTaskScheduler0::deleteEventTrigger(EventTriggerId eventTriggerId) {
    fTriggersAwaitingHandling &= ~eventTriggerId;

    if (eventTriggerId == fLastUsedTriggerMask) {  // common-case optimization:
        fTriggeredEventHandlers[fLastUsedTriggerNum] = NULL;
        fTriggeredEventClientDatas[fLastUsedTriggerNum] = NULL;
    } else {
        // "eventTriggerId" should have just one bit set.
        // However, we do the reasonable thing if the user happened to 'or'
        // together two or more "EventTriggerId"s:
        EventTriggerId mask = 0x80000000;
        for (unsigned i = 0; i < MAX_NUM_EVENT_TRIGGERS; ++i) {
            if ((eventTriggerId & mask) != 0) {
                fTriggeredEventHandlers[i] = NULL;
                fTriggeredEventClientDatas[i] = NULL;
            }
            mask >>= 1;
        }
    }
}

void BasicTaskScheduler0::triggerEvent(EventTriggerId eventTriggerId,
                                       void* clientData) {
    // First, record the "clientData".  (Note that we allow "eventTriggerId" to
    // be a combination of bits for multiple events.)
    EventTriggerId mask = 0x80000000;
    for (unsigned i = 0; i < MAX_NUM_EVENT_TRIGGERS; ++i) {
        if ((eventTriggerId & mask) != 0) {
            fTriggeredEventClientDatas[i] = clientData;
        }
        mask >>= 1;
    }

    // Then, note this event as being ready to be handled.
    // (Note that because this function (unlike others in the library) can be
    // called from an external thread, we do this last, to
    //  reduce the risk of a race condition.)
    fTriggersAwaitingHandling |= eventTriggerId;
}

////////// HandlerSet (etc.) implementation //////////

HandlerDescriptor::HandlerDescriptor(HandlerDescriptor* nextHandler)
    : conditionSet(0), handlerProc(NULL) {
    // Link this descriptor into a doubly-linked list:
    if (nextHandler == this) {  // initialization
        fNextHandler = fPrevHandler = this;
    } else {
        fNextHandler = nextHandler;
        fPrevHandler = nextHandler->fPrevHandler;
        nextHandler->fPrevHandler = this;
        fPrevHandler->fNextHandler = this;
    }
}

HandlerDescriptor::~HandlerDescriptor() {
    // Unlink this descriptor from a doubly-linked list:
    fNextHandler->fPrevHandler = fPrevHandler;
    fPrevHandler->fNextHandler = fNextHandler;
}

HandlerSet::HandlerSet() : fHandlers(&fHandlers) {
    fHandlers.socketNum = -1;  // shouldn't ever get looked at, but in case...
}

HandlerSet::~HandlerSet() {
    // Delete each handler descriptor:
    while (fHandlers.fNextHandler != &fHandlers) {
        delete fHandlers.fNextHandler;  // changes fHandlers->fNextHandler
    }
}

void HandlerSet ::assignHandler(
        int socketNum,
        int conditionSet,
        TaskScheduler::BackgroundHandlerProc* handlerProc,
        void* clientData) {
    // First, see if there's already a handler for this socket:
    HandlerDescriptor* handler = lookupHandler(socketNum);
    if (handler == NULL) {  // No existing handler, so create a new descr:
        handler = new HandlerDescriptor(fHandlers.fNextHandler);
        handler->socketNum = socketNum;
    }

    handler->conditionSet = conditionSet;
    handler->handlerProc = handlerProc;
    handler->clientData = clientData;
}

void HandlerSet::clearHandler(int socketNum) {
    HandlerDescriptor* handler = lookupHandler(socketNum);
    delete handler;
}

void HandlerSet::moveHandler(int oldSocketNum, int newSocketNum) {
    HandlerDescriptor* handler = lookupHandler(oldSocketNum);
    if (handler != NULL) {
        handler->socketNum = newSocketNum;
    }
}

HandlerDescriptor* HandlerSet::lookupHandler(int socketNum) {
    HandlerDescriptor* handler;
    HandlerIterator iter(*this);
    while ((handler = iter.next()) != NULL) {
        if (handler->socketNum == socketNum) break;
    }
    return handler;
}

HandlerIterator::HandlerIterator(HandlerSet& handlerSet) : fOurSet(handlerSet) {
    reset();
}

HandlerIterator::~HandlerIterator() {}

void HandlerIterator::reset() { fNextPtr = fOurSet.fHandlers.fNextHandler; }

HandlerDescriptor* HandlerIterator::next() {
    HandlerDescriptor* result = fNextPtr;
    if (result == &fOurSet.fHandlers) {  // no more
        result = NULL;
    } else {
        fNextPtr = fNextPtr->fNextHandler;
    }

    return result;
}
