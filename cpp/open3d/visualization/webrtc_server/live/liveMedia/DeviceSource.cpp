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
// A template for a MediaSource encapsulating an audio/video input device
//
// NOTE: Sections of this code labeled "%%% TO BE WRITTEN %%%" are incomplete,
// and need to be written by the programmer (depending on the features of the
// particular device). Implementation

#include "DeviceSource.hh"

#include <GroupsockHelper.hh>  // for "gettimeofday()"

DeviceSource* DeviceSource::createNew(UsageEnvironment& env,
                                      DeviceParameters params) {
    return new DeviceSource(env, params);
}

EventTriggerId DeviceSource::eventTriggerId = 0;

unsigned DeviceSource::referenceCount = 0;

DeviceSource::DeviceSource(UsageEnvironment& env, DeviceParameters params)
    : FramedSource(env), fParams(params) {
    if (referenceCount == 0) {
        // Any global initialization of the device would be done here:
        //%%% TO BE WRITTEN %%%
    }
    ++referenceCount;

    // Any instance-specific initialization of the device would be done here:
    //%%% TO BE WRITTEN %%%

    // We arrange here for our "deliverFrame" member function to be called
    // whenever the next frame of data becomes available from the device.
    //
    // If the device can be accessed as a readable socket, then one easy way to
    // do this is using a call to
    //     envir().taskScheduler().turnOnBackgroundReadHandling( ... )
    // (See examples of this call in the "liveMedia" directory.)
    //
    // If, however, the device *cannot* be accessed as a readable socket, then
    // instead we can implement it using 'event triggers': Create an 'event
    // trigger' for this device (if it hasn't already been done):
    if (eventTriggerId == 0) {
        eventTriggerId =
                envir().taskScheduler().createEventTrigger(deliverFrame0);
    }
}

DeviceSource::~DeviceSource() {
    // Any instance-specific 'destruction' (i.e., resetting) of the device would
    // be done here:
    //%%% TO BE WRITTEN %%%

    --referenceCount;
    if (referenceCount == 0) {
        // Any global 'destruction' (i.e., resetting) of the device would be
        // done here:
        //%%% TO BE WRITTEN %%%

        // Reclaim our 'event trigger'
        envir().taskScheduler().deleteEventTrigger(eventTriggerId);
        eventTriggerId = 0;
    }
}

void DeviceSource::doGetNextFrame() {
    // This function is called (by our 'downstream' object) when it asks for new
    // data.

    // Note: If, for some reason, the source device stops being readable (e.g.,
    // it gets closed), then you do the following:
    if (0 /* the source stops being readable */ /*%%% TO BE WRITTEN %%%*/) {
        handleClosure();
        return;
    }

    // If a new frame of data is immediately available to be delivered, then do
    // this now:
    if (0 /* a new frame of data is immediately available to be delivered*/
        /*%%% TO BE WRITTEN %%%*/) {
        deliverFrame();
    }

    // No new data is immediately available to be delivered.  We don't do
    // anything more here. Instead, our event trigger must be called (e.g., from
    // a separate thread) when new data becomes available.
}

void DeviceSource::deliverFrame0(void* clientData) {
    ((DeviceSource*)clientData)->deliverFrame();
}

void DeviceSource::deliverFrame() {
    // This function is called when new frame data is available from the device.
    // We deliver this data by copying it to the 'downstream' object, using the
    // following parameters (class members): 'in' parameters (these should *not*
    // be modified by this function):
    //     fTo: The frame data is copied to this address.
    //         (Note that the variable "fTo" is *not* modified.  Instead,
    //          the frame data is copied to the address pointed to by "fTo".)
    //     fMaxSize: This is the maximum number of bytes that can be copied
    //         (If the actual frame is larger than this, then it should
    //          be truncated, and "fNumTruncatedBytes" set accordingly.)
    // 'out' parameters (these are modified by this function):
    //     fFrameSize: Should be set to the delivered frame size (<= fMaxSize).
    //     fNumTruncatedBytes: Should be set iff the delivered frame would have
    //     been
    //         bigger than "fMaxSize", in which case it's set to the number of
    //         bytes that have been omitted.
    //     fPresentationTime: Should be set to the frame's presentation time
    //         (seconds, microseconds).  This time must be aligned with
    //         'wall-clock time' - i.e., the time that you would get by calling
    //         "gettimeofday()".
    //     fDurationInMicroseconds: Should be set to the frame's duration, if
    //     known.
    //         If, however, the device is a 'live source' (e.g., encoded from a
    //         camera or microphone), then we probably don't need to set this
    //         variable, because - in this case - data will never arrive
    //         'early'.
    // Note the code below.

    if (!isCurrentlyAwaitingData()) return;  // we're not ready for the data yet

    u_int8_t* newFrameDataStart =
            (u_int8_t*)0xDEADBEEF;  //%%% TO BE WRITTEN %%%
    unsigned newFrameSize = 0;      //%%% TO BE WRITTEN %%%

    // Deliver the data here:
    if (newFrameSize > fMaxSize) {
        fFrameSize = fMaxSize;
        fNumTruncatedBytes = newFrameSize - fMaxSize;
    } else {
        fFrameSize = newFrameSize;
    }
    gettimeofday(&fPresentationTime,
                 NULL);  // If you have a more accurate time - e.g., from an
                         // encoder - then use that instead.
    // If the device is *not* a 'live source' (e.g., it comes instead from a
    // file or buffer), then set "fDurationInMicroseconds" here.
    memmove(fTo, newFrameDataStart, fFrameSize);

    // After delivering the data, inform the reader that it is now available:
    FramedSource::afterGetting(this);
}

// The following code would be called to signal that a new frame of data has
// become available. This (unlike other "LIVE555 Streaming Media" library code)
// may be called from a separate thread. (Note, however, that "triggerEvent()"
// cannot be called with the same 'event trigger id' from different threads.
// Also, if you want to have multiple device threads, each one using a different
// 'event trigger id', then you will need to make "eventTriggerId" a non-static
// member variable of "DeviceSource".)
void signalNewFrameData() {
    TaskScheduler* ourScheduler = NULL;  //%%% TO BE WRITTEN %%%
    DeviceSource* ourDevice = NULL;      //%%% TO BE WRITTEN %%%

    if (ourScheduler != NULL) {  // sanity check
        ourScheduler->triggerEvent(DeviceSource::eventTriggerId, ourDevice);
    }
}
