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
// An class that can be used to create (possibly multiple) 'replicas' of an
// incoming stream. Implementation.

#include "StreamReplicator.hh"

////////// Definition of "StreamReplica": The class that implements each stream
/// replica //////////

class StreamReplica : public FramedSource {
protected:
    friend class StreamReplicator;
    StreamReplica(
            StreamReplicator&
                    ourReplicator);  // called only by
                                     // "StreamReplicator::createStreamReplica()"
    virtual ~StreamReplica();

private:  // redefined virtual functions:
    virtual void doGetNextFrame();
    virtual void doStopGettingFrames();

private:
    static void copyReceivedFrame(StreamReplica* toReplica,
                                  StreamReplica* fromReplica);

private:
    StreamReplicator& fOurReplicator;
    int fFrameIndex;  // 0 or 1, depending upon which frame we're currently
                      // requesting; could also be -1 if we've stopped playing

    // Replicas that are currently awaiting data are kept in a (singly-linked)
    // list:
    StreamReplica* fNext;
};

////////// StreamReplicator implementation //////////

StreamReplicator* StreamReplicator::createNew(
        UsageEnvironment& env,
        FramedSource* inputSource,
        Boolean deleteWhenLastReplicaDies) {
    return new StreamReplicator(env, inputSource, deleteWhenLastReplicaDies);
}

StreamReplicator::StreamReplicator(UsageEnvironment& env,
                                   FramedSource* inputSource,
                                   Boolean deleteWhenLastReplicaDies)
    : Medium(env),
      fInputSource(inputSource),
      fDeleteWhenLastReplicaDies(deleteWhenLastReplicaDies),
      fInputSourceHasClosed(False),
      fNumReplicas(0),
      fNumActiveReplicas(0),
      fNumDeliveriesMadeSoFar(0),
      fFrameIndex(0),
      fMasterReplica(NULL),
      fReplicasAwaitingCurrentFrame(NULL),
      fReplicasAwaitingNextFrame(NULL) {}

StreamReplicator::~StreamReplicator() { Medium::close(fInputSource); }

FramedSource* StreamReplicator::createStreamReplica() {
    ++fNumReplicas;
    return new StreamReplica(*this);
}

void StreamReplicator::getNextFrame(StreamReplica* replica) {
    if (fInputSourceHasClosed) {  // handle closure instead
        replica->handleClosure();
        return;
    }

    if (replica->fFrameIndex == -1) {
        // This replica had stopped playing (or had just been created), but is
        // now actively reading.  Note this:
        replica->fFrameIndex = fFrameIndex;
        ++fNumActiveReplicas;
    }

    if (fMasterReplica == NULL) {
        // This is the first replica to request the next unread frame.  Make it
        // the 'master' replica - meaning that we read the frame into its
        // buffer, and then copy from this into the other replicas' buffers.
        fMasterReplica = replica;

        // Arrange to read the next frame into this replica's buffer:
        if (fInputSource != NULL)
            fInputSource->getNextFrame(
                    fMasterReplica->fTo, fMasterReplica->fMaxSize,
                    afterGettingFrame, this, onSourceClosure, this);
    } else if (replica->fFrameIndex != fFrameIndex) {
        // This replica is already asking for the next frame (because it has
        // already received the current frame).  Enqueue it:
        replica->fNext = fReplicasAwaitingNextFrame;
        fReplicasAwaitingNextFrame = replica;
    } else {
        // This replica is asking for the current frame.  Enqueue it:
        replica->fNext = fReplicasAwaitingCurrentFrame;
        fReplicasAwaitingCurrentFrame = replica;

        if (fInputSource != NULL && !fInputSource->isCurrentlyAwaitingData()) {
            // The current frame has already arrived, so deliver it to this
            // replica now:
            deliverReceivedFrame();
        }
    }
}

void StreamReplicator::deactivateStreamReplica(
        StreamReplica* replicaBeingDeactivated) {
    if (replicaBeingDeactivated->fFrameIndex == -1)
        return;  // this replica has already been deactivated (or was never
                 // activated at all)

    // Assert: fNumActiveReplicas > 0
    if (fNumActiveReplicas == 0)
        fprintf(stderr,
                "StreamReplicator::deactivateStreamReplica() Internal "
                "Error!\n");  // should not happen
    --fNumActiveReplicas;

    // Forget about any frame delivery that might have just been made to this
    // replica:
    if (replicaBeingDeactivated->fFrameIndex != fFrameIndex &&
        fNumDeliveriesMadeSoFar > 0)
        --fNumDeliveriesMadeSoFar;

    replicaBeingDeactivated->fFrameIndex = -1;

    // Check whether the replica being deactivated is the 'master' replica, or
    // is enqueued awaiting a frame:
    if (replicaBeingDeactivated == fMasterReplica) {
        // We need to replace the 'master replica', if we can:
        if (fReplicasAwaitingCurrentFrame == NULL) {
            // There's currently no replacement 'master replica'
            fMasterReplica = NULL;
        } else {
            // There's another replica that we can use as a replacement 'master
            // replica':
            fMasterReplica = fReplicasAwaitingCurrentFrame;
            fReplicasAwaitingCurrentFrame =
                    fReplicasAwaitingCurrentFrame->fNext;
            fMasterReplica->fNext = NULL;
        }

        // Check whether the read into the old master replica's buffer is still
        // pending, or has completed:
        if (fInputSource != NULL) {
            if (fInputSource->isCurrentlyAwaitingData()) {
                // We have a pending read into the old master replica's buffer.
                // We need to stop it, and retry the read with a new master (if
                // available)
                fInputSource->stopGettingFrames();

                if (fMasterReplica != NULL) {
                    fInputSource->getNextFrame(
                            fMasterReplica->fTo, fMasterReplica->fMaxSize,
                            afterGettingFrame, this, onSourceClosure, this);
                }
            } else {
                // The read into the old master replica's buffer has already
                // completed.  Copy the data to the new master replica (if any):
                if (fMasterReplica != NULL) {
                    StreamReplica::copyReceivedFrame(fMasterReplica,
                                                     replicaBeingDeactivated);
                } else {
                    // We don't have a new master replica, so we can't copy the
                    // received frame to any new replica that might ask for it.
                    // Fortunately this should be a very rare occurrence.
                }
            }
        }
    } else {
        // The replica that's being removed was not our 'master replica', but
        // make sure it's not on either of our queues:
        if (fReplicasAwaitingCurrentFrame != NULL) {
            if (replicaBeingDeactivated == fReplicasAwaitingCurrentFrame) {
                fReplicasAwaitingCurrentFrame = replicaBeingDeactivated->fNext;
                replicaBeingDeactivated->fNext = NULL;
            } else {
                for (StreamReplica* r1 = fReplicasAwaitingCurrentFrame;
                     r1->fNext != NULL; r1 = r1->fNext) {
                    if (r1->fNext == replicaBeingDeactivated) {
                        r1->fNext = replicaBeingDeactivated->fNext;
                        replicaBeingDeactivated->fNext = NULL;
                        break;
                    }
                }
            }
        }
        if (fReplicasAwaitingNextFrame != NULL) {
            if (replicaBeingDeactivated == fReplicasAwaitingNextFrame) {
                fReplicasAwaitingNextFrame = replicaBeingDeactivated->fNext;
                replicaBeingDeactivated->fNext = NULL;
            } else {
                for (StreamReplica* r2 = fReplicasAwaitingNextFrame;
                     r2->fNext != NULL; r2 = r2->fNext) {
                    if (r2->fNext == replicaBeingDeactivated) {
                        r2->fNext = replicaBeingDeactivated->fNext;
                        replicaBeingDeactivated->fNext = NULL;
                        break;
                    }
                }
            }
        }

        // Check for the possibility that - now that a replica has been
        // deactivated - all other replicas have received the current frame, and
        // so now we need to complete delivery to the master replica:
        if (fMasterReplica != NULL && fInputSource != NULL &&
            !fInputSource->isCurrentlyAwaitingData())
            deliverReceivedFrame();
    }

    if (fNumActiveReplicas == 0 && fInputSource != NULL)
        fInputSource->stopGettingFrames();  // tell our source to stop too
}

void StreamReplicator::removeStreamReplica(StreamReplica* replicaBeingRemoved) {
    // First, handle the replica that's being removed the same way that we would
    // if it were merely being deactivated:
    deactivateStreamReplica(replicaBeingRemoved);

    // Assert: fNumReplicas > 0
    if (fNumReplicas == 0)
        fprintf(stderr,
                "StreamReplicator::removeStreamReplica() Internal Error!\n");  // should not happen
    --fNumReplicas;

    // If this was the last replica, then delete ourselves (if we were set up to
    // do so):
    if (fNumReplicas == 0 && fDeleteWhenLastReplicaDies) {
        Medium::close(this);
        return;
    }
}

void StreamReplicator::afterGettingFrame(void* clientData,
                                         unsigned frameSize,
                                         unsigned numTruncatedBytes,
                                         struct timeval presentationTime,
                                         unsigned durationInMicroseconds) {
    ((StreamReplicator*)clientData)
            ->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime,
                                durationInMicroseconds);
}

void StreamReplicator::afterGettingFrame(unsigned frameSize,
                                         unsigned numTruncatedBytes,
                                         struct timeval presentationTime,
                                         unsigned durationInMicroseconds) {
    // The frame was read into our master replica's buffer.  Update the master
    // replica's state, but don't complete delivery to it just yet.  We do that
    // later, after we're sure that we've delivered it to all other replicas.
    fMasterReplica->fFrameSize = frameSize;
    fMasterReplica->fNumTruncatedBytes = numTruncatedBytes;
    fMasterReplica->fPresentationTime = presentationTime;
    fMasterReplica->fDurationInMicroseconds = durationInMicroseconds;

    deliverReceivedFrame();
}

void StreamReplicator::onSourceClosure(void* clientData) {
    ((StreamReplicator*)clientData)->onSourceClosure();
}

void StreamReplicator::onSourceClosure() {
    fInputSourceHasClosed = True;

    // Signal the closure to each replica that is currently awaiting a frame:
    StreamReplica* replica;
    while ((replica = fReplicasAwaitingCurrentFrame) != NULL) {
        fReplicasAwaitingCurrentFrame = replica->fNext;
        replica->fNext = NULL;
        replica->handleClosure();
    }
    while ((replica = fReplicasAwaitingNextFrame) != NULL) {
        fReplicasAwaitingNextFrame = replica->fNext;
        replica->fNext = NULL;
        replica->handleClosure();
    }
    if ((replica = fMasterReplica) != NULL) {
        fMasterReplica = NULL;
        replica->handleClosure();
    }
}

void StreamReplicator::deliverReceivedFrame() {
    // The 'master replica' has received its copy of the current frame.
    // Copy it (and complete delivery) to any other replica that has requested
    // this frame. Then, if no more requests for this frame are expected,
    // complete delivery to the 'master replica' itself.
    StreamReplica* replica;
    while ((replica = fReplicasAwaitingCurrentFrame) != NULL) {
        fReplicasAwaitingCurrentFrame = replica->fNext;
        replica->fNext = NULL;

        // Assert: fMasterReplica != NULL
        if (fMasterReplica == NULL)
            fprintf(stderr,
                    "StreamReplicator::deliverReceivedFrame() Internal Error "
                    "1!\n");  // shouldn't happen
        StreamReplica::copyReceivedFrame(replica, fMasterReplica);
        replica->fFrameIndex =
                1 - replica->fFrameIndex;  // toggle it (0<->1), because this
                                           // replica no longer awaits the
                                           // current frame
        ++fNumDeliveriesMadeSoFar;

        // Assert: fNumDeliveriesMadeSoFar < fNumActiveReplicas; // because we
        // still have the 'master replica' to deliver to
        if (!(fNumDeliveriesMadeSoFar < fNumActiveReplicas))
            fprintf(stderr,
                    "StreamReplicator::deliverReceivedFrame() Internal Error "
                    "2(%d,%d)!\n",
                    fNumDeliveriesMadeSoFar,
                    fNumActiveReplicas);  // should not happen

        // Complete delivery to this replica:
        FramedSource::afterGetting(replica);
    }

    if (fNumDeliveriesMadeSoFar == fNumActiveReplicas - 1 &&
        fMasterReplica != NULL) {
        // No more requests for this frame are expected, so complete delivery to
        // the 'master replica':
        replica = fMasterReplica;
        fMasterReplica = NULL;
        replica->fFrameIndex =
                1 - replica->fFrameIndex;  // toggle it (0<->1), because this
                                           // replica no longer awaits the
                                           // current frame
        fFrameIndex = 1 - fFrameIndex;  // toggle it (0<->1) for the next frame
        fNumDeliveriesMadeSoFar = 0;    // reset for the next frame

        if (fReplicasAwaitingNextFrame != NULL) {
            // One of the other replicas has already requested the next frame,
            // so make it the next 'master replica':
            fMasterReplica = fReplicasAwaitingNextFrame;
            fReplicasAwaitingNextFrame = fReplicasAwaitingNextFrame->fNext;
            fMasterReplica->fNext = NULL;

            // Arrange to read the next frame into this replica's buffer:
            if (fInputSource != NULL)
                fInputSource->getNextFrame(
                        fMasterReplica->fTo, fMasterReplica->fMaxSize,
                        afterGettingFrame, this, onSourceClosure, this);
        }

        // Move any other replicas that had already requested the next frame to
        // the 'requesting current frame' list: Assert:
        // fReplicasAwaitingCurrentFrame == NULL;
        if (!(fReplicasAwaitingCurrentFrame == NULL))
            fprintf(stderr,
                    "StreamReplicator::deliverReceivedFrame() Internal Error "
                    "3!\n");  // should not happen
        fReplicasAwaitingCurrentFrame = fReplicasAwaitingNextFrame;
        fReplicasAwaitingNextFrame = NULL;

        // Complete delivery to the 'master' replica (thereby completing all
        // deliveries for this frame):
        FramedSource::afterGetting(replica);
    }
}

////////// StreamReplica implementation //////////

StreamReplica::StreamReplica(StreamReplicator& ourReplicator)
    : FramedSource(ourReplicator.envir()),
      fOurReplicator(ourReplicator),
      fFrameIndex(-1 /*we haven't started playing yet*/),
      fNext(NULL) {}

StreamReplica::~StreamReplica() { fOurReplicator.removeStreamReplica(this); }

void StreamReplica::doGetNextFrame() { fOurReplicator.getNextFrame(this); }

void StreamReplica::doStopGettingFrames() {
    fOurReplicator.deactivateStreamReplica(this);
}

void StreamReplica::copyReceivedFrame(StreamReplica* toReplica,
                                      StreamReplica* fromReplica) {
    // First, figure out how much data to copy.  ("toReplica" might have a
    // smaller buffer than "fromReplica".)
    unsigned numNewBytesToTruncate =
            toReplica->fMaxSize < fromReplica->fFrameSize
                    ? fromReplica->fFrameSize - toReplica->fMaxSize
                    : 0;
    toReplica->fFrameSize = fromReplica->fFrameSize - numNewBytesToTruncate;
    toReplica->fNumTruncatedBytes =
            fromReplica->fNumTruncatedBytes + numNewBytesToTruncate;

    memmove(toReplica->fTo, fromReplica->fTo, toReplica->fFrameSize);
    toReplica->fPresentationTime = fromReplica->fPresentationTime;
    toReplica->fDurationInMicroseconds = fromReplica->fDurationInMicroseconds;
}
