#include "sync.h"

using namespace rsimpl;

frame_archive::frame_archive(const std::vector<subdevice_mode_selection> & selection, rs_stream key_stream) : key_stream(key_stream)
{
    // Store the mode selection that pertains to each native stream
    for(auto & mode : selection)
    {
        for(auto & o : mode.get_outputs())
        {
            modes[o.first] = mode;
        }
    }

    // Enumerate all streams we need to keep synchronized with the key stream
    for(auto s : {RS_STREAM_DEPTH, RS_STREAM_INFRARED, RS_STREAM_INFRARED2, RS_STREAM_COLOR})
    {
        if(is_stream_enabled(s) && s != key_stream) other_streams.push_back(s);
    }

    // Allocate an empty image for each stream, and move it to the frontbuffer
    // This allows us to assume that get_frame_data/get_frame_timestamp always return valid data
    alloc_frame(key_stream, 0);
    frontbuffer[key_stream] = std::move(backbuffer[key_stream]);
    for(auto s : other_streams)
    {
        alloc_frame(s, 0);
        frontbuffer[s] = std::move(backbuffer[s]);
    }
}

const byte * frame_archive::get_frame_data(rs_stream stream) const 
{ 
    return frontbuffer[stream].data.data();
}

int frame_archive::get_frame_timestamp(rs_stream stream) const
{ 
    return frontbuffer[stream].timestamp;
}

// Block until the next coherent frameset is available
void frame_archive::wait_for_frames()
{
    std::unique_lock<std::mutex> lock(mutex);
    const auto ready = [this]() { return !frames[key_stream].empty(); };
    if(!ready() && !cv.wait_for(lock, std::chrono::seconds(5), ready)) throw std::runtime_error("Timeout waiting for frames.");
    get_next_frames();
}

// If a coherent frameset is available, obtain it and return true, otherwise return false immediately
bool frame_archive::poll_for_frames()
{
    // TODO: Implement a user-specifiable timeout for how long to wait before returning false?
    std::unique_lock<std::mutex> lock(mutex);
    if(frames[key_stream].empty()) return false;
    get_next_frames();
    return true;
}

// Move frames from the queues to the frontbuffers to form the next coherent frameset
void frame_archive::get_next_frames()
{
    // Always dequeue a frame from the key stream
    dequeue_frame(key_stream);

    // Dequeue from other streams if the new frame is closer to the timestamp of the key stream than the old frame
    for(auto s : other_streams)
    {
        if(!frames[s].empty() && abs(frames[s].front().timestamp - frontbuffer[key_stream].timestamp) <= abs(frontbuffer[s].timestamp - frontbuffer[key_stream].timestamp))
        {
            dequeue_frame(s);
        }
    }
}

// Allocate a new frame in the backbuffer, potentially recycling a buffer from the freelist
byte * frame_archive::alloc_frame(rs_stream stream, int timestamp) 
{ 
    const size_t size = modes[stream].get_image_size(stream);

    {
        std::lock_guard<std::mutex> guard(mutex);

        // Attempt to obtain a buffer of the appropriate size from the freelist
        for(auto it = begin(freelist); it != end(freelist); ++it)
        {
            if(it->data.size() == size)
            {
                backbuffer[stream] = std::move(*it);
                freelist.erase(it);
                break;
            }
        }

        // Discard buffers that have been in the freelist for longer than 1s
        for(auto it = begin(freelist); it != end(freelist); )
        {
            if(timestamp > it->timestamp + 1000) it = freelist.erase(it);
            else ++it;
        }
    }

    backbuffer[stream].data.resize(size); // TODO: Allow users to provide a custom allocator for frame buffers
    backbuffer[stream].timestamp = timestamp;
    return backbuffer[stream].data.data();
}

// Move a frame from the backbuffer to the back of the queue
void frame_archive::commit_frame(rs_stream stream) 
{
    std::unique_lock<std::mutex> lock(mutex);
    frames[stream].push_back(std::move(backbuffer[stream]));
    cull_frames();
    lock.unlock();
    if(!frames[key_stream].empty()) cv.notify_one();
}

// Discard all frames which are older than the most recent coherent frameset
void frame_archive::cull_frames()
{
    // Never keep more than four frames around in any given stream, regardless of timestamps
    for(auto s : {RS_STREAM_DEPTH, RS_STREAM_COLOR, RS_STREAM_INFRARED, RS_STREAM_INFRARED2})
    {
        while(frames[s].size() > 4)
        {
            discard_frame(s);
        }
    }

    // Cannot do any culling unless at least one frame is enqueued for each enabled stream
    if(frames[key_stream].empty()) return;
    for(auto s : other_streams) if(frames[s].empty()) return;

    // We can discard frames from the key stream if we have at least two and the latter is closer to the most recent frame of all other streams than the former
    while(true)
    {
        if(frames[key_stream].size() < 2) break;
        const int t0 = frames[key_stream][0].timestamp, t1 = frames[key_stream][1].timestamp;

        bool valid_to_skip = true;
        for(auto s : other_streams)
        {
            if(abs(t0 - frames[s].back().timestamp) < abs(t1 - frames[s].back().timestamp))
            {
                valid_to_skip = false;
                break;
            }
        }
        if(!valid_to_skip) break;

        discard_frame(key_stream);
    }

    // We can discard frames for other streams if we have at least two and the latter is closer to the next key stream frame than the former
    for(auto s : other_streams)
    {
        while(true)
        {
            if(frames[s].size() < 2) break;
            const int t0 = frames[s][0].timestamp, t1 = frames[s][1].timestamp;

            if(abs(t0 - frames[key_stream].front().timestamp) < abs(t1 - frames[key_stream].front().timestamp)) break;
            discard_frame(s);
        }
    }
}

// Move a single frame from the head of the queue to the front buffer, while recycling the front buffer into the freelist
void frame_archive::dequeue_frame(rs_stream stream)
{
    if(!frontbuffer[stream].data.empty()) freelist.push_back(std::move(frontbuffer[stream]));
    frontbuffer[stream] = std::move(frames[stream].front());
    frames[stream].erase(begin(frames[stream]));
}

// Move a single frame from the head of the queue directly to the freelist
void frame_archive::discard_frame(rs_stream stream)
{
    freelist.push_back(std::move(frames[stream].front()));
    frames[stream].erase(begin(frames[stream]));    
}