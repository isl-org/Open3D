// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeMetalShaders.h"

#include <vector>

#import <Metal/Metal.h>

namespace open3d {
namespace visualization {
namespace rendering {

MetalComputePipelineHandle CompileMetalComputePipeline(
        std::uintptr_t device_handle,
        const std::string& source,
        const std::string& entry_point,
        const std::string& label,
        std::string* error_message) {
    MetalComputePipelineHandle handle;

    id<MTLDevice> device = (__bridge id<MTLDevice>)reinterpret_cast<void*>(
            device_handle);
    if (!device) {
        if (error_message) {
            *error_message = "Metal device handle is null.";
        }
        return handle;
    }

    NSString* source_string =
            [NSString stringWithUTF8String:source.c_str() ?: ""];
    NSString* entry_name =
            [NSString stringWithUTF8String:entry_point.c_str() ?: ""];
    NSString* pipeline_label =
            [NSString stringWithUTF8String:label.c_str() ?: ""];

    NSError* error = nil;
    id<MTLLibrary> library =
            [device newLibraryWithSource:source_string
                                 options:nil
                                   error:&error];
    if (!library) {
        if (error_message) {
            *error_message = error ? [[error localizedDescription] UTF8String]
                                   : "Failed to compile Metal library.";
        }
        return handle;
    }

    id<MTLFunction> function = [library newFunctionWithName:entry_name];
    if (!function) {
        if (error_message) {
            *error_message = "Failed to create Metal compute function '" +
                             entry_point + "'.";
        }
        return handle;
    }

    id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        if (error_message) {
            *error_message = error ? [[error localizedDescription] UTF8String]
                                   : "Failed to create Metal compute pipeline.";
        }
        return handle;
    }

    if (pipeline_label.length > 0) {
        library.label = pipeline_label;
        function.label = pipeline_label;
    }

    [library retain];
    [function retain];
    [pipeline retain];
    handle.library = reinterpret_cast<std::uintptr_t>(library);
    handle.function = reinterpret_cast<std::uintptr_t>(function);
    handle.pipeline = reinterpret_cast<std::uintptr_t>(pipeline);
    handle.valid = true;
    return handle;
}

void DestroyMetalComputePipeline(MetalComputePipelineHandle handle) {
    if (handle.pipeline) {
        [reinterpret_cast<id<MTLComputePipelineState>>(handle.pipeline) release];
    }
    if (handle.function) {
        [reinterpret_cast<id<MTLFunction>>(handle.function) release];
    }
    if (handle.library) {
        [reinterpret_cast<id<MTLLibrary>>(handle.library) release];
    }
}

bool DispatchMetalComputePipelines(std::uintptr_t command_queue_handle,
                                   const std::vector<MetalComputeDispatch>&
                                           dispatches,
                                   std::string* error_message) {
    id<MTLCommandQueue> command_queue =
            reinterpret_cast<id<MTLCommandQueue>>(command_queue_handle);
    if (!command_queue) {
        if (error_message) {
            *error_message = "Metal command queue handle is null.";
        }
        return false;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    if (!command_buffer) {
        if (error_message) {
            *error_message = "Failed to create Metal command buffer.";
        }
        return false;
    }

    for (const MetalComputeDispatch& dispatch : dispatches) {
        if (!dispatch.pipeline.valid || !dispatch.pipeline.pipeline) {
            continue;
        }

        id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
        if (!encoder) {
            if (error_message) {
                *error_message = "Failed to create Metal compute encoder.";
            }
            return false;
        }

        [encoder setComputePipelineState:reinterpret_cast<id<MTLComputePipelineState>>(
                                                dispatch.pipeline.pipeline)];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch.group_count_x,
                                                  dispatch.group_count_y,
                                                  dispatch.group_count_z)
                threadsPerThreadgroup:MTLSizeMake(dispatch.thread_count_x,
                                                  dispatch.thread_count_y,
                                                  dispatch.thread_count_z)];
        [encoder endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error && error_message) {
        *error_message = [[command_buffer.error localizedDescription] UTF8String];
    }
    return command_buffer.error == nil;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d