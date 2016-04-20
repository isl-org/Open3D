// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "context.h"
#include "uvc.h"
#include "r200.h"
#include "f200.h"

rs_context::rs_context() : rs_context(0)
{
    context = rsimpl::uvc::create_context();

    for(auto device : query_devices(context))
    {
        LOG_INFO("UVC device detected with VID = 0x" << std::hex << get_vendor_id(*device) << " PID = 0x" << get_product_id(*device));

		if (get_vendor_id(*device) != 32902)
			continue;
				
        switch(get_product_id(*device))
        {
        case 2688: devices.push_back(rsimpl::make_r200_device(device)); break;
        case 2662: devices.push_back(rsimpl::make_f200_device(device)); break;
        case 2725: devices.push_back(rsimpl::make_sr300_device(device)); break;
        }
    }
}

// Enforce singleton semantics on rs_context

bool rs_context::singleton_alive = false;

rs_context::rs_context(int)
{
    if(singleton_alive) throw std::runtime_error("rs_context has singleton semantics, only one may exist at a time");
    singleton_alive = true;
}

rs_context::~rs_context()
{
    assert(singleton_alive);
    singleton_alive = false;
}
