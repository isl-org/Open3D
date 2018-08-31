// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#ifdef RS_USE_WMF_BACKEND

#if (_MSC_FULL_VER < 180040629)
    #error At least Visual Studio 2013 Update 5 is required to compile this backend
#endif

#include "uvc.h"

#include <Shlwapi.h>        // For QISearch, etc.
#include <mfapi.h>          // For MFStartup, etc.
#include <mfidl.h>          // For MF_DEVSOURCE_*, etc.
#include <mfreadwrite.h>    // MFCreateSourceReaderFromMediaSource
#include <mferror.h>

#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "winusb.lib")

#include <uuids.h>
#include <vidcap.h>
#include <ksmedia.h>
#include <ksproxy.h>

#include <Cfgmgr32.h>
#include <SetupAPI.h>
#include <WinUsb.h>

#include <thread>
#include <chrono>
#include <algorithm>
#include <regex>
#include <map>

#include <strsafe.h>

namespace rsimpl
{
    namespace uvc
    {
        static std::string win_to_utf(const WCHAR * s)
        {
            int len = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, NULL, NULL);
            if(len == 0) throw std::runtime_error(to_string() << "WideCharToMultiByte(...) returned 0 and GetLastError() is " << GetLastError());
            std::string buffer(len-1, ' ');
            len = WideCharToMultiByte(CP_UTF8, 0, s, -1, &buffer[0], (int)buffer.size()+1, NULL, NULL);
            if(len == 0) throw std::runtime_error(to_string() << "WideCharToMultiByte(...) returned 0 and GetLastError() is " << GetLastError());
            return buffer;
        }

        static void check(const char * call, HRESULT hr)
        {
            if(FAILED(hr)) throw std::runtime_error(to_string() << call << "(...) returned 0x" << std::hex << (uint32_t)hr);
        }

        template<class T> class com_ptr
        {
            T * p;

            void ref(T * new_p)
            {
                if(p == new_p) return;
                unref();
                p = new_p;
                if(p) p->AddRef();
            }

            void unref()
            {
                if(p)
                {
                    p->Release();
                    p = nullptr;
                }
            }
        public:
            com_ptr() : p() {}
            com_ptr(T * p) : com_ptr() { ref(p); }
            com_ptr(const com_ptr & r) : com_ptr(r.p) {}
            ~com_ptr() { unref(); }

            operator T * () const { return p; }
            T & operator * () const { return *p; }
            T * operator -> () const { return p; }

            T ** operator & () { unref(); return &p; }
            com_ptr & operator = (const com_ptr & r) { ref(r.p); return *this; }            
        };

        std::vector<std::string> tokenize(std::string string, char separator)
        {
            std::vector<std::string> tokens;
            std::string::size_type i1 = 0;
            while(true)
            {
                auto i2 = string.find(separator, i1);
                if(i2 == std::string::npos)
                {
                    tokens.push_back(string.substr(i1));
                    return tokens;
                }
                tokens.push_back(string.substr(i1, i2-i1));
                i1 = i2+1;
            }
        }

        bool parse_usb_path(int & vid, int & pid, int & mi, std::string & unique_id, const std::string & path)
        {
            auto name = path;
            std::transform(begin(name), end(name), begin(name), ::tolower);
            auto tokens = tokenize(name, '#');
            if(tokens.size() < 1 || tokens[0] != R"(\\?\usb)") return false; // Not a USB device
            if(tokens.size() < 3)
            {
                LOG_ERROR("malformed usb device path: " << name);
                return false;
            }

            auto ids = tokenize(tokens[1], '&');
            if(ids[0].size() != 8 || ids[0].substr(0,4) != "vid_" || !(std::istringstream(ids[0].substr(4,4)) >> std::hex >> vid))
            {
                LOG_ERROR("malformed vid string: " << tokens[1]);
                return false;
            }

            if(ids[1].size() != 8 || ids[1].substr(0,4) != "pid_" || !(std::istringstream(ids[1].substr(4,4)) >> std::hex >> pid))
            {
                LOG_ERROR("malformed pid string: " << tokens[1]);
                return false;
            }

            if(ids[2].size() != 5 || ids[2].substr(0,3) != "mi_" || !(std::istringstream(ids[2].substr(3,2)) >> mi))
            {
                LOG_ERROR("malformed mi string: " << tokens[1]);
                return false;
            }

            ids = tokenize(tokens[2], '&');
            if(ids.size() < 2)
            {
                LOG_ERROR("malformed id string: " << tokens[2]);
                return false;
            }
            unique_id = ids[1];
            return true;
        }

        struct context
        {
            context()
            {
                CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET);
            }
            ~context()
            {   
                MFShutdown();
                CoUninitialize();
            }
        };

        class reader_callback : public IMFSourceReaderCallback
        {
            std::weak_ptr<device> owner; // The device holds a reference to us, so use weak_ptr to prevent a cycle
            int subdevice_index;
            ULONG ref_count;
            volatile bool streaming = false;
        public:
            reader_callback(std::weak_ptr<device> owner, int subdevice_index) : owner(owner), subdevice_index(subdevice_index), ref_count() {}

            bool is_streaming() const { return streaming; }
            void on_start() { streaming = true; }

            // Implement IUnknown
            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void ** ppvObject) override 
            {
                static const QITAB table[] = {QITABENT(reader_callback, IUnknown), QITABENT(reader_callback, IMFSourceReaderCallback), {0}};
                return QISearch(this, table, riid, ppvObject);
            }
            ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_count); }
            ULONG STDMETHODCALLTYPE Release() override 
            { 
                ULONG count = InterlockedDecrement(&ref_count);
                if(count == 0) delete this;
                return count;
            }

            // Implement IMFSourceReaderCallback
            HRESULT STDMETHODCALLTYPE OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample * sample) override;
            HRESULT STDMETHODCALLTYPE OnFlush(DWORD dwStreamIndex) override { streaming = false; return S_OK; }
            HRESULT STDMETHODCALLTYPE OnEvent(DWORD dwStreamIndex, IMFMediaEvent *pEvent) override { return S_OK; }
        };

        struct subdevice
        {
            com_ptr<reader_callback> reader_callback;
            com_ptr<IMFActivate> mf_activate;
            com_ptr<IMFMediaSource> mf_media_source;
            com_ptr<IAMCameraControl> am_camera_control;
            com_ptr<IAMVideoProcAmp> am_video_proc_amp;            
            std::map<int, com_ptr<IKsControl>> ks_controls;
            com_ptr<IMFSourceReader> mf_source_reader;
            std::function<void(const void * frame)> callback;

            com_ptr<IMFMediaSource> get_media_source()
            {
                if(!mf_media_source)
                {
                    check("IMFActivate::ActivateObject", mf_activate->ActivateObject(__uuidof(IMFMediaSource), (void **)&mf_media_source));
                    check("IMFMediaSource::QueryInterface", mf_media_source->QueryInterface(__uuidof(IAMCameraControl), (void **)&am_camera_control));
                    if(SUCCEEDED(mf_media_source->QueryInterface(__uuidof(IAMVideoProcAmp), (void **)&am_video_proc_amp))) LOG_DEBUG("obtained IAMVideoProcAmp");                    
                }
                return mf_media_source;


            }

            IKsControl * get_ks_control(const uvc::extension_unit & xu)
            {
                auto it = ks_controls.find(xu.node);
                if(it != end(ks_controls)) return it->second;

                get_media_source();

                // Attempt to retrieve IKsControl
                com_ptr<IKsTopologyInfo> ks_topology_info = NULL;
                check("QueryInterface", mf_media_source->QueryInterface(__uuidof(IKsTopologyInfo), (void **)&ks_topology_info));

                GUID node_type;
                check("get_NodeType", ks_topology_info->get_NodeType(xu.node, &node_type));
                const GUID KSNODETYPE_DEV_SPECIFIC_LOCAL{0x941C7AC0L, 0xC559, 0x11D0, {0x8A, 0x2B, 0x00, 0xA0, 0xC9, 0x25, 0x5A, 0xC1}};
                if(node_type != KSNODETYPE_DEV_SPECIFIC_LOCAL) throw std::runtime_error(to_string() << "Invalid extension unit node ID: " << xu.node);

                com_ptr<IUnknown> unknown;
                check("CreateNodeInstance", ks_topology_info->CreateNodeInstance(xu.node, IID_IUnknown, (LPVOID *)&unknown));

                com_ptr<IKsControl> ks_control;
                check("QueryInterface", unknown->QueryInterface(__uuidof(IKsControl), (void **)&ks_control));
                LOG_INFO("Obtained KS control node " << xu.node);
                return ks_controls[xu.node] = ks_control;
            }
        };

        struct device
        {
            const std::shared_ptr<context> parent;
            const int vid, pid;
            const std::string unique_id;

            std::vector<subdevice> subdevices;

            HANDLE usb_file_handle = INVALID_HANDLE_VALUE;
            WINUSB_INTERFACE_HANDLE usb_interface_handle = INVALID_HANDLE_VALUE;

            device(std::shared_ptr<context> parent, int vid, int pid, std::string unique_id) : parent(move(parent)), vid(vid), pid(pid), unique_id(move(unique_id))
            {

            }

            ~device() { stop_streaming(); close_win_usb(); }

            IKsControl * get_ks_control(const uvc::extension_unit & xu)
            {
                return subdevices[xu.subdevice].get_ks_control(xu);
            }

            void start_streaming()
            {
                for(auto & sub : subdevices)
                {
                    if(sub.mf_source_reader)
                    {
                        sub.reader_callback->on_start();
                        check("IMFSourceReader::ReadSample", sub.mf_source_reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, NULL, NULL, NULL));                    
                    }
                }
            }

            void stop_streaming()
            {
                for(auto & sub : subdevices)
                {
                    if(sub.mf_source_reader) sub.mf_source_reader->Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
                }
                while(true)
                {
                    bool is_streaming = false;
                    for(auto & sub : subdevices) is_streaming |= sub.reader_callback->is_streaming();                   
                    if(is_streaming) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    else break;
                }

                // Free up our source readers, our KS control nodes, and our media sources, but retain our original IMFActivate objects for later reuse
                for(auto & sub : subdevices)
                {
                    sub.mf_source_reader = nullptr;
                    sub.am_camera_control = nullptr;
                    sub.am_video_proc_amp = nullptr;
                    sub.ks_controls.clear();
                    if(sub.mf_media_source)
                    {
                        sub.mf_media_source = nullptr;
                        check("IMFActivate::ShutdownObject", sub.mf_activate->ShutdownObject());
                    }
                    sub.callback = {};
                }
            }

            com_ptr<IMFMediaSource> get_media_source(int subdevice_index)
            {
                return subdevices[subdevice_index].get_media_source();
            }           

            void open_win_usb(const guid & interface_guid, int interface_number) try
            {    
                static_assert(sizeof(guid) == sizeof(GUID), "struct packing error");
                HDEVINFO device_info = SetupDiGetClassDevs((const GUID *)&interface_guid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
                if (device_info == INVALID_HANDLE_VALUE) throw std::runtime_error("SetupDiGetClassDevs");
                auto di = std::shared_ptr<void>(device_info, SetupDiDestroyDeviceInfoList);

                for(int member_index = 0; ; ++member_index)
                {
                    // Enumerate all the device interfaces in the device information set.
                    SP_DEVICE_INTERFACE_DATA interfaceData = {sizeof(SP_DEVICE_INTERFACE_DATA)};
                    if(SetupDiEnumDeviceInterfaces(device_info, nullptr, (const GUID *)&interface_guid, member_index, &interfaceData) == FALSE)
                    {
                        if(GetLastError() == ERROR_NO_MORE_ITEMS) break;
                        continue;
                    }                           

                    // Allocate space for a detail data struct
                    unsigned long detail_data_size = 0;
                    SetupDiGetDeviceInterfaceDetail(device_info, &interfaceData, nullptr, 0, &detail_data_size, nullptr);
                    if(GetLastError() != ERROR_INSUFFICIENT_BUFFER)
                    {
                        LOG_ERROR("SetupDiGetDeviceInterfaceDetail failed");
                        continue;
                    }
                    auto alloc = std::malloc(detail_data_size);
                    if(!alloc) throw std::bad_alloc();

                    // Retrieve the detail data struct
                    auto detail_data = std::shared_ptr<SP_DEVICE_INTERFACE_DETAIL_DATA>(reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA *>(alloc), std::free);
                    detail_data->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
                    if (!SetupDiGetDeviceInterfaceDetail(device_info, &interfaceData, detail_data.get(), detail_data_size, nullptr, nullptr))
                    {
                        LOG_ERROR("SetupDiGetDeviceInterfaceDetail failed");
                        continue;
                    }
                    if (detail_data->DevicePath == nullptr) continue;

                    // Check if this is our device
                    int usb_vid, usb_pid, usb_mi; std::string usb_unique_id;
                    if(!parse_usb_path(usb_vid, usb_pid, usb_mi, usb_unique_id, win_to_utf(detail_data->DevicePath))) continue;
                    if(usb_vid != vid || usb_pid != pid || usb_mi != interface_number || usb_unique_id != unique_id) continue;                    
                        
                    usb_file_handle = CreateFile(detail_data->DevicePath, GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE | FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
                    if (usb_file_handle == INVALID_HANDLE_VALUE) throw std::runtime_error("CreateFile(...) failed");

                    if(!WinUsb_Initialize(usb_file_handle, &usb_interface_handle))
                    {
                        LOG_ERROR("Last Error: " << GetLastError());
                        throw std::runtime_error("could not initialize winusb");
                    }

                    // We successfully set up a WinUsb interface handle to our device
                    return;
                }
                throw std::runtime_error("Unable to open device via WinUSB");
            }
            catch(...)
            {
                close_win_usb();
                throw;
            }

            void close_win_usb()
            {
                if (usb_interface_handle != INVALID_HANDLE_VALUE)
                {
                    WinUsb_Free(usb_interface_handle);
                    usb_interface_handle = INVALID_HANDLE_VALUE;
                }

                if(usb_file_handle != INVALID_HANDLE_VALUE)
                {
                    CloseHandle(usb_file_handle);
                    usb_file_handle = INVALID_HANDLE_VALUE;
                }
            }

            bool usb_synchronous_read(uint8_t endpoint, void * buffer, int bufferLength, int * actual_length, DWORD TimeOut)
            {
                if (usb_interface_handle == INVALID_HANDLE_VALUE) throw std::runtime_error("winusb has not been initialized");

                auto result = false;

                BOOL bRetVal = true;
                
                ULONG lengthTransferred;

                bRetVal = WinUsb_ReadPipe(usb_interface_handle, endpoint, (PUCHAR)buffer, bufferLength, &lengthTransferred, NULL);

                if (bRetVal)
                    result = true;
                else
                {
                    auto lastResult = GetLastError();
                    WinUsb_ResetPipe(usb_interface_handle, endpoint);
                    result = false;
                }

                *actual_length = lengthTransferred;
                return result;
            }

            bool usb_synchronous_write(uint8_t endpoint, void * buffer, int bufferLength, DWORD TimeOut)
            {
                if (usb_interface_handle == INVALID_HANDLE_VALUE) throw std::runtime_error("winusb has not been initialized");

                auto result = false;

                ULONG lengthWritten;
                auto bRetVal = WinUsb_WritePipe(usb_interface_handle, endpoint, (PUCHAR)buffer, bufferLength, &lengthWritten, NULL);
                if (bRetVal)
                    result = true;
                else
                {
                    auto lastError = GetLastError();
                    WinUsb_ResetPipe(usb_interface_handle, endpoint);
                    LOG_ERROR("WinUsb_ReadPipe failure... lastError: " << lastError);
                    result = false;
                }

                return result;
            }
        };

        HRESULT reader_callback::OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample * sample) 
        {
            if(auto owner_ptr = owner.lock())
            {
                if(sample)
                {
                    com_ptr<IMFMediaBuffer> buffer = NULL;
                    if(SUCCEEDED(sample->GetBufferByIndex(0, &buffer)))
                    {
                        BYTE * byte_buffer; DWORD max_length, current_length;
                        if(SUCCEEDED(buffer->Lock(&byte_buffer, &max_length, &current_length)))
                        {
                            owner_ptr->subdevices[subdevice_index].callback(byte_buffer);
                            HRESULT hr = buffer->Unlock();
                        }
                    }
                }

                HRESULT hr = owner_ptr->subdevices[subdevice_index].mf_source_reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, NULL, NULL, NULL);
                switch(hr)
                {
                case S_OK: break;
                case MF_E_INVALIDREQUEST: LOG_ERROR("ReadSample returned MF_E_INVALIDREQUEST"); break;
                case MF_E_INVALIDSTREAMNUMBER: LOG_ERROR("ReadSample returned MF_E_INVALIDSTREAMNUMBER"); break;
                case MF_E_NOTACCEPTING: LOG_ERROR("ReadSample returned MF_E_NOTACCEPTING"); break;
                case E_INVALIDARG: LOG_ERROR("ReadSample returned E_INVALIDARG"); break;
                case MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED: LOG_ERROR("ReadSample returned MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED"); break;
                default: LOG_ERROR("ReadSample returned HRESULT " << std::hex << (uint32_t)hr); break;
                }
                if(hr != S_OK) streaming = false;
            }
            return S_OK; 
        }

        ////////////
        // device //
        ////////////

        int get_vendor_id(const device & device) { return device.vid; }
        int get_product_id(const device & device) { return device.pid; }

        void get_control(const device & device, const extension_unit & xu, uint8_t ctrl, void *data, int len)
        {
            auto ks_control = const_cast<uvc::device &>(device).get_ks_control(xu);

            KSP_NODE node;
            memset(&node, 0, sizeof(KSP_NODE));
            node.Property.Set = reinterpret_cast<const GUID &>(xu.id);
            node.Property.Id = ctrl;
            node.Property.Flags = KSPROPERTY_TYPE_GET | KSPROPERTY_TYPE_TOPOLOGY;
            node.NodeId = xu.node;

            ULONG bytes_received = 0;
            check("IKsControl::KsProperty", ks_control->KsProperty((PKSPROPERTY)&node, sizeof(node), data, len, &bytes_received));
            if(bytes_received != len) throw std::runtime_error("XU read did not return enough data");
        }

        void set_control(device & device, const extension_unit & xu, uint8_t ctrl, void *data, int len)
        {        
            auto ks_control = device.get_ks_control(xu);

            KSP_NODE node;
            memset(&node, 0, sizeof(KSP_NODE));
            node.Property.Set = reinterpret_cast<const GUID &>(xu.id);
            node.Property.Id = ctrl;
            node.Property.Flags = KSPROPERTY_TYPE_SET | KSPROPERTY_TYPE_TOPOLOGY;
            node.NodeId = xu.node;
                
            check("IKsControl::KsProperty", ks_control->KsProperty((PKSPROPERTY)&node, sizeof(KSP_NODE), data, len, nullptr));
        }

        void claim_interface(device & device, const guid & interface_guid, int interface_number)
        {
            device.open_win_usb(interface_guid, interface_number);
        }

        void bulk_transfer(device & device, uint8_t endpoint, void * data, int length, int *actual_length, unsigned int timeout)
        {       
            if(USB_ENDPOINT_DIRECTION_OUT(endpoint))
            {
                device.usb_synchronous_write(endpoint, data, length, timeout);
            }
            
            if(USB_ENDPOINT_DIRECTION_IN(endpoint))
            {
                auto actualLen = ULONG(actual_length);
                device.usb_synchronous_read(endpoint, data, length, actual_length, timeout);
            }
        }

        void set_subdevice_mode(device & device, int subdevice_index, int width, int height, uint32_t fourcc, int fps, std::function<void(const void * frame)> callback)
        {
            auto & sub = device.subdevices[subdevice_index];
            
            if(!sub.mf_source_reader)
            {
                com_ptr<IMFAttributes> pAttributes;
                check("MFCreateAttributes", MFCreateAttributes(&pAttributes, 1));
                check("IMFAttributes::SetUnknown", pAttributes->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, static_cast<IUnknown *>(sub.reader_callback)));
                check("MFCreateSourceReaderFromMediaSource", MFCreateSourceReaderFromMediaSource(sub.get_media_source(), pAttributes, &sub.mf_source_reader));
            }

            for (DWORD j = 0; ; j++)
            {
                com_ptr<IMFMediaType> media_type;
                HRESULT hr = sub.mf_source_reader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, j, &media_type);
                if (hr == MF_E_NO_MORE_TYPES) break;
                check("IMFSourceReader::GetNativeMediaType", hr);

                UINT32 uvc_width, uvc_height, uvc_fps_num, uvc_fps_denom; GUID subtype;
                check("MFGetAttributeSize", MFGetAttributeSize(media_type, MF_MT_FRAME_SIZE, &uvc_width, &uvc_height));
                if(uvc_width != width || uvc_height != height) continue;

                check("IMFMediaType::GetGUID", media_type->GetGUID(MF_MT_SUBTYPE, &subtype));
                if(reinterpret_cast<const big_endian<uint32_t> &>(subtype.Data1) != fourcc) continue;

                check("MFGetAttributeRatio", MFGetAttributeRatio(media_type, MF_MT_FRAME_RATE, &uvc_fps_num, &uvc_fps_denom));
                if(uvc_fps_denom == 0) continue;
                int uvc_fps = uvc_fps_num / uvc_fps_denom;
                if(std::abs(fps - uvc_fps) > 1) continue;

                check("IMFSourceReader::SetCurrentMediaType", sub.mf_source_reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, media_type));
                sub.callback = callback;
                return;
            }
            throw std::runtime_error("no matching media type");
        }

        void start_streaming(device & device, int num_transfer_bufs) { device.start_streaming(); }
        void stop_streaming(device & device) { device.stop_streaming(); }

        struct pu_control { rs_option option; long property; bool enable_auto; };
        static const pu_control pu_controls[] = {
            {RS_OPTION_COLOR_BACKLIGHT_COMPENSATION, VideoProcAmp_BacklightCompensation},
            {RS_OPTION_COLOR_BRIGHTNESS, VideoProcAmp_Brightness},
            {RS_OPTION_COLOR_CONTRAST, VideoProcAmp_Contrast},
            {RS_OPTION_COLOR_GAIN, VideoProcAmp_Gain},
            {RS_OPTION_COLOR_GAMMA, VideoProcAmp_Gamma},
            {RS_OPTION_COLOR_HUE, VideoProcAmp_Hue},
            {RS_OPTION_COLOR_SATURATION, VideoProcAmp_Saturation},
            {RS_OPTION_COLOR_SHARPNESS, VideoProcAmp_Sharpness},
            {RS_OPTION_COLOR_WHITE_BALANCE, VideoProcAmp_WhiteBalance},
            {RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, VideoProcAmp_WhiteBalance, true},
        };

        void set_pu_control(device & device, int subdevice, rs_option option, int value)
        {
            auto & sub = device.subdevices[subdevice];
            sub.get_media_source();
            if(option == RS_OPTION_COLOR_EXPOSURE)
            {
                check("IAMCameraControl::Set", sub.am_camera_control->Set(CameraControl_Exposure, static_cast<int>(std::round(log2(static_cast<double>(value) / 10000))), CameraControl_Flags_Manual));
                return;
            }
            if(option == RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE)
            {
                if(value) check("IAMCameraControl::Set", sub.am_camera_control->Set(CameraControl_Exposure, 0, CameraControl_Flags_Auto));
                else
                {
                    long min, max, step, def, caps;
                    check("IAMCameraControl::GetRange", sub.am_camera_control->GetRange(CameraControl_Exposure, &min, &max, &step, &def, &caps));
                    check("IAMCameraControl::Set", sub.am_camera_control->Set(CameraControl_Exposure, def, CameraControl_Flags_Manual));
                }
                return;
            }
            for(auto & pu : pu_controls)
            {
                if(option == pu.option)
                {
                    if(pu.enable_auto)
                    {
                        if(value) check("IAMVideoProcAmp::Set", sub.am_video_proc_amp->Set(pu.property, 0, VideoProcAmp_Flags_Auto));
                        else
                        {
                            long min, max, step, def, caps;
                            check("IAMVideoProcAmp::GetRange", sub.am_video_proc_amp->GetRange(pu.property, &min, &max, &step, &def, &caps));
                            check("IAMVideoProcAmp::Set", sub.am_video_proc_amp->Set(pu.property, def, VideoProcAmp_Flags_Manual));    
                        }
                    }
                    else check("IAMVideoProcAmp::Set", sub.am_video_proc_amp->Set(pu.property, value, VideoProcAmp_Flags_Manual));
                    return;
                }
            }
            throw std::runtime_error("unsupported control");
        }

        int win_to_uvc_exposure(int value) { return static_cast<int>(std::round(exp2(static_cast<double>(value)) * 10000)); }

        void get_pu_control_range(const device & device, int subdevice, rs_option option, int * min, int * max)
        {
            if(option >= RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE && option <= RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE)
            {
                if(min) *min = 0;
                if(max) *max = 1;
                return;
            }

            auto & sub = device.subdevices[subdevice];
            const_cast<uvc::subdevice &>(sub).get_media_source();
            long minVal=0, maxVal=0, steppingDelta=0, defVal=0, capsFlag=0;
            if(option == RS_OPTION_COLOR_EXPOSURE)
            {
                check("IAMCameraControl::Get", sub.am_camera_control->GetRange(CameraControl_Exposure, &minVal, &maxVal, &steppingDelta, &defVal, &capsFlag));
                if(min) *min = win_to_uvc_exposure(minVal);
                if(max) *max = win_to_uvc_exposure(maxVal);
                return;
            }
            for(auto & pu : pu_controls)
            {
                if(option == pu.option)
                {
                    check("IAMVideoProcAmp::GetRange", sub.am_video_proc_amp->GetRange(pu.property, &minVal, &maxVal, &steppingDelta, &defVal, &capsFlag));
                    if(min) *min = static_cast<int>(minVal);
                    if(max) *max = static_cast<int>(maxVal);
                    return;
                }
            }
            throw std::runtime_error("unsupported control");
        }

        int get_pu_control(const device & device, int subdevice, rs_option option)
        {
            auto & sub = device.subdevices[subdevice];
            long value=0, flags=0;
            if(option == RS_OPTION_COLOR_EXPOSURE)
            {
                check("IAMCameraControl::Get", sub.am_camera_control->Get(CameraControl_Exposure, &value, &flags));
                return win_to_uvc_exposure(value);
            }
            if(option == RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE)
            {
                check("IAMCameraControl::Get", sub.am_camera_control->Get(CameraControl_Exposure, &value, &flags));
                return flags == CameraControl_Flags_Auto;          
            }
            for(auto & pu : pu_controls)
            {
                if(option == pu.option)
                {
                    check("IAMVideoProcAmp::Get", sub.am_video_proc_amp->Get(pu.property, &value, &flags));
                    if(pu.enable_auto) return flags == VideoProcAmp_Flags_Auto;
                    else return value;
                }
            }
            throw std::runtime_error("unsupported control");
        }

        /////////////
        // context //
        /////////////

        std::shared_ptr<context> create_context()
        {
            return std::make_shared<context>();
        }

        std::vector<std::shared_ptr<device>> query_devices(std::shared_ptr<context> context)
        {
            IMFAttributes * pAttributes = NULL;
            check("MFCreateAttributes", MFCreateAttributes(&pAttributes, 1));
            check("IMFAttributes::SetGUID", pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID));
 
            IMFActivate ** ppDevices;
            UINT32 numDevices;
            check("MFEnumDeviceSources", MFEnumDeviceSources(pAttributes, &ppDevices, &numDevices));

            std::vector<std::shared_ptr<device>> devices;
            for(UINT32 i=0; i<numDevices; ++i)
            {
                com_ptr<IMFActivate> pDevice;
                *&pDevice = ppDevices[i];

                WCHAR * wchar_name = NULL; UINT32 length;
                pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &wchar_name, &length);
                auto name = win_to_utf(wchar_name);
                CoTaskMemFree(wchar_name);

                int vid, pid, mi; std::string unique_id;
                if(!parse_usb_path(vid, pid, mi, unique_id, name)) continue;

                std::shared_ptr<device> dev;
                for(auto & d : devices)
                {
                    if(d->vid == vid && d->pid == pid && d->unique_id == unique_id)
                    {
                        dev = d;
                    }
                }
                if(!dev)
                {
                    dev = std::make_shared<device>(context, vid, pid, unique_id);
                    devices.push_back(dev);
                }

                size_t subdevice_index = mi/2;
                if(subdevice_index >= dev->subdevices.size()) dev->subdevices.resize(subdevice_index+1);

                dev->subdevices[subdevice_index].reader_callback = new reader_callback(dev, static_cast<int>(subdevice_index));
                dev->subdevices[subdevice_index].mf_activate = pDevice;                
            }
            CoTaskMemFree(ppDevices);
            return devices;
        }
    }
}

#endif
