// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// TEMPORARY DIAGNOSTIC (to be removed): prints a native backtrace on
// SIGSEGV/SIGABRT to help root-cause a headless EGL/GL rendering crash seen
// only in CI, not reproducible locally. glibc's execinfo.h backtrace() API
// remains available (unlike the removed libSegFault.so convenience wrapper).
#if defined(__linux__)

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <cstdlib>

namespace {

void PrintBacktraceAndReraise(int sig) {
    static const char kHeader[] =
            "\n[Open3D DIAGNOSTIC] Caught fatal signal, native backtrace:\n";
    ssize_t unused_result = write(STDERR_FILENO, kHeader, sizeof(kHeader) - 1);
    (void)unused_result;
    void *frames[64];
    int num_frames = backtrace(frames, 64);
    backtrace_symbols_fd(frames, num_frames, STDERR_FILENO);
    signal(sig, SIG_DFL);
    raise(sig);
}

struct CrashBacktraceInstaller {
    CrashBacktraceInstaller() {
        signal(SIGSEGV, PrintBacktraceAndReraise);
        signal(SIGABRT, PrintBacktraceAndReraise);
    }
} g_crash_backtrace_installer;

}  // namespace

#endif  // defined(__linux__)
