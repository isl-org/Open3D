# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import platform
import subprocess

import numpy as np
import tabulate
import nvidia_smi


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo | grep 'model name' -m 1"
        name = subprocess.check_output(command, shell=True).strip()
        return str(name, 'utf-8')


def run_command(command):
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            shell=True)
    return result.stdout.strip()


def print_system_info():
    # print results
    nvcc_version = run_command("nvcc --version")
    os_version = run_command("cat /etc/os-release")
    cpu_info = get_processor_name()
    gpu_info = run_command("nvidia-smi")
    print("======System Info======")
    print("[CUDA]")
    print(nvcc_version)
    print("")

    print("[OS]")
    print(os_version)
    print("")

    print("[CPU]")
    print(cpu_info)
    print("")

    print("[GPU]")
    print(gpu_info)
    print("======System Info [End]=")


def measure_time(fn, min_samples=10, max_samples=100, max_time_in_sec=10.0):
    """Measure time to run fn. Returns the elapsed time each run."""
    from time import perf_counter_ns
    t = []
    for i in range(max_samples):
        if sum(t) / 1e9 >= max_time_in_sec and i >= min_samples:
            break
        t.append(-perf_counter_ns())
        try:
            ans = fn()
        except Exception as e:
            print(e)
            return np.array([np.nan])
        t[-1] += perf_counter_ns()
        del ans
    print('.', end='')
    return np.array(t) / 1e9


def measure_memory(fn, handle):
    """Measure memory to run fn. Returns the maximum allocated memory each run."""
    try:
        _ = fn()
    except Exception as e:
        print(e)
        return np.nan
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    memory = info.used / 1000. / 1000. / 1000.
    print('.', end='')
    return memory


def print_table(methods, results):
    headers = [''] + [f'{n}_setup' for n in methods
                     ] + [f'{n}_search' for n in methods]
    rows = []

    for x in results[0]:
        r = [x] + list(
            map(np.median, [r[x]['setup'] for r in results] +
                [r[x]['search'] for r in results]))
        rows.append(r)

    print(tabulate.tabulate(rows, headers=headers))


def print_table_simple(methods, results):
    headers = [''] + [f'{n}_search' for n in methods]
    rows = []

    for x in results[0]:
        r = [x] + list(map(np.median, [r[x]['search'] for r in results]))
        rows.append(r)

    print(tabulate.tabulate(rows, headers=headers))


def print_table_memory(methods, results):
    headers = [''] + [f'{n}' for n in methods]
    rows = []

    for x in results[0]:
        r = [x] + list(map(np.median, [r[x]['memory'] for r in results]))
        rows.append(r)

    print(tabulate.tabulate(rows, headers=headers))


def sample_points(points, num_sample):
    if points.shape[0] < num_sample:
        num_sample = points.shape[0]

    idx = np.round(np.linspace(0, len(points) - 1, num_sample)).astype(int)
    return points[idx]
