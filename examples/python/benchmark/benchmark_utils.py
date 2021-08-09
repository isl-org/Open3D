import os
import platform
import subprocess

import numpy as np
import tabulate


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


def measure_time(fn, min_samples=10, max_samples=100, max_time_in_sec=100.0):
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


def print_table(method_names, results):
    headers = [''] + [f'{n}_setup' for n in method_names
                     ] + [f'{n}_search' for n in method_names]
    rows = []

    for x in results[0]:
        r = [x] + list(
            map(np.median, [r[x]['knn_gpu_setup'] for r in results] +
                [r[x]['knn_gpu_search'] for r in results]))
        rows.append(r)

    print(tabulate.tabulate(rows, headers=headers))
