import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import re
from scipy.stats import gmean
from pprint import pprint
import argparse


def to_float(string):
    return float(string.replace(",", ""))


def decode_name(name):
    operand = re.search(r"(binary|unary)", name).group(1)
    op = re.search(r"\[([a-z_A-Z]+)-", name).group(1)
    dtype = re.search(r"(dtype[0-9]+)", name).group(1)
    size = re.search(r"-([0-9]+)", name).group(1)
    engine = "open3d" if re.search(r"numpy", name) is None else "numpy"
    return operand, op, dtype, size, engine


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.0 * height,
                '%d' % int(height),
                ha='center',
                va='bottom')


def parse_benchmark_log_file(log_file):
    decimal_with_parenthesis = r"([0-9\.\,]+) \([^)]*\)"
    regex_dict = {
        "name": r"(\w+\[[^\]]*])",
        "min": decimal_with_parenthesis,
        "max": decimal_with_parenthesis,
        "mean": decimal_with_parenthesis,
        "stddev": decimal_with_parenthesis,
        "median": decimal_with_parenthesis,
        "iqr": None,
        "outliers": None,
        "ops": None,
        "rounds": None,
        "iterations": None,
        "spaces": r"\s+"
    }

    with open(log_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        spaces = r"\s+"
        line_regex = "".join([
            regex_dict["name"],
            spaces,
            regex_dict["min"],
            spaces,
            regex_dict["max"],
            spaces,
            regex_dict["mean"],
            spaces,
            regex_dict["stddev"],
            spaces,
            regex_dict["median"],
        ])
        entries = []
        for line in lines:
            match = re.search(line_regex, line)
            if match:
                entry = dict()
                entry["name"] = match.group(1).strip()
                entry["operand"], entry["op"], entry["dtype"], entry[
                    "size"], entry["engine"] = decode_name(entry["name"])
                entry["min"] = to_float(match.group(2).strip())
                entry["max"] = to_float(match.group(3).strip())
                entry["mean"] = to_float(match.group(4).strip())
                entry["stddev"] = to_float(match.group(5).strip())
                entry["median"] = to_float(match.group(6).strip())
                entries.append(entry)

    print(f"len(entries): {len(entries)}")
    return entries


# Install Intel python
# conda create -n intel-python -c intel intelpython3_core python=3
# python -c "import numpy as np; print(np.show_config())"
# python --version

if __name__ == "__main__":

    log_file = "open3d_0.13.log"
    entries = parse_benchmark_log_file(log_file)

    operands = ["unary", "binary"]
    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        operand = operands[i]
        ax = axes[i]

        # Compute geometirc mean
        times = dict()
        ops = [entry["op"] for entry in entries if entry["operand"] == operand]
        ops = sorted(list(set(ops)))
        for op in ops:
            open3d_times = [
                entry["mean"]
                for entry in entries
                if entry["op"] == op and entry["engine"] == "open3d"
            ]
            numpy_times = [
                entry["mean"]
                for entry in entries
                if entry["op"] == op and entry["engine"] == "numpy"
            ]
            times[op] = dict()
            times[op]["open3d"] = gmean(open3d_times)
            times[op]["numpy"] = gmean(numpy_times)
        pprint(times)

        # Plot
        # https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
        open3d_times = [times[op]["open3d"] for op in ops]
        numpy_times = [times[op]["numpy"] for op in ops]

        ind = np.arange(len(ops))  # the x locations for the groups
        width = 0.35  # the width of the bars

        rects1 = ax.bar(ind, open3d_times, width, color='r')
        rects2 = ax.bar(ind + width, numpy_times, width, color='y')

        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{log_file}: {operand} op benchmarks')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(ops)

        ax.legend((rects1[0], rects2[0]), ('Open3D', 'Numpy'))

        autolabel(rects1)
        autolabel(rects2)

    plt.show()
