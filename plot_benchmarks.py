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

    old_log_file = "open3d_0.13.log"
    new_log_file = "open3d_0.15.log"
    old_entries = parse_benchmark_log_file(old_log_file)
    new_entries = parse_benchmark_log_file(new_log_file)

    operands = ["unary", "binary"]
    fig, axes = plt.subplots(2, 1)

    for index, operand in enumerate(["unary", "binary"]):
        ax = axes[index]

        # Get ops, e.g. "add", '"mul"
        old_ops = [
            old_entry["op"]
            for old_entry in old_entries
            if old_entry["operand"] == operand
        ]
        new_ops = [
            new_entry["op"]
            for new_entry in new_entries
            if new_entry["operand"] == operand
        ]
        old_ops = sorted(list(set(old_ops)))
        new_ops = sorted(list(set(new_ops)))
        assert old_ops == new_ops
        ops = old_ops

        # Compute geometirc mean
        old_gmean_times = dict()
        new_gmean_times = dict()
        for op in ops:
            old_times = [
                old_entry["mean"]
                for old_entry in old_entries
                if old_entry["op"] == op and old_entry["engine"] == "open3d"
            ]
            new_times = [
                new_entry["mean"]
                for new_entry in new_entries
                if new_entry["op"] == op and new_entry["engine"] == "open3d"
            ]
            old_gmean_times[op] = gmean(old_times)
            new_gmean_times[op] = gmean(new_times)
        pprint(old_gmean_times)
        pprint(new_gmean_times)

        # Plot
        # https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
        old_gmean_times = [old_gmean_times[op] for op in ops]
        new_gmean_times = [new_gmean_times[op] for op in ops]

        ind = np.arange(len(ops))  # the x locations for the groups
        width = 0.35  # the width of the bars

        rects1 = ax.bar(ind, old_gmean_times, width, color='r')
        rects2 = ax.bar(ind + width, new_gmean_times, width, color='y')

        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{operand} op benchmarks')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(ops)

        ax.legend((rects1[0], rects2[0]), ("old", "new"))

        autolabel(rects1)
        autolabel(rects2)

    plt.show()
