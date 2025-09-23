#!/usr/bin/env python3
import csv
import os
import sys


def get_tasks(benchmark_name):
    # Mirror imports used by other scripts that rely on the installed LIBERO package
    from libero.libero import benchmark

    # Instantiate the requested benchmark and return (name, language) pairs
    bm_cls = benchmark.get_benchmark(benchmark_name)
    bm = bm_cls()
    return [(t.name, t.language) for t in bm.tasks]


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "language_instruction"])
        for name, lang in rows:
            writer.writerow([name, lang])


def main():
    # Collect tasks for LIBERO-Object and LIBERO-90
    obj_rows = get_tasks("libero_object")
    n90_rows = get_tasks("libero_90")
    # Test split (10 holdout tasks for LIBERO-90)
    n90_test_rows = get_tasks("libero_10")

    # Write CSVs in the repo root by default
    write_csv(obj_rows, os.path.join(os.getcwd(), "libero_object_tasks.csv"))
    write_csv(n90_rows, os.path.join(os.getcwd(), "libero_90_tasks.csv"))
    write_csv(n90_test_rows, os.path.join(os.getcwd(), "libero_90_test_tasks.csv"))

    print("Wrote:")
    print(" - libero_object_tasks.csv ({} tasks)".format(len(obj_rows)))
    print(" - libero_90_tasks.csv ({} tasks)".format(len(n90_rows)))
    print(" - libero_90_test_tasks.csv ({} tasks)".format(len(n90_test_rows)))


if __name__ == "__main__":
    main()
