# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DATASETS = ["Enzyme", "GPCRs", "ion_channel", "Davis", "KIBA"]
MODES = ["cold_drug", "cold_target"]
MODELS = ["ArcMDTI", "DrugBAN", "DeepDTA", "MPNN", "MolTrans"]
BASELINE_SCRIPTS = {
    "DeepDTA": "baseline/baseline_DeepDTA.py",
    "MPNN": "baseline/baseline_MPNN.py",
    "MolTrans": "baseline/baseline_MolTrans.py",
    "DrugBAN": "baseline/baseline_DrugBAN.py",
}
ARCMDTI_BATCH_SIZE = "64"


@dataclass(frozen=True)
class ColdTask:
    model: str
    dataset: str
    mode: str

    @property
    def dataset_file(self):
        return f"{self.dataset}.txt"

    @property
    def log_file(self):
        return f"result_cold_{self.model}_{self.mode}_{self.dataset}.log"


def build_tasks(datasets=None, modes=None, models=None):
    datasets = datasets or DATASETS
    modes = modes or MODES
    models = models or MODELS
    return [
        ColdTask(model=model, dataset=dataset, mode=mode)
        for model in models
        for dataset in datasets
        for mode in modes
    ]


def final_summary_path(task, root=None):
    root = Path(root or Path.cwd())
    if task.model == "ArcMDTI":
        p1 = (
            root
            / "result"
            / f"{task.dataset}_{task.mode.upper()}_CV_Seed42"
            / f"{task.dataset}_final_summary.csv"
        )
        p2 = (
            root
            / "result"
            / f"{task.dataset}_{task.mode}_Seed42"
            / f"{task.dataset}_{task.mode}_final_summary.csv"
        )
        return p2 if p2.exists() else p1
    return (
        root
        / "result"
        / f"Baseline_{task.model}_{task.dataset}_{task.mode.upper()}"
        / "Final_Summary.txt"
    )


def is_task_complete(task, root=None):
    marker = final_summary_path(task, root)
    return marker.exists() and marker.stat().st_size > 0


def build_command(task):
    py = sys.executable
    if task.model == "ArcMDTI":
        return [
            py,
            "run_cold_start.py",
            "--dataset",
            task.dataset_file,
            "--split_mode",
            task.mode,
            "--batch_size",
            ARCMDTI_BATCH_SIZE,
        ]

    return [
        py,
        BASELINE_SCRIPTS[task.model],
        task.dataset,
        "--split_mode",
        task.mode,
    ]


def write_console(text):
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    sys.stdout.flush()


def format_progress_bar(done, total, width=36):
    if total <= 0:
        ratio = 1.0
    else:
        ratio = max(0.0, min(1.0, done / total))
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({done}/{total})"


def print_progress(done, total):
    print(f"[OVERALL] {format_progress_bar(done, total)}")


def run_subprocess(command, log_file):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    repo_root = str(Path.cwd())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[RUN] {' '.join(command)}")
    with open(log_file, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n\n" + "=" * 72 + "\n")
        log.write(f"Started at: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Command: {' '.join(command)}\n")
        log.write("=" * 72 + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        try:
            for line in process.stdout:
                write_console(line)
                log.write(line)
                log.flush()
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            log.write("\n[INTERRUPTED] User interrupted this task.\n")
            log.flush()
            raise

        log.write(f"\nFinished at: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Return code: {process.returncode}\n")
        log.flush()
        return process.returncode


def run_task(task, max_retries=2, retry_delay=60, runner=run_subprocess):
    if is_task_complete(task):
        print(f"[SKIP] {task.model} {task.dataset} {task.mode} already completed: {final_summary_path(task)}")
        return 0

    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        print("=" * 72)
        print(f"[TASK] {task.model} | {task.dataset} | {task.mode} | attempt {attempt}/{total_attempts}")
        print("=" * 72)

        return_code = runner(build_command(task), task.log_file)

        if is_task_complete(task):
            print(f"[DONE] {task.model} {task.dataset} {task.mode} completed.")
            return 0

        print(f"[WARN] {task.model} {task.dataset} {task.mode} did not complete. Return code: {return_code}")
        if attempt < total_attempts:
            print(f"[RETRY] Rerunning the same dataset/mode after {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"[FAIL] {task.model} {task.dataset} {task.mode} still incomplete after {total_attempts} attempts.")
            return return_code if return_code != 0 else 1

    return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all ArcMDTI cold-start experiments with resume and retry support."
    )
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=int, default=60)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--model", action="append", choices=MODELS)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = build_tasks(
        datasets=args.dataset or DATASETS,
        modes=args.mode or MODES,
        models=args.model or MODELS,
    )

    print("=" * 72)
    print("ArcMDTI cold-start runner")
    print(f"Tasks: {len(tasks)}")
    print(f"Max retries per task: {args.max_retries}")
    print("=" * 72)
    print_progress(0, len(tasks))

    completed_tasks = 0
    failed_tasks = []
    for index, task in enumerate(tasks, start=1):
        print(f"\n[NEXT] {index}/{len(tasks)} | {task.model} | {task.dataset} | {task.mode}")
        try:
            exit_code = run_task(task, max_retries=args.max_retries, retry_delay=args.retry_delay)
        except KeyboardInterrupt:
            print("\n[STOPPED] Interrupted by user.")
            return 130

        if exit_code != 0:
            print(f"\n[WARN] Task {task.model} | {task.dataset} | {task.mode} failed after retries (exit={exit_code}), continuing to next task...")
            failed_tasks.append(f"{task.model}|{task.dataset}|{task.mode}")
            continue
        completed_tasks += 1
        print_progress(completed_tasks, len(tasks))

    print("\n" + "=" * 72)
    subprocess.call([sys.executable, "summarize_cold_start_baselines.py"])
    if failed_tasks:
        print(f"Completed with {len(failed_tasks)} failed tasks: {failed_tasks}")
    else:
        print("All cold-start tasks completed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

