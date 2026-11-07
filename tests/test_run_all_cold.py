import os
import unittest
from pathlib import Path

import run_all_cold


class RunAllColdTests(unittest.TestCase):
    def test_build_tasks_covers_all_datasets_and_modes(self):
        tasks = run_all_cold.build_tasks()

        self.assertEqual(len(tasks), 50)
        self.assertEqual(tasks[0].model, "ArcMDTI")
        self.assertEqual(tasks[0].dataset, "Enzyme")
        self.assertEqual(tasks[0].mode, "cold_drug")
        self.assertEqual(tasks[1].model, "ArcMDTI")
        self.assertEqual(tasks[1].mode, "cold_target")
        self.assertEqual(tasks[-1].dataset, "KIBA")
        self.assertEqual(tasks[-1].mode, "cold_target")

    def test_arcmdti_commands_use_uniform_batch_size_64(self):
        for dataset in run_all_cold.DATASETS:
            task = run_all_cold.ColdTask(model="ArcMDTI", dataset=dataset, mode="cold_drug")
            command = run_all_cold.build_command(task)
            self.assertEqual(command[-2:], ["--batch_size", "64"])

    def test_task_completion_uses_final_summary_marker(self):
        root = Path(f"/tmp/test_run_all_cold_tmp_{os.getpid()}_arc")
        task = run_all_cold.ColdTask(model="ArcMDTI", dataset="GPCRs", mode="cold_target")
        marker = root / "result" / "GPCRs_COLD_TARGET_CV_Seed42" / "GPCRs_final_summary.csv"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("Stats,AUC\nMean,0.7\n", encoding="utf-8")

        self.assertTrue(run_all_cold.is_task_complete(task, root))

    def test_baseline_completion_uses_final_summary_marker(self):
        root = Path(f"/tmp/test_run_all_cold_tmp_{os.getpid()}_base")
        task = run_all_cold.ColdTask(model="MolTrans", dataset="Davis", mode="cold_drug")
        marker = root / "result" / "Baseline_MolTrans_Davis_COLD_DRUG" / "Final_Summary.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("AUC: 0.8\n", encoding="utf-8")

        self.assertTrue(run_all_cold.is_task_complete(task, root))

    def test_baseline_command_uses_baseline_script(self):
        task = run_all_cold.ColdTask(model="DrugBAN", dataset="Enzyme", mode="cold_target")
        command = run_all_cold.build_command(task)

        self.assertEqual(
            command,
            [
                run_all_cold.sys.executable,
                "baseline/baseline_DrugBAN.py",
                "Enzyme",
                "--split_mode",
                "cold_target",
            ],
        )

    def test_progress_bar_formats_current_progress(self):
        bar = run_all_cold.format_progress_bar(done=25, total=50, width=20)
        self.assertEqual(bar, "[##########----------]  50.0% (25/50)")


if __name__ == "__main__":
    unittest.main()

