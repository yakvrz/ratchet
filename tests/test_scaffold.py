from __future__ import annotations

from pathlib import Path
import py_compile
import tempfile
import unittest

from ratchet.config import load_run_config
from ratchet.scaffold import init_scaffold


class ScaffoldTests(unittest.TestCase):
    def test_init_python_function_scaffold_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "function-agent", template="python_function")
            self.assertTrue((root / "ratchet_adapter.py").exists())
            self.assertTrue((root / "agent.py").exists())
            self.assertTrue((root / "ratchet.toml").exists())
            self.assertTrue((root / "evals.sample.jsonl").exists())
            self.assertTrue((root / "README.md").exists())
            py_compile.compile(str(root / "ratchet_adapter.py"), doraise=True)
            py_compile.compile(str(root / "agent.py"), doraise=True)

    def test_init_python_cli_scaffold_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "cli-agent", template="python_cli")
            self.assertTrue((root / "ratchet_adapter.py").exists())
            self.assertTrue((root / "agent_cli.py").exists())
            self.assertTrue((root / "ratchet.toml").exists())
            self.assertTrue((root / "evals.sample.jsonl").exists())
            py_compile.compile(str(root / "ratchet_adapter.py"), doraise=True)
            py_compile.compile(str(root / "agent_cli.py"), doraise=True)

    def test_generated_config_loads_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "function-agent", template="python_function")
            config = load_run_config(root / "ratchet.toml")
            self.assertEqual(config.adapter, "ratchet_adapter:adapter")
            self.assertEqual(config.evals, (root / "evals.sample.jsonl").resolve())
            self.assertEqual(config.out, (root / "results" / "run").resolve())


if __name__ == "__main__":
    unittest.main()
