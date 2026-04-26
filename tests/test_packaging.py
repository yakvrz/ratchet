from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


class PackagingConfigTests(unittest.TestCase):
    def test_distribution_packages_only_ratchet_library_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = tomllib.loads((root / "pyproject.toml").read_text())
        package_find = payload["tool"]["setuptools"]["packages"]["find"]

        self.assertEqual(package_find["include"], ["ratchet*"])
        self.assertNotIn("package-data", payload["tool"]["setuptools"])


if __name__ == "__main__":
    unittest.main()
