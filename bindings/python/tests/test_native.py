import os
import subprocess
import sys
import unittest

import mlsr


class NativeIntegrationTests(unittest.TestCase):
    def test_loaded_rust_library_matches_package(self):
        self.assertEqual(mlsr.abi_version(), 1)
        self.assertEqual(
            mlsr.native_version(),
            os.environ.get("MLSR_EXPECTED_VERSION", mlsr.__version__),
        )

    def test_import_is_lazy_and_bad_library_has_an_actionable_error(self):
        environment = dict(os.environ, MLSR_LIBRARY_PATH="/missing/mlsr/library")
        script = """
import mlsr
try:
    mlsr.native_version()
except mlsr.LibraryLoadError as error:
    assert '/missing/mlsr/library' in str(error)
else:
    raise AssertionError('an explicit missing library must fail')
"""
        subprocess.run([sys.executable, "-c", script], env=environment, check=True)


if __name__ == "__main__":
    unittest.main()
