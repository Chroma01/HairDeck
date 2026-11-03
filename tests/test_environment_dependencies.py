"""Dependency diagnostics: required packages fail, optional features remain optional."""
import builtins
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'scripts/check_environment.py'
spec = importlib.util.spec_from_file_location('hairdeck_dependency_report', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
PLATFORMS = ('linux', 'win32', 'darwin')


@contextmanager
def environment(platform, missing=(), triton_error=None):
    real_import = builtins.__import__

    def version(name):
        if name in missing:
            raise report.importlib.metadata.PackageNotFoundError(name)
        return '1.0'

    def import_module(name, *args, **kwargs):
        if name == 'triton':
            if triton_error is not None:
                raise triton_error
            return types.SimpleNamespace(__version__='3.4.0')
        return real_import(name, *args, **kwargs)

    with patch.object(report.sys, 'platform', platform), \
            patch.object(report.importlib.metadata, 'version', side_effect=version), \
            patch('builtins.__import__', side_effect=import_module):
        yield


class DependencyTests(unittest.TestCase):
    def check(self):
        return report.run_check('Dependencies', report.dependency_check)

    def test_complete_environment_passes_on_each_platform(self):
        for platform in PLATFORMS:
            with self.subTest(platform=platform), environment(platform):
                self.assertTrue(self.check()['ok'])

    def test_missing_core_package_fails_on_each_platform(self):
        for platform in PLATFORMS:
            for name in ('numpy', 'transformers', 'huggingface-hub'):
                with self.subTest(platform=platform, package=name), environment(platform, missing=(name,)):
                    result = self.check()
                    self.assertFalse(result['ok'])
                    self.assertIn(name, result['detail'])

    def test_optional_demo_and_render_packages_may_be_absent(self):
        optional = ('gradio', 'pyrender', 'PyOpenGL')
        for platform in PLATFORMS:
            with self.subTest(platform=platform), environment(platform, missing=optional):
                result = self.check()
                self.assertTrue(result['ok'])
                for name in optional:
                    self.assertIn(f'{name} absent', result['detail'])

    def test_missing_triton_is_allowed_only_on_macos(self):
        for platform in PLATFORMS:
            error = ModuleNotFoundError("No module named 'triton'", name='triton')
            with self.subTest(platform=platform), environment(platform, triton_error=error):
                result = self.check()
                self.assertEqual(result['ok'], platform == 'darwin')
                self.assertIn('triton', result['detail'])

    def test_broken_triton_import_fails_even_on_macos(self):
        for platform in PLATFORMS:
            for name in ('packaging', 'triton.language'):
                error = ModuleNotFoundError(f"No module named '{name}'", name=name)
                with self.subTest(platform=platform, dependency=name), environment(platform, triton_error=error):
                    result = self.check()
                    self.assertFalse(result['ok'])
                    self.assertIn(name, result['detail'])

    def test_missing_dependency_sets_cli_exit_and_json_failure(self):
        for platform in PLATFORMS:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'report.json'
                with environment(platform, missing=('transformers',)), \
                        patch.object(report.sys, 'argv', ['check', '--json', str(output)]), \
                        patch.object(report, 'python_check', return_value='Python ready'), \
                        patch.object(report, 'device_check', return_value='Device ready'), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(report.main(), 1)
                saved = json.loads(output.read_text())
                self.assertFalse(saved['ok'])
                dependency = next(check for check in saved['checks'] if check['name'] == 'Dependencies')
                self.assertFalse(dependency['ok'])
                self.assertIn('transformers', dependency['detail'])


if __name__ == '__main__':
    unittest.main()
