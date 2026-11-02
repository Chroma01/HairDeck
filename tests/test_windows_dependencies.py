"""Dependency-error routing without loading SAM 3 weights or running CUDA.

These tests simulate SAM 3 dependency import failures.
"""
import builtins
import tempfile
import unittest
from unittest import mock

from hairdeck.prep import sam3_ground


class Sam3ImportErrors(unittest.TestCase):
    def _fail_import(self, target, error):
        real_import = builtins.__import__

        def import_module(name, *args, **kwargs):
            # Stand in for the package itself so no local SAM 3 installation or
            # downloaded source is needed to reach the component imports.
            if name == target:
                raise error
            if name.startswith('sam3'):
                return mock.Mock()
            return real_import(name, *args, **kwargs)

        return mock.patch('builtins.__import__', side_effect=import_module)

    def test_windows_triton_missing_during_component_import(self):
        error = ModuleNotFoundError("No module named 'triton'", name='triton')
        with mock.patch.object(sam3_ground.sys, 'platform', 'win32'), \
                self._fail_import('sam3.model_builder', error):
            with self.assertRaisesRegex(ImportError, 'requirements-windows.txt') as caught:
                sam3_ground._sam3()
        self.assertIs(caught.exception.__cause__, error)

    def test_windows_triton_missing_during_package_import(self):
        error = ModuleNotFoundError("No module named 'triton.language'", name='triton.language')
        with mock.patch.object(sam3_ground.sys, 'platform', 'win32'), \
                self._fail_import('sam3', error), mock.patch.object(sam3_ground.paths, 'sam3_repo') as repo:
            with self.assertRaisesRegex(ImportError, 'compilation disabled') as caught:
                sam3_ground._sam3()
        repo.assert_not_called()
        self.assertIs(caught.exception.__cause__, error)

    def test_linux_keeps_original_triton_error(self):
        error = ModuleNotFoundError("No module named 'triton'", name='triton')
        with mock.patch.object(sam3_ground.sys, 'platform', 'linux'), \
                self._fail_import('sam3.model_builder', error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                sam3_ground._sam3()
        self.assertIs(caught.exception, error)

    def test_unrelated_package_dependency_is_not_hidden_by_source_fallback(self):
        error = ModuleNotFoundError("No module named 'timm'", name='timm')
        with mock.patch.object(sam3_ground.sys, 'platform', 'win32'), \
                self._fail_import('sam3', error), mock.patch.object(sam3_ground.paths, 'sam3_repo') as repo:
            with self.assertRaises(ModuleNotFoundError) as caught:
                sam3_ground._sam3()
        repo.assert_not_called()
        self.assertIs(caught.exception, error)

    def test_unrelated_component_import_error_is_preserved(self):
        error = ImportError('incompatible SAM 3 component')
        with mock.patch.object(sam3_ground.sys, 'platform', 'win32'), \
                self._fail_import('sam3.model_builder', error):
            with self.assertRaises(ImportError) as caught:
                sam3_ground._sam3()
        self.assertIs(caught.exception, error)

    def test_missing_sam3_source_has_download_guidance(self):
        error = ModuleNotFoundError("No module named 'sam3'", name='sam3')
        with tempfile.TemporaryDirectory() as temp, self._fail_import('sam3', error), \
                mock.patch.object(sam3_ground.paths, 'sam3_repo', return_value=temp + '/missing'):
            with self.assertRaisesRegex(ImportError, '--only sam3_src'):
                sam3_ground._sam3()

    def test_triton_error_after_local_source_fallback_is_reported(self):
        real_import = builtins.__import__
        error = ModuleNotFoundError("No module named 'triton'", name='triton')

        def import_module(name, *args, **kwargs):
            if name == 'sam3':
                raise ModuleNotFoundError("No module named 'sam3'", name='sam3')
            if name == 'sam3.model_builder':
                raise error
            if name.startswith('sam3.'):
                return mock.Mock()
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(sam3_ground.sys, 'platform', 'win32'), \
                mock.patch.object(sam3_ground.paths, 'sam3_repo', return_value=temp), \
                mock.patch.object(sam3_ground.sys, 'path', list(sam3_ground.sys.path)), \
                mock.patch('builtins.__import__', side_effect=import_module):
            with self.assertRaisesRegex(ImportError, 'requirements-windows.txt') as caught:
                sam3_ground._sam3()
            self.assertEqual(sam3_ground.sys.path[0], temp)
        self.assertIs(caught.exception.__cause__, error)


if __name__ == '__main__':
    unittest.main()
