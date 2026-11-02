"""Platform-boundary regressions; Blender and GPU rendering are explicit test doubles."""
import contextlib
import ctypes
import errno
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from hairdeck import process
from hairdeck.vis import gl_backend, turn_preview
from hairdeck.vrm import export as vrm


class ProcessTests(unittest.TestCase):
    def test_windows_never_uses_kill_for_liveness(self):
        with patch.object(sys, 'platform', 'win32'), patch.object(process, '_windows_process_alive', return_value=True) as probe, \
                patch.object(os, 'kill', side_effect=AssertionError('must not signal a Windows process')):
            self.assertTrue(process.process_alive(123))
            self.assertFalse(process.process_alive(0))
            self.assertFalse(process.process_alive(-1))
            probe.assert_called_once_with(123)

    def test_windows_handle_probe_closes_handle(self):
        kernel = Mock()
        kernel.OpenProcess.return_value = 1234
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True):
            kernel.WaitForSingleObject.return_value = 258
            self.assertTrue(process._windows_process_alive(123))
            kernel.WaitForSingleObject.return_value = 0
            self.assertFalse(process._windows_process_alive(123))
        kernel.OpenProcess.assert_called_with(0x00100000, False, 123)
        kernel.WaitForSingleObject.assert_called_with(1234, 0)
        self.assertEqual(kernel.CloseHandle.call_count, 2)

    def test_windows_missing_and_inaccessible_pids(self):
        kernel = Mock()
        kernel.OpenProcess.return_value = 0
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
                patch.object(ctypes, 'get_last_error', return_value=87, create=True) as error:
            self.assertFalse(process._windows_process_alive(123))
            error.return_value = 5
            self.assertTrue(process._windows_process_alive(123))
        kernel.WaitForSingleObject.assert_not_called()
        kernel.CloseHandle.assert_not_called()

    def test_windows_failed_wait_still_closes_handle(self):
        kernel = Mock()
        kernel.OpenProcess.return_value = 1234
        kernel.WaitForSingleObject.return_value = 0xFFFFFFFF
        with patch.object(ctypes, 'WinDLL', return_value=kernel, create=True), \
                patch.object(ctypes, 'get_last_error', return_value=6, create=True), \
                patch.object(ctypes, 'WinError', side_effect=lambda n: OSError(n, 'invalid handle'), create=True):
            with self.assertRaises(OSError):
                process._windows_process_alive(123)
        kernel.CloseHandle.assert_called_once_with(1234)

    def test_posix_permission_denied_still_means_alive(self):
        with patch.object(sys, 'platform', 'linux'), patch.object(os, 'kill', side_effect=OSError(errno.EPERM, 'denied')):
            self.assertTrue(process.process_alive(123))

    def test_real_child_liveness_without_terminating_it(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            self.assertTrue(process.process_alive(child.pid))
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=10)
        self.assertFalse(process.process_alive(child.pid))

    def test_platform_launch_flags(self):
        with patch.object(sys, 'platform', 'win32'):
            self.assertEqual(process.background_process_kwargs(), {'creationflags': 0x08000200})
        with patch.object(sys, 'platform', 'linux'):
            self.assertEqual(process.background_process_kwargs(), {'start_new_session': True})


class RenderBackendTests(unittest.TestCase):
    def test_windows_defaults_to_desktop_gl(self):
        with patch.object(sys, 'platform', 'win32'):
            env = gl_backend.render_env({'CUDA_VISIBLE_DEVICES': '2,3'})
        self.assertEqual(env['HAIRDECK_GL'], 'pyglet')
        self.assertNotIn('PYOPENGL_PLATFORM', env)
        self.assertNotIn('EGL_DEVICE_ID', env)

    def test_linux_default_and_explicit_egl_device(self):
        with patch.object(sys, 'platform', 'linux'):
            env = gl_backend.render_env({'CUDA_VISIBLE_DEVICES': '2,3'})
            self.assertEqual(env['PYOPENGL_PLATFORM'], 'egl')
            self.assertEqual(env['EGL_DEVICE_ID'], '2')
            explicit = gl_backend.render_env({'EGL_DEVICE_ID': '4', 'CUDA_VISIBLE_DEVICES': '2'})
        self.assertEqual(explicit['EGL_DEVICE_ID'], '4')

    def test_osmesa_override_clears_gpu_selection(self):
        original = {'HAIRDECK_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'egl', 'EGL_DEVICE_ID': '2'}
        result = gl_backend.render_env(original)
        self.assertEqual(result['PYOPENGL_PLATFORM'], 'osmesa')
        self.assertNotIn('EGL_DEVICE_ID', result)
        self.assertEqual(original['EGL_DEVICE_ID'], '2')  # caller's environment is not mutated

    def test_pyglet_configuration_is_repeatable(self):
        with patch.dict(os.environ, {'HAIRDECK_GL': 'pyglet', 'PYOPENGL_PLATFORM': 'egl', 'EGL_DEVICE_ID': '2'}, clear=True):
            gl_backend.configure_gl()
            gl_backend.configure_gl()
            self.assertNotIn('PYOPENGL_PLATFORM', os.environ)
            self.assertNotIn('EGL_DEVICE_ID', os.environ)

    def test_invalid_backend_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'HAIRDECK_GL'):
            gl_backend.render_env({'HAIRDECK_GL': 'unknown'})


class FFmpegTests(unittest.TestCase):
    def test_configured_executable_wins(self):
        with patch.dict(os.environ, {'HAIRDECK_FFMPEG': 'C:/Program Files/FFmpeg/ffmpeg.exe'}, clear=True), \
                patch.object(turn_preview.shutil, 'which', side_effect=lambda s: s):
            self.assertEqual(turn_preview.ffmpeg_executable(), 'C:/Program Files/FFmpeg/ffmpeg.exe')

    def test_invalid_override_is_not_silently_ignored(self):
        with patch.dict(os.environ, {'HAIRDECK_FFMPEG': 'missing'}, clear=True), \
                patch.object(turn_preview.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'HAIRDECK_FFMPEG'):
                turn_preview.ffmpeg_executable()

    def test_bundled_binary_fallback(self):
        bundled = types.SimpleNamespace(get_ffmpeg_exe=Mock(return_value='/bundle/ffmpeg.exe'))
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {'imageio_ffmpeg': bundled}), \
                patch.object(turn_preview.shutil, 'which', side_effect=lambda s: s if s == '/bundle/ffmpeg.exe' else None):
            self.assertEqual(turn_preview.ffmpeg_executable(), '/bundle/ffmpeg.exe')

    def test_no_binary_has_install_guidance(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {'imageio_ffmpeg': None}), \
                patch.object(turn_preview.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'imageio-ffmpeg'):
                turn_preview.ffmpeg_executable()


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hairdeck-vrm-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'out'
        self.out.mkdir()
        self.body = {'file_tag': 'avatar'}
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'HAIRDECK_ROOT': str(self.root)}).start()
        patch.object(vrm.B, 'body', return_value=self.body).start()
        patch.object(vrm, 'prepare', return_value=(2, 1)).start()
        self.check = patch.object(vrm, 'check', return_value={'ok': True, 'problems': []}).start()
        self.blender = patch.object(vrm, '_blender', side_effect=self.export_stub).start()

    def export_stub(self, prep, directory, body):
        d = Path(directory)
        (d / 'avatar_gen.vrm').write_bytes(b'test-generated-vrm')
        (d / 'avatar_export.json').write_text(json.dumps({'n_new_chains': 1}))
        return 0.1, 'one-shot'

    def run_export(self, **kwargs):
        return vrm.export_vrm(str(self.out), 0, [0.1, 0.2, 0.3], **kwargs)

    def test_preview_failure_preserves_validated_vrm_and_removes_stale_image(self):
        d = self.out / 'vrm_seed0'
        d.mkdir()
        (d / 'rest.png').write_bytes(b'old-preview')

        def fail_render(command, *args):
            self.assertFalse((d / 'rest.png').exists())
            Path(command[3]).write_bytes(b'partial-preview')
            raise RuntimeError('OpenGL context unavailable')

        with patch.object(vrm, '_run', side_effect=fail_render):
            info = self.run_export()
        self.assertTrue(info['ok'])
        self.assertEqual(Path(info['vrm']).read_bytes(), b'test-generated-vrm')
        self.assertIsNone(info['rest_png'])
        self.assertIn('OpenGL context unavailable', info['render_error'])
        self.assertFalse((d / 'rest.png').exists())
        self.assertEqual(json.loads((d / 'vrm_info.json').read_text())['render_error'], info['render_error'])

    def test_success_without_preview_file_is_reported_as_preview_failure(self):
        with patch.object(vrm, '_run', return_value=0.1):
            info = self.run_export()
        self.assertTrue(Path(info['vrm']).is_file())
        self.assertIsNone(info['rest_png'])
        self.assertIn('without writing', info['render_error'])

    def test_successful_preview_is_returned(self):
        from PIL import Image

        def render(command, *args):
            Image.new('RGB', (2, 2), 'white').save(command[3])
            return 0.1

        with patch.object(vrm, '_run', side_effect=render):
            info = self.run_export()
        self.assertTrue(Path(info['rest_png']).is_file())
        self.assertIsNone(info['render_error'])
        self.assertTrue(info['ok'])

    def test_render_disabled_does_not_import_or_run_renderer(self):
        with patch.object(vrm, '_run') as renderer:
            info = self.run_export(render=False)
        renderer.assert_not_called()
        self.assertIsNone(info['rest_png'])
        self.assertIsNone(info['render_error'])

    def test_validation_failures_still_fail_and_save_diagnostics(self):
        self.check.return_value = {'ok': False, 'problems': ['invalid joint index']}
        with patch.object(vrm, '_run') as renderer, self.assertRaisesRegex(RuntimeError, 'invalid joint index'):
            self.run_export()
        renderer.assert_not_called()
        self.assertFalse((self.out / 'avatar_seed0.vrm').exists())
        self.assertFalse(json.loads((self.out / 'vrm_seed0' / 'check.json').read_text())['ok'])

    def test_blender_error_is_not_relabelled_as_optional_preview_error(self):
        self.blender.side_effect = RuntimeError('Blender export failed')
        with self.assertRaisesRegex(RuntimeError, 'Blender export failed'):
            self.run_export()
        self.check.assert_not_called()

    def test_link_permissions_fall_back_to_copy(self):
        src, dst = self.root / 'source', self.root / 'destination'
        src.write_bytes(b'vrm-bytes')
        for symbolic, function in [(True, 'symlink'), (False, 'link')]:
            with self.subTest(symbolic=symbolic), patch.object(vrm.os, function, side_effect=PermissionError('link unavailable')):
                vrm._link_or_copy(str(src), str(dst), symbolic=symbolic)
                self.assertEqual(dst.read_bytes(), src.read_bytes())
                self.assertFalse(dst.is_symlink())
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_worker_probe_uses_safe_shared_helper(self):
        ready = Path(vrm.work_dir('jobs', 'ready'))
        ready.parent.mkdir(parents=True)
        ready.write_text(json.dumps({'pid': 123, 'body': 'body.json'}))
        with patch.object(vrm.B, 'body_path', return_value='body.json'), patch.object(vrm, 'process_alive', return_value=True) as probe:
            self.assertEqual(vrm.worker_pid(), 123)
            probe.assert_called_once_with(123)
        with patch.object(vrm, 'process_alive', return_value=False):
            self.assertIsNone(vrm.worker_pid())

    def test_no_render_cli_exports_asset_without_video(self):
        (self.out / 'info.json').write_text(json.dumps({'samples': [{'seed': 0, 'n_meshed': 2}], 'hair_color': [0.1, 0.2, 0.3]}))
        with patch.object(sys, 'argv', ['export', str(self.out), '0', '--no-render']), \
                patch.object(vrm, 'export_vrm', return_value={'ok': True}) as export, \
                patch.object(vrm, 'turn_preview') as preview, contextlib.redirect_stdout(io.StringIO()):
            vrm.main()
        export.assert_called_once_with(str(self.out), 0, [0.1, 0.2, 0.3], None, render=False)
        preview.assert_not_called()

    def test_no_render_cli_rejects_turn_only(self):
        with patch.object(sys, 'argv', ['export', str(self.out), '0', 'turn', '--no-render']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            vrm.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
