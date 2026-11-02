"""Portable setup regressions; Windows path semantics are exercised without a GPU."""
import hashlib
import importlib.util
import json
import ntpath
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from hairdeck import blender_runtime as runtime
from hairdeck import paths

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('windows_setup_downloader', ROOT / 'scripts/setup/download_weights.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class PathTests(unittest.TestCase):
    def setUp(self):
        env = {'HAIRDECK_ROOT': r'D:\hairdeck workspace', 'HAIRDECK_DATA': r'E:\assets',
               'HAIRDECK_CKPT': r'F:\weights', 'HAIRDECK_WEIGHTS': r'G:\third party'}
        self.win_os = types.SimpleNamespace(path=ntpath, sep='\\', environ=env, fspath=os.fspath)
        self.patches = [patch.object(paths, 'os', self.win_os), patch.object(paths, '_config', return_value={}),
                        patch.object(download, 'os', self.win_os)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_windows_overrides_accept_both_slashes(self):
        for value in ('ckpt/card_vae/model.safetensors', r'ckpt\card_vae\model.safetensors'):
            self.assertEqual(paths.resolve(value), r'F:\weights\card_vae\model.safetensors')
        self.assertEqual(paths.resolve('third_party/weights/sam3/sam3.pt'), r'G:\third party\sam3\sam3.pt')
        self.assertEqual(paths.resolve('results/a'), r'D:\hairdeck workspace\results\a')
        self.assertEqual(paths.resolve('results'), r'D:\hairdeck workspace\results')
        self.assertEqual(paths.resolve(r'C:\inputs\hair.png'), r'C:\inputs\hair.png')

    def test_windows_default_nested_paths_are_native(self):
        self.assertEqual(paths.dinov3_repo(), r'D:\hairdeck workspace\third_party\dinov3_src')

    def test_manifest_mapping_and_inverse_across_drives(self):
        examples = {'ckpt/image/model.safetensors': r'F:\weights\image\model.safetensors',
                    'data/avatar/body.json': r'E:\assets\avatar\body.json',
                    'third_party/weights/sam3/sam3.pt': r'G:\third party\sam3\sam3.pt'}
        for logical, native in examples.items():
            self.assertEqual(download.dst_of(logical), native)
            self.assertEqual(download.rel_of(native), logical)
        self.assertEqual(download.dst_of('config.json'), r'D:\hairdeck workspace\config.json')

    def test_hub_layout_uses_native_suffix(self):
        self.assertEqual(download.hub_local_dir('ckpt/image/model.safetensors',
                         r'D:\workspace\ckpt\image\model.safetensors'), r'D:\workspace')
        self.assertIsNone(download.hub_local_dir('ckpt/image/model.safetensors',
                          r'F:\weights\image\model.safetensors'))

    def test_invalid_manifest_paths_cannot_escape(self):
        for name in ('../secret', '/root/file', r'..\secret', 'C:/file', 'data/file:stream'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.dst_of(name)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'HAIRDECK_ROOT': self.temp.name}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_managed_windows_binary_and_environment(self):
        original = {'PATH': 'tools', 'HOME': 'original', 'DISPLAY': 'old'}
        with patch.object(runtime, 'sys', types.SimpleNamespace(platform='win32')):
            command = runtime.blender_cmd()
            child = runtime.blender_env(original)
        self.assertEqual(Path(command[0]).name, 'blender.exe')
        self.assertIn('blender-4.2.23-windows-x64', command[0])
        self.assertEqual(original, {'PATH': 'tools', 'HOME': 'original', 'DISPLAY': 'old'})
        self.assertEqual(child['PATH'], 'tools')
        self.assertNotIn('DISPLAY', child)
        self.assertEqual(child['USERPROFILE'], child['HOME'])
        self.assertTrue(Path(child['TEMP']).is_dir())
        self.assertEqual(Path(child['BLENDER_USER_EXTENSIONS']).relative_to(self.temp.name).parts,
                         ('third_party', 'blender', 'user', 'extensions'))

    def test_linux_managed_binary_and_legacy_command(self):
        with patch.object(runtime, 'sys', types.SimpleNamespace(platform='linux')):
            self.assertEqual(Path(runtime.blender_cmd()[0]).parts[-2:],
                             ('blender-4.2.23-linux-x64', 'blender'))
            os.environ['HAIRDECK_BLENDER'] = 'bash "/my workspace/bl.sh"'
            self.assertEqual(runtime.blender_cmd(), ['bash', '/my workspace/bl.sh'])

    def test_windows_custom_paths_preserve_spaces_backslashes_and_arguments(self):
        with patch.object(runtime, 'sys', types.SimpleNamespace(platform='win32')):
            for value in (r'C:\Program Files\Blender\blender.exe', r'"C:\Program Files\Blender\blender.exe"'):
                os.environ['HAIRDECK_BLENDER'] = value
                self.assertEqual(runtime.blender_cmd(), [r'C:\Program Files\Blender\blender.exe'])
            os.environ['HAIRDECK_BLENDER'] = r'"C:\Program Files\Blender\blender.exe" --background'
            self.assertEqual(runtime.blender_cmd(), [r'C:\Program Files\Blender\blender.exe', '--background'])
            os.environ['HAIRDECK_BLENDER'] = json.dumps([r'C:\Program Files\Blender\blender.exe', '--factory-startup'])
            self.assertEqual(runtime.blender_cmd(), [r'C:\Program Files\Blender\blender.exe', '--factory-startup'])

    def test_custom_blender_keeps_its_profile(self):
        os.environ['HAIRDECK_BLENDER'] = '/custom/blender'
        original = {'HOME': '/user', 'BLENDER_USER_EXTENSIONS': '/user/addons'}
        self.assertEqual(runtime.blender_env(original), original)
        with patch.object(download, 'url_get') as get, patch.object(download.subprocess, 'run') as run:
            download.get_blender(self.temp.name, types.SimpleNamespace(src=''))
            get.assert_not_called()
            run.assert_not_called()

    def test_bad_command_forms_fail_with_actionable_error(self):
        for value in ('[]', '[null]', '[1]'):
            os.environ['HAIRDECK_BLENDER'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.blender_cmd()
        with patch.object(runtime, 'sys', types.SimpleNamespace(platform='win32')):
            os.environ['HAIRDECK_BLENDER'] = r'C:\Program Files\Blender\blender.exe -b'
            with self.assertRaisesRegex(ValueError, 'Quote'):
                runtime.blender_cmd()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'HAIRDECK_ROOT': str(self.root)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_linux_path_overrides_still_work(self):
        os.environ['HAIRDECK_CKPT'] = str(self.root / 'my weights')
        expected = str(self.root / 'my weights' / 'vae' / 'model.safetensors')
        self.assertEqual(paths.resolve('ckpt/vae/model.safetensors'), expected)
        self.assertEqual(download.dst_of('ckpt/vae/model.safetensors'), expected)
        self.assertEqual(download.rel_of(expected), 'ckpt/vae/model.safetensors')

    def test_modelscope_uses_pinned_mirrors_and_original_weight_hashes(self):
        args = types.SimpleNamespace(src='', gated_source='modelscope')
        with patch.object(download, 'url_get') as get, patch.object(download, 'hub_get') as hub:
            download.get_sam3(str(self.root), args)
            download.get_dinov3(str(self.root), args)
        hub.assert_not_called()
        for call, repo, filename, checksum in (
                (get.call_args_list[0], download.SAM3[0], download.SAM3[1], download.SAM3[2]),
                (get.call_args_list[2], download.DINOV3[0], download.DINOV3[2], download.DINOV3[3])):
            self.assertEqual(call.args[0], f'https://modelscope.cn/api/v1/models/{repo}/repo?'
                             f'Revision={download.MODELSCOPE_REVISIONS[repo]}&FilePath={filename}')
            self.assertEqual(call.args[2], checksum)
        self.assertIn('FilePath=LICENSE', get.call_args_list[1].args[0])
        self.assertIn('FilePath=LICENSE.md', get.call_args_list[3].args[0])

    def test_default_gated_source_keeps_huggingface_revision(self):
        args = types.SimpleNamespace(src='')
        with patch.object(download, 'url_get') as get, patch.object(download, 'hub_get') as hub:
            download.get_dinov3(str(self.root), args)
        get.assert_not_called()
        self.assertEqual(hub.call_args_list[0].args[3], download.DINOV3[3])
        self.assertEqual(hub.call_args_list[0].kwargs['revision'], download.DINOV3[1])

    def test_local_gated_weights_never_contact_mirror(self):
        args = types.SimpleNamespace(src=str(self.root / 'source'), gated_source='modelscope')
        with patch.object(download, 'local_get') as local, patch.object(download, 'gated_get') as remote:
            download.get_sam3(str(self.root), args)
            download.get_dinov3(str(self.root), args)
        self.assertEqual(local.call_count, 2)
        remote.assert_not_called()

    def test_unicode_config_with_or_without_powershell_bom(self):
        config = self.root / 'paths.json'
        workspace = str(self.root / '角色工作区')
        checkpoints = str(self.root / '模型权重')
        os.environ.pop('HAIRDECK_ROOT')
        os.environ['HAIRDECK_CONFIG'] = str(config)
        for encoding in ('utf-8', 'utf-8-sig'):
            config.write_text(json.dumps({'root': workspace, 'ckpt': checkpoints}, ensure_ascii=False), encoding=encoding)
            with self.subTest(encoding=encoding):
                self.assertEqual(paths.get('root'), workspace)
                self.assertEqual(paths.resolve('ckpt/vae/model.safetensors'),
                                 os.path.join(checkpoints, 'vae', 'model.safetensors'))

    def test_hub_override_stages_beside_target_and_preserves_source(self):
        cache = self.root / 'shared-cache.bin'
        cache.write_bytes(b'model')
        target = self.root / 'custom' / 'vae' / 'model.safetensors'
        requested = []

        def fake_hub(repo, filename, revision, local_dir):
            requested.append((filename, local_dir))
            self.assertEqual(Path(local_dir).parent, target.parent)
            return str(cache)

        hub = types.SimpleNamespace(hf_hub_download=fake_hub)
        with patch.dict('sys.modules', {'huggingface_hub': hub}):
            download.hub_get('example/model', 'ckpt/vae/model.safetensors', str(target), hashlib.sha256(b'model').hexdigest())
        self.assertEqual(target.read_bytes(), b'model')
        self.assertEqual(cache.read_bytes(), b'model')
        self.assertFalse(Path(requested[0][1]).exists())

    def test_windows_blender_download_uses_zip_checksum_and_isolated_exe(self):
        build = runtime.BUILDS['win32']
        downloaded = []

        def get(url, dst, want):
            downloaded.append((url, want))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            if url == build[0]:
                with zipfile.ZipFile(dst, 'w') as archive:
                    archive.writestr(build[2] + '/blender.exe', b'executable')
            else:
                Path(dst).write_bytes(b'addon')
            return dst

        def run(cmd, **kwargs):
            self.assertEqual(Path(cmd[0]).name, 'blender.exe')
            self.assertEqual(cmd[1:4], ['--command', 'extension', 'install-file'])
            addon = Path(kwargs['env']['BLENDER_USER_EXTENSIONS']) / 'user_default' / 'vrm'
            addon.mkdir(parents=True)
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')

        with patch.object(download, 'blender_build', return_value=build), patch.object(download, 'url_get', get), \
                patch.object(download.subprocess, 'run', run), \
                patch.object(runtime, 'sys', types.SimpleNamespace(platform='win32')):
            download.get_blender(str(self.root), types.SimpleNamespace(src=''))
        self.assertEqual(downloaded[0], (build[0], '82e791475779a7342424a480bdde9a20b43710da9264c60346125aa16cd910cb'))
        self.assertTrue((self.root / 'third_party/blender' / build[2] / 'blender.exe').is_file())
        self.assertFalse((self.root / 'third_party/blender/bl.sh').exists())

    def test_installed_linux_blender_keeps_legacy_wrapper(self):
        build = runtime.BUILDS['linux']
        base = self.root / 'third_party/blender'
        binary = base / build[2] / 'blender'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'blender')
        (base / 'user/extensions/user_default/vrm').mkdir(parents=True)
        with patch.object(download, 'blender_build', return_value=build), patch.object(download, 'url_get') as get:
            download.get_blender(str(self.root), types.SimpleNamespace(src=''))
            get.assert_not_called()
        self.assertEqual((base / 'bl.sh').read_bytes(), (ROOT / 'scripts/setup/blender_wrapper.sh').read_bytes())

    def test_zip_rejects_traversal_before_writing_any_member(self):
        for bad in ('../escape', '/absolute', r'..\escape', 'C:/escape', 'top/file:stream'):
            archive = self.root / 'bad.zip'
            output = self.root / 'extract'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('safe/first', b'first')
                z.writestr(bad, b'bad')
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                download.extract_zip(archive, output)
            self.assertFalse(output.exists())

    def test_zip_rejects_symbolic_link(self):
        archive = self.root / 'link.zip'
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(info, '../outside')
        with self.assertRaisesRegex(ValueError, 'unsafe ZIP'):
            download.extract_zip(archive, self.root / 'out')


if __name__ == '__main__':
    unittest.main()
