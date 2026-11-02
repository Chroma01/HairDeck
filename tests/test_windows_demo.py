"""A failed optional preview must not hide a valid VRM download."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DemoExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'HAIRDECK_ROOT': cls.workspace.name, 'GRADIO_ANALYTICS_ENABLED': 'False'})
        cls.env.start()
        path = Path(__file__).resolve().parents[1] / 'demo/app.py'
        spec = importlib.util.spec_from_file_location('hairdeck_windows_test_demo', path)
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.workspace.cleanup()

    def test_render_failure_keeps_download_and_explains_missing_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d / 'info.json').write_text(json.dumps({'hair_color': [0.2, 0.3, 0.4]}))
            output = str(d / 'avatar_seed0.vrm')
            info = dict(ok=True, problems=[], vrm=output, rest_png=None, render_error='OpenGL unavailable',
                        timings={}, size_mb=1)
            state = [dict(seed=0, out_dir=temp)]
            with patch.object(self.app, 'ensure_chains'), patch.object(self.app.VX, 'vrm_path', return_value=output), \
                    patch.object(self.app.VX, 'export_vrm', return_value=info):
                result = self.app.make_vrm(state, 0)
            self.assertEqual(result[0], output)
            self.assertIsNone(result[1])
            self.assertIn('ready to download', result[2])
            self.assertNotIn('OpenGL', result[2])
            self.assertIn(output, result[3])

    def test_invalid_old_export_is_regenerated(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d / 'info.json').write_text(json.dumps({'hair_color': [0.2, 0.3, 0.4]}))
            (d / 'vrm_seed0').mkdir()
            (d / 'vrm_seed0/vrm_info.json').write_text(json.dumps({'ok': False}))
            output = d / 'avatar_seed0.vrm'
            output.write_bytes(b'old')
            info = dict(ok=True, problems=[], vrm=str(output), rest_png=None, timings={}, size_mb=1)
            with patch.object(self.app, 'ensure_chains'), patch.object(self.app.VX, 'vrm_path', return_value=str(output)), \
                    patch.object(self.app.VX, 'export_vrm', return_value=info) as export:
                result = self.app.make_vrm([dict(seed=0, out_dir=temp)], 0)
            export.assert_called_once()
            self.assertEqual(result[0], str(output))


if __name__ == '__main__':
    unittest.main()
