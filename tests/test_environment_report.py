import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch
import torchvision  # noqa: F401  (import before the torch.ones patch below)

path = Path(__file__).resolve().parents[1] / 'scripts/check_environment.py'
spec = importlib.util.spec_from_file_location('hairdeck_environment_report', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class EnvironmentReportTests(unittest.TestCase):
    def test_missing_requirement_is_failure_not_warning(self):
        def missing():
            raise ImportError('No module named triton')
        result = report.run_check('SAM 3', missing)
        self.assertFalse(result['ok'])
        self.assertIn('ImportError', result['detail'])

    def test_cuda_device_report_uses_separate_name_and_memory(self):
        props = types.SimpleNamespace(name='Fake GPU', total_memory=80 * 2 ** 30)
        with patch('hairdeck.dev.select', return_value=torch.device('cuda')), \
                patch('hairdeck.dev.sync'), \
                patch.object(torch, 'ones', return_value=torch.ones((16, 16))), \
                patch.object(torch.cuda, 'current_device', return_value=0), \
                patch.object(torch.cuda, 'get_device_properties', return_value=props):
            detail = report.device_check()
        self.assertIn('cuda: Fake GPU, 80.0 GiB', detail)

    def test_cli_returns_failure_and_records_environment_scope(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'report.json'
            with patch.object(report.sys, 'argv', ['check', '--json', str(output)]), \
                    patch.object(report, 'python_check', return_value='Python 3.12'), \
                    patch.object(report, 'device_check', side_effect=RuntimeError('no CUDA or MPS device')), \
                    patch.object(report, 'dependency_check', return_value='Dependencies imported'):
                self.assertEqual(report.main(), 1)
            saved = json.loads(output.read_text())
            self.assertFalse(saved['ok'])
            self.assertEqual(saved['scope'], 'environment')
            self.assertEqual(len(saved['checks']), 3)


if __name__ == '__main__':
    unittest.main()
