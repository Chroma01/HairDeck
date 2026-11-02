import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_cli_returns_failure_and_records_environment_scope(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'report.json'
            with patch.object(report.sys, 'argv', ['check', '--json', str(output)]), \
                    patch.object(report, 'python_check', return_value='Python 3.12'), \
                    patch.object(report, 'cuda_check', side_effect=RuntimeError('CUDA unavailable')), \
                    patch.object(report, 'dependency_check', return_value='Dependencies imported'):
                self.assertEqual(report.main(), 1)
            saved = json.loads(output.read_text())
            self.assertFalse(saved['ok'])
            self.assertEqual(saved['scope'], 'environment')
            self.assertEqual(len(saved['checks']), 3)


if __name__ == '__main__':
    unittest.main()
