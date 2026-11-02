import importlib.util
import ntpath
from pathlib import Path
import posixpath
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/image_to_hair.py'
spec = importlib.util.spec_from_file_location('hairdeck_windows_cli', path)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ImageNamesTests(unittest.TestCase):
    def test_windows_case_alias_is_rejected_before_loading_models(self):
        with patch.object(cli.sys, 'argv', ['image_to_hair.py', 'a/Portrait.png', 'b/portrait.jpg', '--out', 'results']), \
                patch.object(cli.os.path, 'normcase', ntpath.normcase), patch.object(cli, 'Pipeline') as model:
            with self.assertRaisesRegex(SystemExit, 'same name'):
                cli.main()
            model.assert_not_called()

    def test_posix_case_distinction_is_preserved(self):
        with patch.object(cli.sys, 'argv', ['image_to_hair.py', 'a/Portrait.png', 'b/portrait.jpg', '--out', 'results']), \
                patch.object(cli.os.path, 'normcase', posixpath.normcase), \
                patch.object(cli, 'Pipeline', side_effect=RuntimeError('model loading reached')):
            with self.assertRaisesRegex(RuntimeError, 'model loading reached'):
                cli.main()


if __name__ == '__main__':
    unittest.main()
