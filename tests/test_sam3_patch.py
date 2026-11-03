"""patch_sam3_mps: an incomplete patch must fail, with or without --check."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hairdeck_sam3_patch', ROOT / 'scripts/setup/patch_sam3_mps.py')
patch_sam3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_sam3)

EDIT = [('sam3/model_builder.py', 'old text', 'new text')]


class PatchSam3FailureTests(unittest.TestCase):
    def _main(self, repo, *argv):
        with patch.object(patch_sam3.paths, 'sam3_repo', return_value=repo), \
                patch.object(patch_sam3, 'EDITS', EDIT), \
                patch.object(patch_sam3.sys, 'argv', ['patch_sam3_mps.py', *argv]):
            out = io.StringIO()
            with patch('sys.stdout', out):
                code = patch_sam3.main()
        return code, out.getvalue()

    def test_unmatched_edit_fails_without_check(self):
        with tempfile.TemporaryDirectory() as repo:
            target = Path(repo) / EDIT[0][0]
            target.parent.mkdir(parents=True)
            target.write_text('unrelated text', encoding='utf-8')
            code, out = self._main(repo)
        self.assertEqual(code, 1)
        self.assertIn('UNEXPECTED', out)

    def test_absent_file_fails_without_check(self):
        with tempfile.TemporaryDirectory() as repo:
            code, _ = self._main(repo)
        self.assertEqual(code, 1)

    def test_matching_edit_succeeds_and_is_applied(self):
        with tempfile.TemporaryDirectory() as repo:
            target = Path(repo) / EDIT[0][0]
            target.parent.mkdir(parents=True)
            target.write_text('old text', encoding='utf-8')
            code, out = self._main(repo)
            self.assertEqual(target.read_text(encoding='utf-8'), 'new text')
        self.assertEqual(code, 0)
        self.assertIn('patched', out)


if __name__ == '__main__':
    unittest.main()
