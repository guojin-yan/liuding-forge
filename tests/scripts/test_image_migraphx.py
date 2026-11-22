"""Development-only argument and independent identity contracts for the example."""
import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path


class Arguments(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([str(EXAMPLE), *map(str, args)], capture_output=True,
                              text=True, timeout=10)

    def test_help(self):
        result = self.invoke('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('pack SOURCE_BUNDLE', 'sign UNSIGNED', 'evaluate BUNDLE128'):
            self.assertIn(command, result.stdout)

    def test_required_arguments(self):
        for args in ((), ('unknown',), ('pack',), ('sign',), ('evaluate',)):
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('Image MIGraphX example failed:', result.stderr)

    def test_batch_rejected_before_source_or_device_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            for batch in ('0', '1', '2', '4', '8', '127', '129', '-16', '16x', '016'):
                with self.subTest(batch=batch):
                    result = self.invoke('pack', 'missing-source', '0' * 64, batch, output)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn('Use fixed batch 128 or 16 explicitly', result.stderr)
                    self.assertFalse(output.exists())

    def test_missing_policy_cannot_create_evaluation_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            item = [Path(temporary) / 'bundle', '0' * 64,
                    Path(temporary) / 'policy.json', '0' * 64]
            result = self.invoke('evaluate', *item, *item, temporary, output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=Path, required=True)
    EXAMPLE = parser.parse_args().example.resolve()
    unittest.main(argv=['test_image_migraphx'])
