"""Residency lifecycle tests with fake CUDA; real GPU results live in the test log."""
from contextlib import nullcontext
import os
import types
import unittest
from unittest.mock import Mock, patch

from hairdeck.infer import memory


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.cuda = Mock()
        self.cuda.mem_get_info.return_value = (20 * memory.GIB, 24 * memory.GIB)
        self.cuda.memory_allocated.return_value = 0
        self.cuda.memory_reserved.return_value = 0
        self.cuda.device.side_effect = lambda device: nullcontext()
        patcher = patch.object(memory.torch, 'cuda', self.cuda)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_small_gpu_starts_models_on_cpu(self):
        self.cuda.mem_get_info.return_value = (7 * memory.GIB, 8 * memory.GIB)
        policy = memory.ModelResidency(mode='auto')
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.initial_device, 'cpu')

    def test_available_memory_matters_on_large_gpu(self):
        self.cuda.mem_get_info.return_value = (8 * memory.GIB, 24 * memory.GIB)
        self.assertTrue(memory.ModelResidency(mode='auto').enabled)

    def test_new_pressure_switches_resident_pipeline_at_next_request(self):
        policy = memory.ModelResidency(mode='auto')
        first, second = Mock(), Mock()
        policy.register('sam3', first)
        policy.register('matte', second)
        self.assertFalse(policy.enabled)
        self.cuda.mem_get_info.return_value = (2 * memory.GIB, 24 * memory.GIB)
        with policy.request():
            policy.activate('sam3')
            second.to.assert_called_once_with('cpu')
            first.to.assert_not_called()  # already resident
        first.to.assert_called_once_with('cpu')
        self.assertTrue(policy.enabled)
        self.cuda.mem_get_info.return_value = (20 * memory.GIB, 24 * memory.GIB)
        with policy.request():
            self.assertTrue(policy.enabled)  # stable policy after pressure, no oscillation

    def test_unused_allocator_cache_counts_as_available(self):
        policy = memory.ModelResidency(mode='auto')
        self.cuda.mem_get_info.return_value = (2 * memory.GIB, 24 * memory.GIB)
        self.cuda.memory_allocated.return_value = 6 * memory.GIB
        self.cuda.memory_reserved.return_value = 9 * memory.GIB
        with policy.request():
            self.assertFalse(policy.enabled)

    def test_off_and_zerogpu_do_not_probe_or_move_models(self):
        for options in ({'mode': 'off'}, {'mode': 'on', 'zero_gpu': True}, {'mode': 'auto', 'device': 'cpu'}):
            with self.subTest(options=options):
                policy = memory.ModelResidency(**options)
                model = Mock()
                policy.register('sam3', model)
                with policy.request():
                    policy.activate('sam3')
                model.to.assert_not_called()
        self.cuda.mem_get_info.assert_not_called()
        self.cuda.empty_cache.assert_not_called()

    def test_platform_defaults_preserve_linux_and_allow_override(self):
        with patch.object(memory.sys, 'platform', 'linux'):
            self.assertEqual(memory.ModelResidency().mode, 'off')
            with patch.dict(os.environ, {'HAIRDECK_OFFLOAD': 'on'}):
                self.assertTrue(memory.ModelResidency().enabled)
        with patch.object(memory.sys, 'platform', 'win32'):
            self.assertEqual(memory.ModelResidency().mode, 'auto')

    def test_query_failure_is_conservative_and_invalid_option_is_actionable(self):
        self.cuda.mem_get_info.side_effect = RuntimeError('query unavailable')
        policy = memory.ModelResidency(mode='auto')
        self.assertTrue(policy.enabled)
        policy.register('sam3', Mock())
        with policy.request():
            policy.activate('sam3')
        self.assertEqual(policy.on_gpu, set())
        with self.assertRaisesRegex(ValueError, 'HAIRDECK_OFFLOAD'):
            memory.ModelResidency(mode='wrong')

    def test_eviction_precedes_loading_and_same_stage_is_not_reloaded(self):
        policy = memory.ModelResidency(mode='on')
        calls = []
        first = Mock(to=lambda device: calls.append(('sam3', str(device))))
        second = Mock(to=lambda device: calls.append(('matte', str(device))))
        policy.register('sam3', first)
        policy.register('matte', second)
        calls.clear()
        with policy.request():
            policy.activate('sam3')
            policy.activate('sam3')
            policy.activate('matte')
        self.assertEqual(calls, [('sam3', 'cuda'), ('sam3', 'cpu'), ('matte', 'cuda'), ('matte', 'cpu')])
        self.assertEqual(policy.on_gpu, set())

    def test_partial_upload_failure_is_evicted_and_next_request_can_run(self):
        policy = memory.ModelResidency(mode='on')
        model = Mock()
        policy.register('sam3', model)
        model.to.side_effect = lambda device: (_ for _ in ()).throw(RuntimeError('upload failed')) if str(device) == 'cuda' else None
        with self.assertRaisesRegex(RuntimeError, 'upload failed'), policy.request():
            policy.activate('sam3')
        self.assertEqual(policy.on_gpu, set())
        self.assertEqual(model.to.call_args.args, ('cpu',))
        model.to.side_effect = None
        with policy.request():
            policy.activate('sam3')
        self.assertEqual(policy.on_gpu, set())

    def test_pipeline_exception_also_releases_models(self):
        policy = memory.ModelResidency(mode='on')
        model = Mock()
        policy.register('sam3', model)

        @memory.managed_request
        def failing(owner):
            owner.residency.activate('sam3')
            raise ValueError('bad input')

        with self.assertRaisesRegex(ValueError, 'bad input'):
            failing(types.SimpleNamespace(residency=policy))
        self.assertEqual(policy.on_gpu, set())
        self.assertEqual(model.to.call_args.args, ('cpu',))

    def test_transfers_inside_inference_keep_normal_tensor_lifetime(self):
        tensor = memory.torch.ones(2)
        copied = []
        model = Mock(to=lambda device: copied.append(tensor.clone()))
        with memory.torch.inference_mode():
            memory.move_model(model, 'cpu')
        self.assertFalse(copied[0].is_inference())
        self.assertEqual(copied[0]._version, 0)


if __name__ == '__main__':
    unittest.main()
