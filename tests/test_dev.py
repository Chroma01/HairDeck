"""hairdeck.dev: device selection, MPS-safe autocast/noise, and one bf16 forward of the image set DiT."""
import os
import unittest
from unittest.mock import patch

import torch

from hairdeck import dev
from hairdeck.models.imgmodel import ImgSetDiT

HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


class SelectTests(unittest.TestCase):
    def test_hairdeck_device_is_honoured(self):
        with patch.dict(os.environ, {'HAIRDECK_DEVICE': 'cpu'}):
            self.assertEqual(dev.select().type, 'cpu')

    def test_favour_prefers_cuda_then_mps(self):
        with patch.dict(os.environ, {}, clear=True):
            best = 'cuda' if HAS_CUDA else 'mps' if HAS_MPS else 'cpu'
            self.assertEqual(dev.select().type, best)


class AmpTests(unittest.TestCase):
    def test_cpu_disables_autocast(self):
        with dev.amp('cpu'):
            self.assertFalse(torch.is_autocast_enabled())

    def test_disabled_amp_keeps_the_target_device(self):
        for kind in ('cuda', 'mps'):
            with self.subTest(kind=kind), patch.object(dev.torch, 'autocast') as autocast:
                dev.amp(kind, enabled=False)
                autocast.assert_called_once_with(kind, dtype=torch.bfloat16, enabled=False)

    @unittest.skipUnless(HAS_CUDA or HAS_MPS, 'no CUDA or Apple MPS device')
    def test_nested_amp_disables_on_the_target_device_and_restores_it(self):
        device = torch.device('cuda' if HAS_CUDA else 'mps')
        with dev.amp(device):
            self.assertTrue(torch.is_autocast_enabled(device.type))
            with dev.amp(device, enabled=False):
                self.assertFalse(torch.is_autocast_enabled(device.type))
                layer = torch.nn.Linear(4, 2).to(device)
                self.assertEqual(layer(torch.randn(2, 4, device=device)).dtype, torch.float32)
            self.assertTrue(torch.is_autocast_enabled(device.type))

    @unittest.skipUnless(HAS_MPS, 'no Apple MPS device')
    def test_seeded_noise_is_repeatable_on_mps(self):
        a = dev.randn((4, 8), 'mps', dev.generator('mps', 3))
        b = dev.randn((4, 8), 'mps', dev.generator('mps', 3))
        self.assertEqual(a.device.type, 'mps')
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(dev.generator('mps', 3).device.type, 'cpu')

    def test_peak_gb_is_zero_without_a_cuda_counter(self):
        self.assertEqual(dev.peak_gb('cpu'), 0.0)


class ImgSetDiTForwardTests(unittest.TestCase):
    def test_bf16_forward_on_the_selected_device(self):
        device = dev.select()
        torch.manual_seed(0)
        net = ImgSetDiT(40, dim=64, depth=2, heads=4, enc_dim=32, n_ctx=5).to(device).eval()
        x = torch.randn(2, 8, 40, device=device)
        t = torch.rand(2, device=device)
        ctx = net.encode_cond(torch.randn(2, 5, 32, device=device).to(torch.bfloat16))
        with torch.no_grad(), dev.amp(device):
            y = net(x, t, ctx)
        dev.sync(device)
        self.assertEqual(y.shape, (2, 8, 40))
        self.assertEqual(y.dtype, torch.float32, 'output head ran under bf16 autocast')
        self.assertTrue(torch.isfinite(y).all(), 'non-finite output')


if __name__ == '__main__':
    unittest.main()
