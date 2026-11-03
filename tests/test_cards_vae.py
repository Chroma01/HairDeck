"""CardVAE decode: the fp32 output head stays fp32 under an outer bf16 autocast."""
import unittest

import torch

from hairdeck import dev
from hairdeck.cards.vae import CardVAE

HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


class DecodeHeadTests(unittest.TestCase):
    @unittest.skipUnless(HAS_CUDA or HAS_MPS, 'no CUDA or Apple MPS device')
    def test_decode_output_head_is_fp32_under_outer_autocast(self):
        device = torch.device('cuda' if HAS_CUDA else 'mps')
        torch.manual_seed(0)
        vae = CardVAE(L=8, dim=32, depth=1, heads=4, zdim=8, nf=4, dec='mlp').to(device).eval()
        z = torch.randn(2, 8, device=device)
        u = torch.rand(2, 8, device=device)
        mask = torch.ones(2, 8, dtype=torch.bool, device=device)
        with torch.no_grad(), dev.amp(device):
            out = vae.decode(z, u, mask)
        self.assertEqual(out.dtype, torch.float32, 'output head ran under bf16 autocast')


if __name__ == '__main__':
    unittest.main()
