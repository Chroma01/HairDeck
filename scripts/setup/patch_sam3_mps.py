"""Make the downloaded SAM 3 tree device-agnostic: apply after scripts/setup/download_weights.py fetched sam3_src.

SAM 3 hardcodes "cuda" in three construction-time places, and _setup_device_and_mode only honours "cuda", so on Apple
MPS the model never reaches the device. The edits below are idempotent and limited to those spots; run this script again
after re-downloading sam3_src. Usage: python scripts/setup/patch_sam3_mps.py [--check]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hairdeck import paths  # noqa: E402

EDITS = [
    # 1. any device, not just "cuda"
    ('sam3/model_builder.py',
     '''    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    if eval_mode:''',
     '''    """Setup model device and evaluation mode."""
    if device is not None and str(device) != "cpu":
        model = model.to(device)
    if eval_mode:'''),
    # 2. the precomputed position-encoding cache lands on "cuda"; without CUDA, forward() fills it lazily on x.device
    ('sam3/model_builder.py',
     '''    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(''',
     '''    """Create position encoding for visual backbone."""
    if not torch.cuda.is_available():
        precompute_resolution = None  # a "cuda" cache; forward() fills it lazily on the features' device instead
    return PositionEmbeddingSine('''),
    # 3. the compilable box-RoPE coordinate cache likewise; forward() fills it on reference_boxes.device instead
    ('sam3/model/decoder.py',
     '''            if resolution is not None and stride is not None:
                feat_size = resolution // stride''',
     '''            if resolution is not None and stride is not None and torch.cuda.is_available():
                feat_size = resolution // stride'''),
    # 4. aten::_addmm_activation is CUDA-only (SAM 3's fused MLP)
    ('sam3/perflib/fused.py',
     '''addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):''',
     '''addmm_act_op = torch.ops.aten._addmm_activation


def _mm_act(self, mat1_flat, mat2, gelu):
    """the fused aten op on CUDA, addmm plus the activation elsewhere (not implemented for MPS)"""
    if torch.cuda.is_available():
        return addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=gelu)
    y = torch.addmm(self, mat1_flat, mat2.t())
    return torch.nn.functional.gelu(y) if gelu else torch.nn.functional.relu(y)


def addmm_act(activation, linear, mat1):'''),
    ('sam3/perflib/fused.py',
     '        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)',
     '        y = _mm_act(self, mat1_flat, mat2, False)'),
    ('sam3/perflib/fused.py',
     '        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)',
     '        y = _mm_act(self, mat1_flat, mat2, True)'),
    # 5. an empty point prompt reaches grid_sample, which MPS rejects ("Placeholder tensor is empty")
    ('sam3/model/geometry_encoders.py',
     '''    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]
''',
     '''    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]
        if n_points == 0:  # an empty prompt (e.g. _get_dummy_prompt): MPS cannot grid_sample an empty grid
            return (
                torch.zeros(0, bs, self.d_model, device=points.device, dtype=img_feats[-1].dtype),
                points_mask,
            )
'''),
    # 6. pin_memory() then .to(mps) is rejected; a 4-element tensor can be made on the device directly
    ('sam3/model/geometry_encoders.py',
     '''            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)''',
     '''            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)'''),
    # 7. aten::_assert_async is CUDA-only; the same two checks as a synchronising assert elsewhere
    ('sam3/model/geometry_encoders.py',
     '''    torch._assert_async(is_right_padded(mask1))
    torch._assert_async(is_right_padded(mask2))''',
     '''    if torch.cuda.is_available():
        torch._assert_async(is_right_padded(mask1))
        torch._assert_async(is_right_padded(mask2))
    else:
        assert bool(is_right_padded(mask1)), "mask1 is not right padded"
        assert bool(is_right_padded(mask2)), "mask2 is not right padded"'''),
    ('sam3/model/sam3_image.py',
     '                torch._assert_async((img_ids >= 0).all())',
     '                assert bool((img_ids >= 0).all()), "negative img_id"'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='report whether the edits are in place, change nothing')
    a = ap.parse_args()
    repo = paths.sam3_repo()
    missing = []
    for rel, old, new in EDITS:
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            missing.append(f'{rel} (absent; run download_weights.py --only sam3_src)')
            continue
        with open(p, encoding='utf-8') as f:
            s = f.read()
        if new in s:
            print(f'present {rel}')
        elif old in s:
            if a.check:
                print(f'MISSING {rel}')
                missing.append(rel)
            else:
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(s.replace(old, new, 1))
                print(f'patched {rel}')
        else:
            print(f'UNEXPECTED {rel}: neither the original nor the patched text; re-download sam3_src')
            missing.append(rel)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
