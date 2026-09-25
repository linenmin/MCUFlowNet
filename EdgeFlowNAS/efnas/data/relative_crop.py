"""Integer relative crops with exact pixel-unit flow and per-batch audit data."""
import hashlib
import numpy as np


class AuditedTuple(tuple):
    def __new__(cls, values, audit):
        obj = super().__new__(cls, values)
        obj.audit = audit
        return obj


def relative_crop(img0, img1, flow, crop_h, crop_w, rng, config):
    h, w = img0.shape[:2]
    if img1.shape != img0.shape or flow.shape != (h, w, 2):
        raise ValueError('Relative crop requires aligned RGB images and 2-channel pixel flow')
    if h < crop_h or w < crop_w:
        raise ValueError('Crop exceeds the source image')
    probability, radius = float(config['probability']), int(config['max_offset'])
    if not 0 <= probability <= 1 or not 0 <= radius <= 32:
        raise ValueError('Invalid relative crop settings')
    # Exactly the two draws used by _random_crop_triplet; augmentation uses a copy.
    top, left = int(rng.randint(0, h-crop_h+1)), int(rng.randint(0, w-crop_w+1))
    augmentation_rng = np.random.RandomState()
    augmentation_rng.set_state(rng.get_state())
    requested = bool(augmentation_rng.rand() < probability)
    dx = dy = 0
    if requested:
        dx, dy = map(int, augmentation_rng.randint(-radius, radius+1, size=2))
    fallback = not (0 <= top+dy <= h-crop_h and 0 <= left+dx <= w-crop_w)
    if fallback:
        dx = dy = 0
    first = img0[top:top+crop_h, left:left+crop_w]
    second = img1[top+dy:top+dy+crop_h, left+dx:left+dx+crop_w]
    target = flow[top:top+crop_h, left:left+crop_w].copy()
    target -= np.asarray([dx, dy], np.float32)
    return AuditedTuple((first, second, target), dict(dx=dx, dy=dy,
        requested=int(requested), fallback=int(fallback), applied=int(bool(dx or dy))))


def first_batch_motion_audit(labels, sample_audits):
    """Only run on the first consumed batch per reporting block (not all pixels)."""
    offsets = np.asarray([[a['dx'], a['dy']] for a in sample_audits], np.float32)
    before = labels + offsets[:, None, None, :]
    result = {'before_label_sha256': hashlib.sha256(before.tobytes()).hexdigest()}
    h, w = labels.shape[1:3]
    yy, xx = np.mgrid[:h, :w]
    for name, flow in [('before', before), ('after', labels)]:
        magnitude = np.linalg.norm(flow, axis=-1)
        for low, high, label in [(0,10,'0_10'),(10,40,'10_40'),(40,160,'40_160'),(160,np.inf,'160_plus')]:
            result[f'{name}_{label}_fraction'] = float(np.mean((magnitude >= low) & (magnitude < high)))
        result[f'{name}_mask400_fraction'] = float(np.mean(magnitude >= 400))
        x, y = xx+flow[...,0], yy+flow[...,1]
        result[f'{name}_outside_fraction'] = float(np.mean((x<0)|(x>=w)|(y<0)|(y>=h)))
    return result
