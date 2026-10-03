"""Predeclared joint crop/resize geometry; independent of thread scheduling."""
import math
import numpy as np

RECIPE = dict(full_image_probability=0.2, crop_area_fraction=[0.15, 1.0],
              source_aspect_ratio=[1.3, 2.5], attempts=10,
              fallback='largest fitting crop at the last sampled ratio',
              images='AREA', flow='LINEAR_then_separate_u_v_scale',
              seed_namespace=20261004)


def sample_box(h, w, seed):
    """Return y,x,h,w. Both images and the flow use this same rectangle."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if rng.random() < RECIPE['full_image_probability']:
        return 0, 0, h, w
    for _ in range(RECIPE['attempts']):
        area = h*w*rng.uniform(*RECIPE['crop_area_fraction'])
        ratio = math.exp(rng.uniform(*np.log(RECIPE['source_aspect_ratio'])))
        cw, ch = round(math.sqrt(area*ratio)), round(math.sqrt(area/ratio))
        if 1 <= ch <= h and 1 <= cw <= w:
            break
    else:
        cw, ch = (w, min(h, round(w/ratio))) if w/h <= ratio else (min(w, round(h*ratio)), h)
    return int(rng.integers(h-ch+1)), int(rng.integers(w-cw+1)), ch, cw


def step_lr(step, steps, initial=1e-5, minimum=1e-6):
    if not 1 <= step <= steps or steps < 2 or not 0 < minimum <= initial:
        raise ValueError('Invalid step or learning-rate bounds')
    return minimum+(initial-minimum)*(1+math.cos(math.pi*(step-1)/(steps-1)))/2
