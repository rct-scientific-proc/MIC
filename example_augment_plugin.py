"""Example --augment-plugin file: ship your own training augmentations
without touching the pipeline code.

    python train.py data.h5 --augment-plugin example_augment_plugin.py \
        --augment hflip:p=0.5 gaussnoise:p=0.8,sigma=12 gridmask

The contract (identical to the built-in catalog in dataset.py):

  AUGMENTATIONS - required: {name: factory}. Each factory's KEYWORD
      arguments (with defaults) become that augmentation's CLI-settable
      parameters: `--augment name:key=value,key=value` on the command
      line, `{"name": ..., "key": value}` objects in a --config file.
      Unknown names and unknown parameters are rejected up front, so a
      typo never trains silently. Ranges parse as lo-hi (sigma=0.1-1.5
      arrives as the tuple (0.1, 1.5)).

  The factory returns the transform: any callable mapping an image
      tensor to an image tensor - a torchvision v2 transform, a
      torch.nn.Module of your own, or v2.Lambda. Input depends on the
      stage:
        * pre-resize (the default): the raw CHW uint8 crop at source
          resolution - 3 channels, or 1 for a grayscale dataset; return
          the same dtype and layout.
        * post-resize (names listed in POST_RESIZE): the model-scale
          float tensor - 224x224, scaled to [0,1] and ImageNet-normalized
          when --imagenet-norm is on. Use this stage when the effect
          should be consistent at model scale (see the built-in erasing).

  POST_RESIZE - optional: a set of names from your AUGMENTATIONS that
      run in the post-resize stage.

  Probability gating: wrap the transform in v2.RandomApply (the _maybe
      helper below) and expose p as a parameter - the whole catalog
      follows the "p=1 applies always" convention.

Augmentations only ever touch the TRAINING split; validation, evaluation,
and inference stay untouched. Runs record --augment-plugin in their
config.json, so ship the plugin file alongside the config to keep a run
reproducible. Names must not collide with the built-ins or other plugins.
"""

import torch
from torchvision.transforms import v2


def _maybe(p, transform):
    return transform if p >= 1 else v2.RandomApply([transform], p=p)


class GaussianNoise(torch.nn.Module):
    """Additive gaussian pixel noise on the raw uint8 crop (pre-resize):
    computed in float, clamped, and returned as uint8 again."""

    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, img):
        noise = torch.randn_like(img, dtype=torch.float32) * self.sigma
        return (img.float() + noise).clamp(0, 255).to(img.dtype)


AUGMENTATIONS = {
    # simplest case: a stock torchvision transform with a probability knob
    "hflip": lambda p=0.5: v2.RandomHorizontalFlip(p=p),
    # a custom torch.nn.Module wrapped in the probability gate; `sigma`
    # becomes CLI-settable, e.g. gaussnoise:p=0.8,sigma=12
    "gaussnoise": lambda p=0.5, sigma=10.0: _maybe(p, GaussianNoise(sigma)),
    # a post-resize entry (listed in POST_RESIZE below): gray patches at
    # model scale, e.g. gridmask:p=0.5,scale=0.05-0.2
    "gridmask": lambda p=0.3, scale=(0.02, 0.1), value=0.5:
        v2.RandomErasing(p=p, scale=scale, value=value),
}

POST_RESIZE = {"gridmask"}
