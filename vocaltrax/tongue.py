# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>
# Luke Mo <lukemo@mit.edu>
# Quinn Langford <langford@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jax.numpy as jnp
import chex
from flax import linen as nn
from utils.misc import unnormalize_params

class Tongue(nn.Module):
    tongue_diam_min: float = 2.05
    tongue_diam_max: float = 3.5
    tongue_idx_min: int = 12
    tongue_idx_max: int = 29

    blade_start: int = 10
    lip_start: int = 39
    tip_start: int = 32
    grid_offset: float = 1.7

    def setup(self):
        self.tongue_diam = self.param("tongue_diam", lambda rng: 0.5)
        self.tongue_idx = self.param("tongue_idx", lambda rng: 0.5)

    def __call__(self, base_diam: chex.Array) -> chex.Array:
        """
        Return chex.Array of diameters after adding tongue
        """
        tongue_idxs = jnp.arange(self.blade_start, self.lip_start) * 1.0
        curve_mod = jnp.ones_like(tongue_idxs)
        curve_mod = curve_mod.at[0].set(0.94)
        curve_mod = curve_mod.at[-1].set(0.8)
        curve_mod = curve_mod.at[-2].set(0.94)
        tongue_idx = unnormalize_params(self.tongue_idx, self.tongue_idx_min, self.tongue_idx_max)
        tongue_diam = unnormalize_params(self.tongue_diam, self.tongue_diam_min, self.tongue_diam_max)

        t = 1.1 * jnp.pi * (tongue_idx - tongue_idxs) / (self.tip_start - self.blade_start)
        fixed_tongue_diameter = 2 + (tongue_diam - 2) / 1.5
        curve = (1.5 - fixed_tongue_diameter + self.grid_offset) * jnp.cos(t) * curve_mod
        new_diam = 1.5 - curve
        return jnp.concatenate(
            [
                base_diam[:self.blade_start],
                new_diam,
                base_diam[self.lip_start:]
            ]
        )
