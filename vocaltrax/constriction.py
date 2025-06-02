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

class ThroatConstriction(nn.Module):
    constr_idx: int = 12

    constr_val_min: float = -0.99
    constr_val_max: float = 0.99

    def setup(self):
        self.constr_val = self.param("constr_val", lambda rng: 0.5)

    def __call__(self, curr_diam: chex.Array) -> chex.Array:
        """
        Returns chex.Array of diams after applying constriction
        """
        constr_val = unnormalize_params(self.constr_val, self.constr_val_min, self.constr_val_max)

        indices = jnp.arange(curr_diam.size)
        condition = indices <= self.constr_idx
        new_diam = jnp.where(
            condition,
            curr_diam*(1-constr_val),
            curr_diam
        )
        return new_diam

class LipConstriction(nn.Module):
    constr_idx: int = 39

    constr_val_min: float = -0.99
    constr_val_max: float = 0.99

    def setup(self):
        self.constr_val = self.param("constr_val", lambda rng: 0.5)

    def __call__(self, curr_diam: chex.Array) -> chex.Array:
        """
        Returns chex.Array of diams after applying constriction
        """
        constr_val = unnormalize_params(self.constr_val, self.constr_val_min, self.constr_val_max)

        indices = jnp.arange(curr_diam.size)
        condition = indices >= self.constr_idx
        new_diam = jnp.where(
            condition,
            curr_diam*(1-constr_val),
            curr_diam
        )
        return new_diam
