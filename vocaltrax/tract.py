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

import jax
import jax.numpy as jnp
import chex
from flax import linen as nn
from tongue import Tongue
from constriction import ThroatConstriction, LipConstriction
from glottis import glottis_make_waveform
from utils.misc import unnormalize_params, upsample_frames


class PhysicalTract(nn.Module):
    num_frames: int
    min_diam: float = 0.2
    max_diam: float = 1.5
    num_segments: int = 44
    base_diams: chex.Array = jnp.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.1,
                                        1.1, 1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                        1.5, 1.5, 1.5, 1.5])

    def setup(self):
        tongue = Tongue()
        throatconstriction = ThroatConstriction()
        lipconstriction = LipConstriction()

        self.tongue_init_fn = lambda rng, diams: jax.vmap(
            tongue.init,
            in_axes=0
        )(jax.random.split(
                rng,
                self.num_frames
            ),
          diams
          )
        self.throatconstriction_init_fn = lambda rng, diams: jax.vmap(
            throatconstriction.init,
            in_axes=0
        )(jax.random.split(
            rng,
            self.num_frames
        ),
          diams
          )
        self.lipconstriction_init_fn = lambda rng, diams: jax.vmap(
            lipconstriction.init,
            in_axes=0
        )(jax.random.split(
            rng,
            self.num_frames
        ),
          diams
          )

        self.tongue_params = self.param(
            "tongue",
            self.tongue_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )
        self.throatconstriction_params = self.param(
            "throatconstriction",
            self.throatconstriction_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )
        self.lipconstriction_params = self.param(
            "lipconstriction",
            self.lipconstriction_init_fn,
            jnp.tile(self.base_diams, (self.num_frames, 1))
        )

        self.apply_tongue = jax.vmap(tongue.apply)
        self.apply_throatconstriction = jax.vmap(throatconstriction.apply)
        self.apply_lipconstriction = jax.vmap(lipconstriction.apply)

    def __call__(self):
        base_diams = jnp.tile(
            unnormalize_params(self.base_diams, self.min_diam, self.max_diam),
            (self.num_frames, 1)
        )
        tongue_output = self.apply_tongue(self.tongue_params, base_diams)
        diams = self.apply_throatconstriction(self.throatconstriction_params, tongue_output)
        diams = self.apply_lipconstriction(self.lipconstriction_params, diams)
        return diams


class VocalTract(nn.Module):
    num_frames: int
    f0s: chex.Array
    upsample_factor: int
    frame_len: int
    sample_rate: int

    min_tenseness: float = 0.1
    max_tenseness: float = 1.0
    min_reflection: float = -0.9
    max_reflection: float = 0.9

    glottal_reflection: float = 0.75
    lip_reflection: float = -0.85

    def setup(self):
        self.key = self.make_rng("key")

        physical = PhysicalTract(
            num_frames=self.num_frames
        )
        self.physical_apply = physical.apply
        self.physical_params = self.param(
            "physical",
            lambda rng: physical.init(rng)
            )
        self.tenses = self.param(
            "tenses",
            lambda rng: jnp.ones((self.num_frames, 1))
        )

    def __call__(self):
        diams = self.physical_apply(self.physical_params)
        tenses = unnormalize_params(
            self.tenses,
            self.min_tenseness,
            self.max_tenseness
        )

        f0s = self.f0s

        tenses = upsample_frames(tenses, self.upsample_factor)
        f0s = upsample_frames(f0s, self.upsample_factor)

        waveform = glottis_make_waveform(
            tenses,
            f0s,
            jnp.zeros(self.frame_len//self.upsample_factor),
            self.sample_rate,
            self.key
        )

        return process_diams(
            waveform,
            diams,
            self.glottal_reflection,
            self.lip_reflection,
            frame_size=self.frame_len
        )


def process_diams(
    input: chex.Array,
    diameters: chex.Array,
    glottal_reflection: float = 0.75,
    lip_reflection: float = -0.85,
    frame_size: float = 1024
) -> chex.Array:
    A = diameters**2
    reflections = (A[:, :-1] - A[:, 1:]) / (A[:, :-1] + A[:, 1:] + 1e-12)

    n_frames, size = jnp.shape(diameters)
    n = (n_frames - 1) * frame_size
    reflections = upsample_frames(reflections, frame_size).reshape((n, size - 1))
    input = jnp.pad(input, (0, n - input.size), mode='constant', constant_values=0)

    R = jnp.zeros(size)
    L = jnp.zeros(size)

    # Vectorized update functions
    def update_junctions(R, L, reflection, input_sample):
        junction_outR = jnp.zeros(size + 1)
        junction_outL = jnp.zeros(size + 1)

        junction_outR = junction_outR.at[0].set(
            L[0] * glottal_reflection + input_sample
        )
        junction_outL = junction_outL.at[size].set(R[size - 1] * lip_reflection)

        w = reflection * (R[:-1] + L[1:])
        junction_outR = junction_outR.at[1:-1].set(R[:-1] - w)
        junction_outL = junction_outL.at[1:-1].set(L[1:] + w)

        return junction_outR, junction_outL

    def update_RL(junction_outR, junction_outL):
        R = junction_outR[:size] * 0.999
        L = junction_outL[1 : size + 1] * 0.999
        return R, L

    # Main simulation loop
    def simulation_step(carry, x):
        R, L = carry
        reflection, input_sample = x
        junction_outR, junction_outL = update_junctions(R, L, reflection, input_sample)
        R, L = update_RL(junction_outR, junction_outL)
        output1 = R[size - 1]
        junction_outR, junction_outL = update_junctions(R, L, reflection, input_sample)
        R, L = update_RL(junction_outR, junction_outL)
        output2 = R[size - 1]
        return (R, L), jnp.array([output1, output2])

    # Run the simulation using jax.lax.scan
    _, out = jax.lax.scan(simulation_step, (R, L), (reflections, input))

    out = jnp.append(0, jnp.ravel(out))
    out = out[1:] + out[:-1]

    return out[1::2] * 0.25
