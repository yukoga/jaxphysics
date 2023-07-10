# MIT License

# Copyright (c) 2023 yukoga@

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
# ==============================================================================

import sys
sys.path.append('./src/')

import pytest
import jax.numpy as jnp
from jaxphysics.systems.particle import Particle, Particles
from jaxphysics.systems.system import System


@pytest.fixture
def system():
    particles = Particles([
        Particle(jnp.array([0., 1.]), jnp.array([1., 1.]), 1.0),
        Particle(jnp.array([0., 2.]), jnp.array([2., 2.]), 2.0)
    ])
    return System(particles)


def test_init_system(system):
    q_list = [jnp.array([0., 1.]), jnp.array([0., 2.])]
    p_list = [jnp.array([1., 1.]), jnp.array([2., 2.])]
    m_list = [1.0, 2.0]

    for i, particle in enumerate(system.particles):
        assert particle.m == m_list[i]
        assert all(particle.q == q_list[i])
        assert all(particle.p == p_list[i])
        assert all(particle.v == p_list[i] / m_list[i])


def test_kinetic_energy(system):
    assert system.ke == 6.0


def test_generalized_coordinates(system):
    q_list = [jnp.array([0., 1.]), jnp.array([0., 2.])]
    p_list = [jnp.array([1., 1.]), jnp.array([2., 2.])]

    q_true = jnp.stack(q_list)
    p_true = jnp.stack(p_list)

    assert jnp.alltrue(system.q == q_true)
    assert jnp.alltrue(system.p == p_true)
