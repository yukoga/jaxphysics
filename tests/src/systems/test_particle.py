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


@pytest.fixture
def particles():
    return Particles([
        Particle(jnp.array([0., 1.]), jnp.array([1., 1.]), 1.0),
        Particle(jnp.array([0., 2.]), jnp.array([2., 2.]), 2.0)
    ])


def test_init_particle(particles):
    assert all(particles[0].q == jnp.array([0., 1.]))
    assert all(particles[0].p == jnp.array([1., 1.]))


def test_update_particle(particles):
    particles[0].q = jnp.array([2., 2.])
    particles[0].p = jnp.array([1., 0.])
    assert all(particles[0].q == jnp.array([2., 2.]))
    assert all(particles[0].p == jnp.array([1., 0.]))


def test_particle_velocity(particles):
    assert all(particles[0].v == jnp.array([1., 1.])/1.0)


def test_kinetic_energy(particles):
    assert particles[0].ke == 2.0


def test_init_multiple_particles(particles):
    assert all(particles[0].q == jnp.array([0., 1.]))
    assert all(particles[0].p == jnp.array([1., 1.]))
    assert all(particles[1].q == jnp.array([0., 2.]))
    assert all(particles[1].p == jnp.array([2., 2.]))


def test_add_particle_to_particles(particles):
    particles.add(Particle(jnp.array([0., 3.]), jnp.array([3., 3.]), 3.0))

    assert all(particles[2].q == jnp.array([0., 3.]))
    assert all(particles[2].p == jnp.array([3., 3.]))


def test_iterate_particles(particles):
    q_list = [jnp.array([0., 1.]), jnp.array([0., 2.])]
    p_list = [jnp.array([1., 1.]), jnp.array([2., 2.])]
    m_list = [1.0, 2.0]

    for i, particle in enumerate(particles):
        assert particle.m == m_list[i]
        assert all(particle.q == q_list[i])
        assert all(particle.p == p_list[i])
        assert all(particle.v == p_list[i] / m_list[i])
