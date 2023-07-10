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

from typing import List
import jax.numpy as jnp


class Particle:
    def __init__(self, q: jnp.ndarray, p: jnp.ndarray, m: float):
        self.__q = q
        self.__p = p
        self.__m = m

    @property
    def q(self) -> jnp.ndarray:
        return self.__q

    @property
    def p(self) -> jnp.ndarray:
        return self.__p

    @property
    def m(self) -> float:
        return self.__m

    @property
    def v(self) -> jnp.ndarray:
        return self.p / self.m

    @property
    def ke(self) -> float:
        return jnp.dot(self.p, self.p) / self.m

    @q.setter
    def q(self, value: jnp.ndarray) -> None:
        self.__q = value

    @p.setter
    def p(self, value) -> None:
        self.__p = value


class Particles:
    def __init__(self, particles: List[Particle]) -> None:
        self.__particles = particles
        self.idx = 0

    @property
    def particles(self) -> List[Particle]:
        return self.__particles
    
    @particles.setter
    def particles(self, particles: List[Particle]) -> None:
        self.__particles = particles

    def __getitem__(self, idx: int) -> Particle:
        return self.particles[idx]

    def add(self, particle: Particle) -> None:
        self.particles.append(particle)

    def __iter__(self):
        return ParticleIterator(self)


class ParticleIterator:
    def __init__(self, particles: Particles) -> None:
        self.__particles = particles
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            rtrn = self.__particles[self.current_index]
            self.current_index += 1
            return rtrn
        except IndexError:
            raise StopIteration
