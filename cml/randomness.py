"""
 ~ CML // Creative Machine Learning ~
 random.py : Handling JAX randomness generator elegantly
 
 As JAX is very explicit in randomness, we can handle it in an elegant 
 by relying on a Singleton pattern serving 
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import jax
from jax import random

class Random(object):
    _instance = None
    key: random.PRNGKey = None

    def __new__(cls, seed: int = 0):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(Random, cls).__new__(cls)
            cls.key = random.PRNGKey(seed)
            # Put any initialization here.
        return cls._instance
    
    def split(self, nb_split: int = 2):
        key, subkeys = random.split(self.key, nb_split)
        self.key = key
        return subkeys
    
    def bernoulli(self, **kwargs):
        key = self.split()
        return random.bernoulli(key, **kwargs)
    
    def beta(self, **kwargs):
        key = self.split()
        return random.beta(key, **kwargs)
    
    def gumbel(self, **kwargs):
        key = self.split()
        return random.gumbel(key, **kwargs)
    
    def multivariate_normal(self, **kwargs):
        key = self.split()
        return random.multivariate_normal(key, **kwargs)
    
    def normal(self, **kwargs):
        key = self.split()
        return random.normal(key, **kwargs)
    
    def randint(self, **kwargs):
        key = self.split()
        return random.randint(key, **kwargs)
    
    def uniform(self, **kwargs):
        key = self.split()
        return random.uniform(key, **kwargs)
        