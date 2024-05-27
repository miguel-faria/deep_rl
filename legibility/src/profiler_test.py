#! /usr/bin/env python

import jax
from pathlib import Path

with jax.profiler.trace(Path(__file__).parent.absolute().parent.absolute() / 'logs', create_perfetto_link=True):

	key = jax.random.key(0)
	x = jax.random.normal(key, (5000, 5000))
	y = x @ x
	y.block_until_ready()

# jax.profiler.stop_trace()