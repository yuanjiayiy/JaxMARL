"""
Check that the environment can be reset and stepped with random actions.
TODO: replace this with proper unit tests.
"""
import jax
# import pytest 

from jaxmarl.environments.investesg import InvestESG, State

env = InvestESG()

def test_random_rollout():

    

    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    obs, state = env.reset(rng_reset)
    reset_state = state
    
    
    for i in range(10):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.n_agents)
        actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
        _, state, _, _, _ = env.step(key=rng, state=state, actions=actions, reset_state=reset_state)
        

    
    