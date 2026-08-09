import cProfile
import pstats

from footballcoach.ai.curriculum.envs import build_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID

phase = PHASES_BY_ID[1]
env = build_env(phase)


def run():
    for _ in range(30):
        env.reset()
        done = False
        while not done:
            _obs, _r, done, _info = env.step()


pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("cumulative")
st.print_stats(30)
st.sort_stats("tottime")
st.print_stats(20)
