#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""

import random

import gym
import numpy as np

import gym_remote.client as grc
import gym_remote.exceptions as gre

#from retro_contest.local import make
#import time

EXPLOIT_BIAS = 0.40
TOTAL_TIMESTEPS = int(1e6)

ROLL_PROB = 1
MOMENTUM_REQUIRED = 7

# v2: move forward steps increased to 200
# v3: move forward steps changed to 150
# v4: exploit bias changed to 0.40 from 0.25 and v3
# v5: v4 and jump_repeat to 8 from 4
# v6: v5 and jump prob to 2/10 from 1/10
# v7: minus v6 changes and added always spin attack if not jumping # spin attack not working
# v8: fixed v7 to actually always spin attack if not jumping # spin attack not working
# v9: added go right for exploit-waste and v5
# v10: v9 and move right to 175 from 150
# v11: v9 and jump rep. to 10 from 8
# v12: v9 and roll if has momentum 7
# v13: v12 and exploit-waste as move()
# v14: v9 and exploit-waste as move()
# v15: v14 and momentum left,right (4, 7, 10) with roll prob. (0.3, 0.5, 0.7, 1)
# v16: v14 and waste-exploit only go RIGHT
# v17: v14 and momentum roll with jumping next if scheduled
# v18: v14 and left dir. reward set to 50


def main():
    """Run JERK on the attached environment."""
    env = grc.RemoteEnv('tmp/sock')
#    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act1')
#    env.render()
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
#    obs = env.reset()
    while True:
        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, 150) # increased to 200 from 100 for v2
        print("REWARD FOR TOTAL RIGHT MOVE IS: ", rew)
        if not new_ep and rew <= 50:
            print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))
#        env.render()
#    env.close() # close environment 

def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=8):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
#    has_momentum = 0
    last_action = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
#            is_jumping = True
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
#                is_jumping = True
#        if has_momentum >= 7: # in momentum (has some good speed)
#            action = np.zeros((12,), dtype=np.bool)
#            action[5] = True # roll
            
        _, rew, done, _ = env.step(action)
#        if ((sum(action) == 1) and (action[7] or action[6])) == True:
#            has_momentum++
#        if (sum(action) == 1):
#            if action[last_action]:
#                has_momentum += 1
#            else:
#                last_action = np.where(action)[0][0] # get index of action that is true
#                has_momentum = 0
#        else:
#            in_speed = False
        total_rew += rew
#        if steps_taken % 10 == 0:
#            env.render()
#        env.render()
        steps_taken += 1
        if done:
            break
    return total_rew, done

def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            rew, done = move(env, 150) #
            if done:
                break
            if rew <= 0:
                print('backtracking due to negative reward: %f' % rew)
                _, done = move(env, 70, left=True)
#            action = np.zeros((12,), dtype=np.bool)
#            action[7] = True # go right
#            print('WASTED A MOVE@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#            _, rew, done, _ = env.step(action)
        else:
            _, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
#        self.env.render()
#        time.sleep(.1)
#        print(info)
#        print(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
