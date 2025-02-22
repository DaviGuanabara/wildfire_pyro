
# TODO: Code this class. 
# This class should be used to evaluate the supervised learning policy
class evaluate_policy:
    def __init__(self, env, policy, num_episodes=100, max_steps=1000):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_steps = max_steps