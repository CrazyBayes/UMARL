import numpy as np
class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError
    def get_visibility_matrix(self):
        agents_location = self.env.unwrapped.observation()[0]['left_team'][-self.n_agents:]
        #print(agents_location)
        matricx = np.zeros((self.n_agents, self.n_agents))
        for agent_a in range(0,self.n_agents):
            matricx[agent_a][agent_a] = 1.0
            for agent_b  in range(0,self.n_agents):
                if agent_a > agent_b:
                    distance = np.sqrt((agents_location[agent_a][0] - agents_location[agent_b][0]) ** 2
                                       + (agents_location[agent_a][1] - agents_location[agent_b][1]) ** 2)
                    if distance <= 0.75:
                        matricx[agent_a][agent_b] = 1
                        matricx[agent_b][agent_a] = 1
        #print(matricx)
        #1/0
        return  matricx

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
