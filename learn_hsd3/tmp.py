    def step(self, action):
        """
        Execute one timestep within the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            np.ndarray: Observation after taking the action.
            float: Reward received after taking the action.
            bool: A flag indicating whether the episode has ended.
            dict: Additional information about the step.
        """
        
        # Define function to calculate distance to the goal
        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space['twist_feats']:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        # Get the previous distance to the goal
        d_prev = distance_to_goal()

        # Execute the action in the environment
        next_obs, reward, done, info = super().step(action)

        # Get the new distance to the goal
        d_new = distance_to_goal()

        # Populate the info dictionary with additional information
        info['potential'] = d_prev - d_new  # Potential is the change in distance to the goal
        info['distance'] = d_new  # Distance to the goal after the step
        info['reached_goal'] = info['distance'] < self.precision  # Whether the goal has been reached

        # Calculate the reward based on the reward type
        if self.reward == 'potential':
            reward = info['potential']
        elif self.reward == 'potential2':
            reward = d_prev - self.gamma * d_new
        elif self.reward == 'potential3':
            reward = 1.0 if info['reached_goal'] else 0.0
            reward += d_prev - self.gamma * d_new
        elif self.reward == 'potential4':
            reward = (d_prev - d_new) / self._d_initial
        elif self.reward == 'distance':
            reward = -info['distance']
        elif self.reward == 'sparse':
            reward = 1.0 if info['reached_goal'] else 0.0
        else:
            raise ValueError(f'Unknown reward: {self.reward}')
        reward -= self.ctrl_cost * np.square(action).sum()  # Subtract control cost from the reward

        # Update the episode status
        info['EpisodeContinues'] = True
        if info['reached_goal'] == True and not self.full_episodes:
            done = True
        info['time'] = self._step
        self._step += 1  # Increment the step counter
        if self._step >= self.max_steps:  # If maximum steps reached, end the episode
            done = True
        elif (
            not info['reached_goal'] and self.np_random.random() < self.reset_p
        ):
            info['RandomReset'] = True  # Randomly reset the episode
            done = True

        # Check if the agent fell over, if falling over is not allowed
        if not self.allow_fallover and self.fell_over():
            reward = self.fallover_penalty  # Assign penalty for falling over
            done = True
            self._do_hard_reset = True
            info['reached_goal'] = False
            info['fell_over'] = True

        # If the episode has ended and a hard reset is needed, remove the 'EpisodeContinues' flag
        if done and (
            self._do_hard_reset
            or (self._reset_counter % self.hard_reset_interval == 0)
        ):
            del info['EpisodeContinues']

        # If it's the final step of the task, set the 'LastStepOfTask' flag
        if done:
            info['LastStepOfTask'] = True

        # If the episode is over but it should continue due to implicit soft resets
        if done and 'EpisodeContinues' in info and self.implicit_soft_resets:
            need_hard_reset = self._do_hard_reset or (
                self.hard_reset_interval > 0
                and self._reset_counter % self.hard_reset_interval == 0
            )
            if not need_hard_reset:
                # Perform a soft reset and let the episode continue
                next_obs = self.reset()
                done = False
                del info['EpisodeContinues']
                info['SoftReset'] = True

        # Record the features used in this step
        info['features'] = self._features_s

        # Return observation, reward, done flag, and info dictionary
        return next_obs, reward, done, info

