import random
# from argparse import ArgumentParser
import gym
import os
import pickle

INF = int(1e15)


class GamePlayer:

    def __init__(self):

        self.env = gym.make('Pong-v0')

        self.prev_ball_pos = (0,0)
        self.scores_log = open("scores.txt", "w")
        self.testing = False
        self.min_epsilon = 0.2


    def prepro(self, state):
        state = state[35:195]  # crop
        state = state[::2, ::2, 0]  # downsample by factor of 2

        # [(Ball),P1,P2,(BallDirc)]
        vector = [(0,0), 0, 0,(0,0)]

        p1_flag = False
        p2_flag = False
        ball_flag = False

        # Players posotions: # before downsample
        p1_col = 8  # 16
        p2_col = 70  # 140

        # game board border lines: # before downsample
        game_borad_up = 0  # 34
        game_borad_down = 80  # 194
        game_borad_left = 0
        game_borad_right = 80  # 160

        p1_Rcolor = 213
        p2_Rcolor = 92
        ball_Rcolor = 236

        for i in range(game_borad_up, game_borad_down):

            if not p1_flag:
                # player1(left) position:
                if state[i][p1_col] == p1_Rcolor:
                    vector[1] = i
                    p1_flag = True

            if not p2_flag:
                # player2(right) position:
                if state[i][p2_col] == p2_Rcolor:
                    vector[2] = i
                    p2_flag = True

            if not ball_flag:
                # # Ball position:
                for j in range(game_borad_left, game_borad_right):
                    if state[i][j] == ball_Rcolor:
                        vector[0] = (i,j)
                        ball_flag = True
            if p1_flag and p2_flag and ball_flag:
                break

        if vector[0] == (0,0) or self.prev_ball_pos == (0,0):
            vector[3] = (0,0)
        else:
            newX, newY = vector[0]
            oldX, oldY = self.prev_ball_pos
            vector[3] = (newX-oldX,newY-oldY)

        self.prev_ball_pos = vector[0]

        return vector[0], vector[1], vector[2], vector[3]

    def reset_game(self):
        return self.prepro(self.env.reset())

    def epsilon_greedy(self,Q, seen_combinations, state, legal_actions, epsilon):
        not_tried_yet = []
        for action in legal_actions:
            if (state, action) not in seen_combinations:
                not_tried_yet.append(action)

        if not_tried_yet != []:
            return random.choice(not_tried_yet)

        if random.random() < epsilon and not self.testing:
            return random.choice(legal_actions)
        else:
            return self.best_action(Q, state, legal_actions)


    def best_action(self,Q, state, legal_actions):
        best_action = 0
        best_action_utility = -INF

        for action in legal_actions:
            if (state, action) not in Q:
                Q[state, action] = 0
            if Q[state, action] > best_action_utility:
                best_action_utility = Q[state, action]
                best_action = action

        return best_action

    def get_action(self, Q, seen_combinations, state, actions, epsilon):
        action = self.epsilon_greedy(Q, seen_combinations, state, actions, epsilon)
        return action

    # Print the action(DOWN,UP,STAY):
    # Argument: action - 0/1/2/3/4/5
    def action_to_string(self,action):
        if action == 1 or action == 0:
            print("STAY")
        elif action == 2 or action == 4:
            print("UP")
        elif action == 3 or action == 5:
            print("DOWN")

    def q_learning(self):
        Q = {}
        actions_array = [0,2,3]

        combinations = {}

        epsilon = 0.99
        alpha = 0.1
        gamma = 0.95
        episodes = 10000

        restore = False
        episodeStart = 1
        if restore:
            with open('objs.pkl', 'rb') as f:
                Q,episodeStart,epsilon = pickle.load(f)

        scores = []

        for episode in range(episodeStart, episodes + 1):

            state = self.reset_game()
            epsilon *= 0.995

            if epsilon < self.min_epsilon:
                epsilon = self.min_epsilon

            curr_score = 0
            done = False

            while not done:
                # if episode-1 % 50 == 0:
                #     self.env.render()
                self.env.render()

                action = self.get_action(Q, combinations, state, actions_array, epsilon)

                combinations[state, action] = True
                newState, reward, done, _ = self.env.step(action)
                curr_score += reward
                next_state = self.prepro(newState)
                next_action = self.best_action(Q, next_state, actions_array)

                if (state, action) not in Q:
                    Q[state, action] = 0

                Q[state,action] = Q[state,action] + alpha * (reward + gamma * Q[next_state,next_action] - Q[state,action])
                state = next_state

            scores.append(curr_score)

            to_file = "%d,%d" %(episode,curr_score)
            self.scores_log.write(to_file + "\n")

            self.scores_log.flush()
            os.fsync(self.scores_log.fileno())

            if episode % 200 == 0:
                with open('data_' + str(episode) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([Q,episode,epsilon], f)

            print("Episode:", episode, ", Score:",curr_score, "Epsilon:",epsilon)



if __name__ == '__main__':
    gp = GamePlayer()
    gp.q_learning()