import tensorflow as tf
import numpy as np
import random
import gym
import math
import os
import threading
import time
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#######################
# Model hyperparameters
#######################
state_size = 4  # Our vector size.
original_state_size = (210, 160, 3)
action_size = 6  # Actions: [stay,stay,up,down,up,down]
stack_size = 4  # stack with 4 states.
stack_states_size = [stack_size, state_size]  # The size of the input to neural network.
batch_size = 1000  # Mini batch size.

learning_rate = 0.00001  # Alpha(learning rate).
gamma = 0.99  # Discounting rate.

total_episodes = 50000  # Total episodes for training.
saveEvery = 1000  # Save the model every few games.

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start

explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.000000001  # 0.00000001 exponential decay rate for exploration prob

memory_size = 300000  # Number of experiences the Memory can keep

rewards_list = []  # list of all training rewards.

# MODIFY THIS TO FALSE IF IS NOT THE FIRST TRAINING EPISODE.
firstTraining = True
# firstTraining = False


class DQNetwork:
    def __init__(self, device):
        with tf.device(device):
            self.graph = tf.Graph()

            with self.graph.as_default():
                # Create the placeholders
                self.state = tf.placeholder(tf.float32, [None, *stack_states_size], name="inputs")
                self.target_ = tf.placeholder(tf.float32, [None, action_size], name="target_")

                # [?,4,4] -> [?,16]
                self.flatten = tf.contrib.layers.flatten(self.state)
                self.flatten = tf.contrib.layers.layer_norm(self.flatten)

                # hidden layer:
                self.t1 = tf.layers.dense(self.flatten, 128, activation="relu")
                self.t2 = tf.layers.dense(self.t1, 64, activation="relu")

                # output layer:
                self.t3 = tf.layers.dense(self.t2, 6, activation=None)
                self.Q4actions = self.t3

                self.loss = tf.reduce_mean(tf.pow(self.Q4actions - self.target_, 2))
                self.update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()


class GamePlayer:

    def get_epsilon(self, curr_game):

        if curr_game > self.epsilon_decay_time:
            return self.epsilon_end

        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * float(curr_game) / self.epsilon_decay_time

    def prepro(self, state):
        # [P1,P2,xBall,yBall]
        vector = [0, 0, 0, 0]

        p1_flag = False
        p2_flag = False
        ball_flag = False

        # Players posotions:
        self.p1_col = 16
        self.p2_col = 140

        # game board border lines:
        self.game_borad_up = 34
        self.game_borad_down = 194
        self.game_borad_left = 0
        self.game_borad_right = 160

        self.player_size = 16

        self.p1_Rcolor = 213
        self.p2_Rcolor = 213
        self.board_Rcolor = 144
        self.ball_Rcolor = 236

        for i in range(self.game_borad_up, self.game_borad_down):

            if not p1_flag:
                # player1(left) position:
                if state[i][self.p1_col][0] == self.p1_Rcolor:
                    if i >= self.game_borad_up and i < self.game_borad_up+self.player_size:
                        for i2 in range(self.game_borad_up, self.game_borad_up+self.player_size+1):
                            if state[i2][self.p1_col][0] == self.p1_Rcolor and state[i2 + 1][self.p1_col][0] == self.board_Rcolor:
                                vector[0] = i2 - self.player_size + 1
                    else:
                        vector[0] = i
                    p1_flag = True

            if not p2_flag:
                # player2(right) position:
                if state[i][self.p2_col][0] == self.p2_Rcolor:
                    if i >= self.game_borad_up and i < self.game_borad_up+self.player_size:
                        for i2 in range(self.game_borad_up, self.game_borad_up+self.player_size+1):
                            if state[i2][self.p2_col][0] == self.p2_Rcolor and state[i2 + 1][self.p2_col][0] == self.board_Rcolor:
                                vector[1] = i2 - self.player_size + 1
                    else:
                        vector[1] = i
                    p2_flag = True


            if not ball_flag:
                # # Ball position:
                for j in range(self.game_borad_left, self.game_borad_right):
                    if state[i][j][0] == self.ball_Rcolor:
                        vector[2] = i
                        vector[3] = j

                        ball_flag = True

            if p1_flag and p2_flag and ball_flag:
                break

        return vector

    def __init__(self):
        self.game_to_play = 'Pong-v0'  # 'Breakout-v0' #'MsPacman-v0'
        env = gym.make(self.game_to_play)
        self.num_of_actions = env.action_space.n
        self.show_every = 100
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay_time = 10000  # 30000
        self.gamma = 0.99
        self.random_games = 2000  # 5000
        self.save_every_updates = 1000 # also updates_to_update_target 1000
        self.checkpoint_dir_name = 'checkpoints'

        self.graph_target = DQNetwork('/gpu:2')
        self.graph_main = DQNetwork('/gpu:0')
        # self.history_batch = 1000
        self.history_batch = 1000
        self.replay_factor = 100
        self.start_learning = self.history_batch * self.replay_factor/2
        self.max_history_len = self.start_learning * 2
        self.past_size = 100  # just for statistics

        # (prev_s, prev_a, prev_r, prev_Qs, s, best_a, r, done)
        self.history = {}
        self.histr_count = 0

        self.game_counter = -1
        self.tot_score = 0
        self.past_episodes = []

        return

    def worker(self, num):
        scores_log = open("logs/scores.log", "w")
        env = gym.make(self.game_to_play)

        tot_score = 0
        past_counter = 0

        while True:
            self.game_counter += 1
            start_state = env.reset()
            curr_score = 0
            st_g = self.prepro(start_state)
            s = np.stack([st_g, st_g, st_g, st_g])
            frame_count = 0
            prev_s = s
            prev_a = 0
            [prev_Qs] = self.target_sess.run(self.graph_target.Q4actions, feed_dict={self.graph_target.state: [prev_s]})
            prev_r = 0
            done = False
            while not done:
                [Qs_for_action] = self.main_sess.run(self.graph_main.Q4actions, feed_dict={self.graph_main.state: [s]})

                best_a = np.argmax(Qs_for_action)
                if self.game_counter < self.random_games or random.random() < self.get_epsilon(self.game_counter):
                    next_action = env.action_space.sample()
                else:
                    next_action = best_a
                s_n, r, done, _ = env.step(next_action)
                s_ng = self.prepro(s_n)
                # add a lock!
                self.history[self.histr_count] = (prev_s, prev_a, prev_r, prev_Qs, s, best_a, r, done)
                # print(len(self.history))

                tot_score += r
                curr_score += r
                prev_r = r
                prev_a = next_action
                prev_s = s
                s = np.stack([prev_s[1], prev_s[2], prev_s[3], s_ng])  # move to the next state
                prev_Qs = np.copy(Qs_for_action)
                frame_count += 1
                self.histr_count += 1
                if (self.histr_count > self.max_history_len):
                    self.histr_count = 0
                # if (self.game_counter % self.show_every) == 0 and num == 0:  # only main thread shows game
                #     env.render()

            if past_counter < len(self.past_episodes):
                self.past_episodes[past_counter] = curr_score
            else:
                self.past_episodes.append(curr_score)
            past_counter += 1
            if past_counter > self.past_size:
                past_counter = 0

            if (self.game_counter % 1) == 0:
                # print(Qs_for_action)
                str1 = "round %d:" % self.game_counter
                print(str1)
                str2 = "current score:%f" % curr_score
                print(str2)
                str3 = "current average:%f" % (tot_score / (self.game_counter + 1))
                print(str3)
                str4 = "past episodes average:%f" % (sum(self.past_episodes) / len(self.past_episodes))
                print(str4)

                to_file = str1 + ", " + str2 + ", " + str3 + ", " + str4
                scores_log.write(to_file + "\n")

                scores_log.flush()
                os.fsync(scores_log.fileno())

        return

    def learn(self):
        while True:

            if len(self.history) > self.start_learning:

                if self.tot_updates % 100 == 0:
                    print("updating: self.histr_count=%d self.tot_updates=%d" % (self.histr_count, self.tot_updates))

                # extract random episodes
                next_states_for_target_Q = []
                selected_replays = []

                random_sample = random.sample(self.history.keys(), self.history_batch)
                for episode_key in random_sample:  # list makes a copy of keys (since changing dict in iterations)
                    # if random.random() < 1.0 / self.replay_factor:
                    episode_val = self.history.get(episode_key)
                    selected_replays.append(episode_val)
                    (s, a, r, curr_Qs, next_s, next_best_a, next_r, next_done) = episode_val
                    next_states_for_target_Q.append(next_s)
                next_Qs = self.target_sess.run(self.graph_target.Q4actions,
                                               feed_dict={self.graph_target.state: np.array(next_states_for_target_Q)})
                # all this can probably done faster in matrix notation...
                curr_states = []  # np.array()
                target_Qs = []  # np.array()
                for (i, rep) in enumerate(selected_replays):
                    (s, a, r, curr_Qs, next_s, next_best_a, next_r, next_done) = rep
                    curr_states.append(s)
                    target_Q = np.copy(curr_Qs)
                    target_Q[a] = r + self.gamma * next_Qs[i][next_best_a]
                    if next_done:
                        target_Q = np.zeros(self.num_of_actions) + next_r
                    target_Qs.append(target_Q)
                np_curr_states = np.array(curr_states)
                # print(np_curr_states.shape)
                np_target_Qs = np.array(target_Qs)
                [curr_loss, _] = self.main_sess.run([self.graph_main.loss, self.graph_main.update],
                                                    feed_dict={self.graph_main.state: np_curr_states,
                                                               self.graph_main.target_: np_target_Qs})
                curr_loss = np.sum(curr_loss)
                if math.isnan(curr_loss):
                    print("error in frame%d loss=NaN")

                self.tot_loss += curr_loss
                self.tot_updates += 1
                if (self.tot_updates % self.save_every_updates) == 0 and self.tot_updates > self.last_saved:
                    print("curr_loss=", curr_loss)


            if (self.tot_updates % self.save_every_updates) == 0 and self.tot_updates > self.last_saved:
                self.last_saved = self.tot_updates
                if not os.path.exists(self.checkpoint_dir_name):
                    os.makedirs(self.checkpoint_dir_name)
                filename = self.checkpoint_dir_name + '/' + str(self.tot_updates) + '-model.ckpt'
                self.graph_main.saver.save(self.main_sess, filename)
                print("updated checkpoints: tot_loss=", self.tot_loss)
                self.tot_loss = 0
                # update target
                self.graph_target.saver.restore(self.target_sess, filename)

                with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([self.game_counter, self.tot_score, self.past_episodes,self.history], f)

            time.sleep(0.01)
        return

    def go(self, restore, num_of_updates=1000):
        if __name__ == '__main__':

            self.target_sess = tf.Session(graph=self.graph_target.graph)
            self.tot_updates = 0
            self.tot_loss = 0
            self.last_hist_count = 0
            self.last_saved = -1
            self.main_sess = tf.Session(graph=self.graph_main.graph)

            if restore:
                self.tot_updates = num_of_updates
                filename = self.checkpoint_dir_name + '/' + str(self.tot_updates) + '-model.ckpt'
                self.graph_main.saver.restore(self.main_sess, filename)
                self.graph_target.saver.restore(self.target_sess, filename)
                self.tot_updates += 1

                # Getting back the objects:
                with open('objs.pkl', 'rb') as f:
                    self.game_counter, self.tot_score, self.past_episodes, self.history = pickle.load(f)

            else:
                self.target_sess.run(self.graph_target.init)
                self.main_sess.run(self.graph_main.init)
            jobs = []
            for i in range(1):
                p = threading.Thread(target=self.worker, args=(i,))
                jobs.append(p)
                p.start()

            self.learn()

        return


gm = GamePlayer()
gm.go(False, 6150)
