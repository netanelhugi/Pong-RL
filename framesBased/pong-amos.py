import tensorflow as tf
import numpy as np
import random
import gym
import pdb
import math
#import cv2
import os
import threading
import time

import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 105x80 2D matrix """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I #.astype(np.float).ravel()

class QCNNGraph:
    def __init__(self, num_of_actions, device):
        with tf.device(device):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.state = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])

                self.conv1 = tf.contrib.layers.convolution2d(inputs=self.state,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', activation_fn=tf.nn.relu)
                self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', activation_fn=tf.nn.relu)
                self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', activation_fn=tf.nn.relu)
                self.flattend = tf.contrib.layers.flatten(self.conv3)
                self.fc1 = tf.contrib.layers.fully_connected(self.flattend, num_outputs=512, activation_fn=tf.nn.relu)
                self.Q4actions = tf.contrib.layers.fully_connected(self.fc1, num_outputs=num_of_actions, activation_fn=None)

                self.Q_n = tf.placeholder(tf.float32, shape=[None, num_of_actions])
                self.loss = tf.reduce_mean(tf.pow(self.Q4actions - self.Q_n, 2))
                self.update = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

                self.init = tf.global_variables_initializer()

                self.saver = tf.train.Saver()

class GamePlayer:

    def get_epsilon(self, curr_game):
     if curr_game > self.epsilon_decay_time:
            return self.epsilon_end
     return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * float(curr_game)/self.epsilon_decay_time

    def __init__(self):
        self.game_to_play = 'Pong-v0' #'Breakout-v0' #'MsPacman-v0'
        env = gym.make(self.game_to_play)
        self.num_of_actions = env.action_space.n
        self.show_every = 100
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay_time = 10000 #30000
        self.gamma = 0.99
        self.random_games = 2000#5000
        self.save_every_updates = 1000#also updates_to_update_target 1000
        self.checkpoint_dir_name = 'checkpoints'

        self.graph_target = QCNNGraph(self.num_of_actions, '/gpu:2')
        self.graph_main = QCNNGraph(self.num_of_actions, '/gpu:0')
        #self.history_length = 3000 #000
        self.history_batch = 1000
        self.replay_factor = 100
        self.start_learning = self.history_batch * self.replay_factor
        self.max_history_len =  self.start_learning * 2
        self.past_size = 100 #just for statistics

        #(prev_s, prev_a, prev_r, prev_Qs, s, best_a, r, done)
        self.history = {} #([self.history_length, 84, 84, 4])
        #self.history_corrected_Qs = np.zeros([self.history_length, env.action_space.n])
        self.histr_count = 0

        self.game_counter = -1
        self.tot_score = 0
        self.past_episodes = []

        return

    def worker(self, num):
        scores_log = open("logs/scores.log", "w")
        env = gym.make(self.game_to_play)
        #sess = tf.Session(graph=self.graph_target.graph)#, config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1))
        #sess.run(self.graph_target.init)
        print("num=",num)
        self.tot_score = 0
        past_counter = 0

        while True:
            self.game_counter += 1
            start_state = env.reset()
            curr_score = 0
            st_g = prepro(start_state)#cv2.resize(cv2.cvtColor(start_state, cv2.COLOR_RGB2GRAY), (84, 84)) #convert to 84x84 and grayscale
            s = np.stack([st_g,st_g,st_g,st_g],2)
            frame_count = 0
            prev_s = s
            prev_a = 0
            [prev_Qs] = self.target_sess.run(self.graph_target.Q4actions, feed_dict={self.graph_target.state:[prev_s]})
            prev_r = 0
            done = False
            while not done:
                #all_Qs = self.target_sess.run(self.graph_target.Q4actions, feed_dict={self.graph_target.state:[s]})
                [Qs_for_action] = self.main_sess.run(self.graph_main.Q4actions, feed_dict={self.graph_main.state:[s]})
                #Q_corrected = np.copy(all_Qs)
                #Q_corrected[0][prev_a] = prev_r  + self.gamma * np.max(all_Qs)
                best_a = np.argmax(Qs_for_action)
                if self.game_counter < self.random_games or random.random() < self.get_epsilon(self.game_counter):
                    next_action = env.action_space.sample()
                else:
                    next_action = best_a
                s_n, r, done, _ = env.step(next_action)
                s_ng = prepro(s_n) #cv2.resize(cv2.cvtColor(s_n, cv2.COLOR_RGB2GRAY), (84, 84)) #convert to 84x84 and grayscale
                #if (done):
                    #Q_corrected = np.zeros(env.action_space.n) + r
                #add a lock!
                self.history[self.histr_count] = (prev_s, prev_a, prev_r, prev_Qs, s, best_a, r, done)
                #self.history_states[self.histr_count%self.history_length,:] = prev_s
                #self.history_corrected_Qs[self.histr_count%self.history_length,:] = Q_corrected
                self.tot_score += r
                curr_score += r
                prev_r = r
                prev_a = next_action
                prev_s = s
                s = np.stack([prev_s[:,:,1], prev_s[:,:,2], prev_s[:,:,3], s_ng],2) #move to the next state
                prev_Qs = np.copy(Qs_for_action)
                frame_count += 1
                self.histr_count += 1
                if (self.histr_count > self.max_history_len):
                    self.histr_count = 0
                if (self.game_counter % self.show_every) == 0  and num==0 : #only main thread shows game
                    env.render()
                #self.check_for_update()
            if (past_counter < len(self.past_episodes)):
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
                str3 = "current average:%f" % (self.tot_score / (self.game_counter + 1))
                print(str3)
                str4 = "past episodes average:%f" % (sum(self.past_episodes) / len(self.past_episodes))
                print(str4)

                to_file = str1 + ", " + str2 + ", " + str3 + ", " + str4
                scores_log.write(to_file + "\n")

                scores_log.flush()
                os.fsync(scores_log.fileno())

        return

    def learn(self):
        while (True):
            if len(self.history) > self.start_learning: # or done:
                print("updating: self.histr_count=%d self.tot_updates=%d" %(self.histr_count,self.tot_updates))
                # extract random episodes
                next_states_for_target_Q = [] #np.array()
                selected_replays = []
                random_sample = random.sample(self.history.keys(), self.history_batch)
                for episode_key in random_sample: # list makes a copy of keys (since changing dict in iterations)
                  #if random.random() < 1.0 / self.replay_factor:
                    episode_val = self.history.get(episode_key)
                    selected_replays.append(episode_val)
                    (s, a, r, curr_Qs, next_s, next_best_a, next_r, next_done) = episode_val
                    next_states_for_target_Q.append(next_s)
                next_Qs = self.target_sess.run(self.graph_target.Q4actions, feed_dict={self.graph_target.state: np.array(next_states_for_target_Q)})
                # all this can probably done faster in matrix notation...
                curr_states = [] #np.array()
                target_Qs = [] #np.array()
                for (i,rep) in enumerate(selected_replays):
                    (s, a, r, curr_Qs, next_s, next_best_a, next_r, next_done) = rep
                    curr_states.append(s)
                    target_Q = np.copy(curr_Qs)
                    target_Q[a] = r + self.gamma * next_Qs[i][next_best_a]
                    if next_done:
                        target_Q = np.zeros(self.num_of_actions) + next_r
                    target_Qs.append(target_Q)
                np_curr_states = np.array(curr_states)
                #print(np_curr_states.shape)
                np_target_Qs = np.array(target_Qs)
                [curr_loss,_] = self.main_sess.run([self.graph_main.loss,self.graph_main.update], feed_dict={self.graph_main.state:np_curr_states, self.graph_main.Q_n:np_target_Qs})
                curr_loss = np.sum(curr_loss)
                if math.isnan(curr_loss):
                    print("error in frame%d loss=NaN")

                self.tot_loss += curr_loss
                self.tot_updates += 1
                print("curr_loss=", curr_loss)
                #pdb.set_trace()
                #sess.run(update, feed_dict={state:[s], Q_n: Q_corrected})
            #else:

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

    def go(self, restore, num_of_updates = 1000):
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
                with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                    self.game_counter, self.tot_score, self.past_episodes, self.history = pickle.load(f)

            else:
                self.target_sess.run(self.graph_target.init)
                self.main_sess.run(self.graph_main.init)
            jobs = []
            for i in range(1): #range(min(self.max_cores, multiprocessing.cpu_count() - 1)):
                p = threading.Thread(target=self.worker, args=(i,))
                jobs.append(p)
                p.start()
            #self.worker(0)
            self.learn()

            #for p in jobs:
                #p.join()

        return
gm = GamePlayer()
gm.go(False, 0)


