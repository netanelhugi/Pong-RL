##################
# Import Libraries
##################
import threading
import random
import time
import gym  # Game environment.
import numpy as np  # Handle matrices.
import pickle  # Save and restore data package.
from collections import deque  # For stacking states.
import tensorflow as tf  # Deep Learning library.
import datetime
import glob

# Ignore warning messages.
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
saveEvery = 100  # Save the model every few games.

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start

explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.000000001  # 0.00000001 exponential decay rate for exploration prob

memory_size = 300000  # Number of experiences the Memory can keep

rewards_list = []  # list of all training rewards.

# MODIFY THIS TO FALSE IF IS NOT THE FIRST TRAINING EPISODE.
firstTraining = True


# firstTraining = False


################
# Neural Network
################
class DQNetwork:
    def __init__(self, device):
        with tf.device(device):
            self.graph = tf.Graph()

            with self.graph.as_default():
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
                self.output = self.t3

                self.loss = tf.reduce_mean(tf.pow(self.output - self.target_, 2))
                self.update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

                # self.secondsCounter = tf.Variable(.0)
                # Tensor flow variables:
                # Episodes counter:
                self.episodeCounter = tf.Variable(1)
                self.step = tf.constant(1)
                self.update = tf.assign(self.episodeCounter, self.episodeCounter + self.step)
                # Time counter:
                self.secondsCounter = tf.Variable(.0)
                # Initialize the decay rate (that will use to reduce epsilon):
                self.decay_step = tf.Variable(0)
                self.decay_stepVar = 0
                self.min_decay_rate = False

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()


####################
# Experiences memory
####################
class Memory:

    # Init deque for the memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add experience to memory:
    def add(self, experience):
        self.buffer.append(experience)

    # Take random 'size' experiences from memory:
    def sample(self, size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=size, replace=False)

        # Obtain random mini-batch from memory
        batch = [self.buffer[i] for i in index]

        states_mb = np.array([each[0] for each in batch], ndmin=2)
        actions_mb = np.array([each[1] for each in batch])
        prev_rewards_mb = np.array([each[2] for each in batch])
        qs_mb = np.array([each[3] for each in batch], ndmin=2)
        next_states_mb = np.array([each[4] for each in batch])
        best_action_mb = np.array([each[5] for each in batch])
        rewards_mb = np.array([each[6] for each in batch])
        dones_mb = np.array([each[7] for each in batch])

        return states_mb, actions_mb, prev_rewards_mb, qs_mb, next_states_mb, best_action_mb, rewards_mb, dones_mb

    # Get all experiences:
    def get_all_memory(self):
        return self.buffer

    # Get the size of the memory:
    def get_memory_size(self):
        return len(self.buffer)

    # Get max size of the memory:
    def get_capacity(self):
        return self.buffer.maxlen


###############
# My functions:
###############
# State to vector function:
# Argument: state - matrix of pixels.
# Return: vector of [P1,P2,xBall,yBall]
def state_to_vector(state):
    state = state[35:195]  # crop
    state = state[::2, ::2, 0]  # downsample by factor of 2

    # [ballX,ballY,P1,P2]
    vector = [0, 0, 0, 0]

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
                    vector[0] = j
                    vector[1] = i
                    ball_flag = True
        if p1_flag and p2_flag and ball_flag:
            break

    return vector

# Get time vector:
# Argument: counter of seconds from the starting training.
# Return: vector of: [DAYS,HOURS,MINUTES,SECONDS].
def get_time(counter):
    time_vector = []

    day = counter // (24 * 3600)
    time_vector.append(day)

    counter = counter % (24 * 3600)
    hour = counter // 3600
    time_vector.append(hour)

    counter %= 3600
    minutes = counter // 60
    time_vector.append(minutes)

    seconds = counter % 60
    time_vector.append(seconds)

    return time_vector


# Print the action(DOWN,UP,STAY):
# Argument: action - 0/1/2/3/4/5
def action_to_string(action):
    if action == 1 or action == 0:
        print("STAY")
    elif action == 2 or action == 4:
        print("UP")
    elif action == 3 or action == 5:
        print("DOWN")


# Return log file:
# If is the first training - create log file.
# Else - append to the old log file.
def get_log_file(trainingNumber):
    if firstTraining:
        # Create log file:
        exists = os.path.isfile("./saveData/log_training_number_" + str(trainingNumber) + ".txt")

        if exists:
            print("The number of training already exists, select a new number or change \"firstTraining\" to False.")
            exit()
        else:
            log = open("featuresBased_log_training_number_" + str(trainingNumber) + ".txt", "w")
            now = datetime.datetime.now()

            trinHeader = "Training number: " + str(trainingNumber) + "\n" + "Start Time: " + str(
                now) + "\n" + "Learning rate: " + str(learning_rate) + "\n"
            log.write(trinHeader)

            log.flush()
            os.fsync(log.fileno())
            # log.close()
    else:
        log = open("featuresBased_log_training_number_" + str(trainingNumber) + ".txt", "a")

    return log


################
# Initialization
################


class CreateGame:

    def __init__(self, training_number):
        # Create our environment:
        self.env = gym.make('Pong-v0')
        # Initialize deque with zero-vectors states:
        self.stacked_vectors = deque([np.zeros(state_size, dtype=np.float) for i in range(stack_size)], maxlen=4)
        # Instantiate the DQNetwork:
        self.graph_main = DQNetwork('/gpu:0')
        # # Instantiate memory:
        # self.memory = self.init_memory()

        # Create log file:
        self.logFile = get_log_file(training_number)

        self.checkpoint_dir_name = 'checkpoints'

        # Tensor flow variables:
        # Episodes counter:
        self.episodeCounter = tf.Variable(1)
        self.step = tf.constant(1)
        self.update = tf.assign(self.episodeCounter, self.episodeCounter + self.step)
        # Time counter:
        self.secondsCounter = tf.Variable(.0)
        # Initialize the decay rate (that will use to reduce epsilon):
        self.decay_step = tf.Variable(0)
        self.decay_stepVar = 0
        self.min_decay_rate = False

        self.random_counter = 0
        self.random_games = 200
        self.play_random = True

        self.episode_render = False
        self.print_actions = False
        self.print_q_values = False
        self.test_next_game = False
        self.games_for_test = 1

    def init_memory(self):
        temp_memory = Memory(max_size=memory_size)

        if not firstTraining:
            # restore memory data:
            with open("./saveData/memory.dq", "rb") as fp:
                temp = pickle.load(fp)
            # Add to memory buffer:
            for i in temp:
                temp_memory.add(i)

        # else:
        #     state = self.env.reset()
        #     prev_s = self.stack_states(state, True)
        #     prev_a = 0
        #
        #     [prev_qs] = self.main_sess.run(self.graph_main.output,
        #                                    feed_dict={self.graph_main.inputs_: [prev_s]})
        #     # prev_qs = [.0, .0, .0, .0, .0, .0]
        #     prev_r = 0
        #
        #     # Init memory with states:
        #     for i in range(batch_size):
        #         # next_qs = [.0, .0, .0, .0, .0, .0]
        #         [next_qs] = self.main_sess.run(self.graph_main.output, feed_dict={self.graph_main.inputs_: [prev_s]})
        #
        #         # Get the next_state, the rewards, done by taking a random action
        #         action = random.randint(1, action_size) - 1
        #         next_state, reward, done, info = self.env.step(action)
        #         next_state = self.stack_states(next_state, False)
        #         best_action = action
        #
        #         # Add experience to memory
        #         temp_memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, best_action, reward, done))
        #
        #         # Our new state is now the next_state
        #         prev_s = next_state
        #         prev_a = best_action
        #         prev_r = reward
        #         prev_qs = next_qs
        #
        #     self.env.close()

        return temp_memory

    # Predict action function: predict the next action:
    # Arguments: 1. sess - tensorflow session.
    #            2. DQNetwork2 - neural network model.
    #            3. explore_start - 1.0(const), for epsilon greedy strategy.
    #            4. explore_stop - 0.1(const), for epsilon greedy strategy.
    #            5. decay_rate - variable, for reducing the selection of a random step during the game.
    #            6. decay_step - variable, for reducing the selection of a random step during the game.
    #            7. state - matrix/vector of the current state.
    #            8. actions - possible actions.
    # Return: 1. action - the predicted action.
    #         2. explore_probability - the current probability for taking random action.
    #         3. qs - predicted Q-values.
    def predict_action(self, state):
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        if not self.min_decay_rate:
            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
                -decay_rate * self.main_sess.run(self.graph_main.decay_step))

            if explore_probability < explore_stop + 0.01:
                self.min_decay_rate = True
        else:
            explore_probability = explore_stop

        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        # state.reshape((1, *state.shape)): reshape from [4,4] to [1,4,4]
        # the input of the net is: [none,4,4].
        qs = self.main_sess.run(self.graph_main.output,
                                feed_dict={self.graph_main.state: [state]})

        if explore_probability > exp_exp_tradeoff or self.play_random:
            # Make a random action (exploration)
            action = random.randint(1, action_size) - 1
        else:
            # Take the biggest Q value (= the best action)
            action = np.argmax(qs)

            if self.print_actions:
                action_to_string(action)
            if self.print_q_values:
                print(qs)

        return action, explore_probability, qs

    # stack_states function:
    # Arguments: 1. state - (matrix) vector of current state.
    #            2. is_new_episode - (boolean) check if we start an new episode.
    # Return: 1. stacked_state - (numpy stack).
    def stack_states(self, state, is_new_episode):
        # Preprocess frame
        state_vec = state_to_vector(state)
        state_vec = np.asarray(state_vec)

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_vectors = deque([np.zeros(state_size, dtype=np.int) for i in range(stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same state 4x
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)
            self.stacked_vectors.append(state_vec)

            # Stack the frames
            stacked_state = np.stack(self.stacked_vectors)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_vectors.append(state_vec)
            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_vectors)

        return stacked_state

    def save_model(self):
        episode = self.main_sess.run(self.graph_main.episodeCounter)
        filename = self.checkpoint_dir_name + '/' + str(episode) + '-model.ckpt'
        self.graph_main.saver.save(self.main_sess, filename)
        print("Model Saved")

        # Save memory data:
        with open("featuresBasrd_memory.dq", "wb") as fp:  # Pickling
            pickle.dump(self.memory.get_all_memory(), fp)

        # Save model data:
        with open("FeaturesBasrd_model-" + str(episode) + ".ml", "wb") as mo:  # Pickling
            out = self.main_sess.run(self.graph_main.output,
                                     feed_dict={self.graph_main.state: [np.stack(self.stacked_vectors)]})
            t3 = self.main_sess.run(self.graph_main.t3,
                                    feed_dict={self.graph_main.state: [np.stack(self.stacked_vectors)]})
            t2 = self.main_sess.run(self.graph_main.t2,
                                    feed_dict={self.graph_main.state: [np.stack(self.stacked_vectors)]})
            t1 = self.main_sess.run(self.graph_main.t1,
                                    feed_dict={self.graph_main.state: [np.stack(self.stacked_vectors)]})
            t3 = self.main_sess.run(self.graph_main.t3,
                                    feed_dict={self.graph_main.state: [np.stack(self.stacked_vectors)]})

            ep = self.main_sess.run(self.graph_main.episodeCounter)
            sec = self.main_sess.run(self.graph_main.secondsCounter)
            dec = self.main_sess.run(self.graph_main.decay_step)

            to_file = [out, t3, t2, t1, ep, sec, dec]

            pickle.dump(to_file, mo)


    def print_timer(self):
        # Get and print total training time:
        time_vector = get_time(self.main_sess.run(self.graph_main.secondsCounter))
        print("Ep: %d" % self.main_sess.run(self.graph_main.episodeCounter), ",Total time: D:%d,H:%d,M:%d,S:%d" % (
            int(time_vector[0]), int(time_vector[1]), int(time_vector[2]), int(time_vector[3])))

    def get_game_summery(self, total_reward, explore_probability, loss):
        # Print episode summery:
        print('Episode: {}'.format(self.main_sess.run(self.graph_main.episodeCounter)),
              'Total reward: {}'.format(total_reward),
              'Explore P: {:.4f}'.format(explore_probability),
              'Training Loss {}'.format(loss))

        # Send the summery to log file:
        time_vector = get_time(self.main_sess.run(self.graph_main.secondsCounter))

        str2 = "Time: D: " + str(int(time_vector[0])) + "H: " + str(int(time_vector[1])) + "M: " + str(
            int(time_vector[2])) + "S: " + str(
            int(time_vector[3])) + ", Episode: " + str(
            self.main_sess.run(self.graph_main.episodeCounter)) + ", Total reward:" + str(
            total_reward) + ", Explore P: " + str(explore_probability) + ", loss: " + str(
            loss) + "\n"
        self.logFile.write(str2)

        self.logFile.flush()
        os.fsync(self.logFile.fileno())

    def terminal_input(self):
        while True:
            inp = input()

            inp = str(inp)

            cmd = inp[0]

            if cmd == "i":
                print("Commands list:")
                print("\"r\" - enable/disable episode render.")
                print("\"a\" - enable/disable printing actions.")
                print("\"t<number>\" - test the model in the next episode(for <number> episodes).")
                print("\"q\" - enable/disable printing q-values.")
                print("\"e\" - to exit program.")


            elif cmd == "r":
                self.episode_render = not self.episode_render
            elif cmd == "a":
                self.print_actions = not self.print_actions
            elif cmd == "t":
                num = int(inp[1])
                self.test_next_game = True
                self.games_for_test = num
            elif cmd == "q":
                self.print_q_values = not self.print_q_values
            elif cmd == "e":
                print("exit")
                exit()

            time.sleep(1)

    ##########
    # Training
    ##########

    def training(self):

        self.main_sess = tf.Session(graph=self.graph_main.graph)

        if firstTraining:
            self.main_sess.run(self.graph_main.init)

        else:
            self.graph_main.saver.restore(self.main_sess, tf.train.latest_checkpoint('./checkpoints'))
            print(self.main_sess.run(self.secondsCounter))

        # Instantiate memory:
        self.memory = self.init_memory()

        for episode in range(total_episodes):
            start_time_ep = time.time()  # Start episode time.
            # Print total training time:
            self.print_timer()

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            state = self.env.reset()
            state = self.stack_states(state, True)

            prev_s = state
            prev_a = 0
            [prev_qs] = self.main_sess.run(self.graph_main.output,
                                           feed_dict={self.graph_main.state: [prev_s]})
            prev_r = 0
            done = False

            while not done:
                # Increase decay_step
                if not self.play_random:
                    self.decay_stepVar += 1

                # Predict the next action:
                action, explore_probability, [next_qs] = self.predict_action(state)
                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, info = self.env.step(action)

                # Game display:
                if self.episode_render:
                    self.env.render()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_state = np.zeros(original_state_size, dtype=np.int)
                    next_state = self.stack_states(next_state, False)

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    # print summery and write to log
                    self.get_game_summery(total_reward, explore_probability, loss)

                    # Add reward to total rewards list:
                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    self.memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, action, reward, done))


                else:
                    next_state = self.stack_states(next_state, False)
                    # Add experience to memory
                    self.memory.add((prev_s, prev_a, prev_r, prev_qs, next_state, action, reward, done))

                prev_s = next_state
                prev_a = action
                prev_r = reward
                prev_qs = next_qs

                # LEARNING PART
                # Obtain random mini-batch from memory
                if not self.play_random:
                    batch = self.memory.sample(batch_size)
                    states_mb, actions_mb, prev_rewards_mb, qs_mb, next_states_mb, best_action_mb, rewards_mb, dones_mb = batch
                    target_qs_batch = []

                    # Get Q values for next_state
                    qs_next_state = self.main_sess.run(self.graph_main.output,
                                                       feed_dict={self.graph_main.inputs_: next_states_mb})

                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, batch_size):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target = np.zeros(action_size) + rewards_mb[i]
                            target_qs_batch.append(target)

                        else:
                            target = np.copy(qs_mb[i])
                            a = actions_mb[i]
                            r = prev_rewards_mb[i]
                            best = best_action_mb[i]
                            target[a] = r + gamma * qs_next_state[i][best]
                            target_qs_batch.append(target)

                    targets_mb = np.array([each for each in target_qs_batch])

                if not self.play_random:
                    loss, _ = self.main_sess.run([self.graph_main.loss, self.graph_main.optimizer],
                                                 feed_dict={self.graph_main.inputs_: states_mb,
                                                            self.graph_main.target_: targets_mb})
                else:
                    loss = 1000000

            # Update episode number:
            self.main_sess.run(self.graph_main.update)

            # Time update:
            end_time_ep = time.time()
            time_update = tf.assign_add(self.graph_main.secondsCounter, end_time_ep - start_time_ep)
            self.main_sess.run(time_update)

            # Decay update:
            if not self.play_random:
                decay_step_update = tf.assign_add(self.graph_main.decay_step, self.decay_stepVar)
                self.main_sess.run(decay_step_update)

            self.random_counter += 1

            if self.random_counter > self.random_games:
                self.play_random = False

            # Save model every 100 episodes or before testing
            if episode % saveEvery == 0 or self.test_next_game:
                self.save_model()
                self.testing(self.games_for_test)
                self.test_next_game = False
                self.games_for_test = 1

    #########
    # Testing
    #########

    def testing(self, games):

        self.env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True, force=True)

        total_test_rewards = []

        for episode in range(games):
            total_rewards = 0

            state = self.env.reset()
            state = self.stack_states(state, True)

            print("TEST EPISODE")

            while True:
                # Get action from Q-network
                # Estimate the Qs values state
                qs = self.main_sess.run(self.graph_main.output,
                                        feed_dict={self.graph_main.state: [state]})

                # Take the biggest Q value (= the best action)
                action = np.argmax(qs[0])

                if self.print_actions:
                    action_to_string(action)
                if self.print_q_values:
                    print(qs[0])

                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = self.env.step(action)

                if self.episode_render:
                    self.env.render()

                total_rewards += reward

                if done:
                    print("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break

                next_state = self.stack_states(next_state, False)
                state = next_state
            ep_num = self.main_sess.run(self.graph_main.episodeCounter)
            new_file_name = "./vid/test_game_episode:" + str(ep_num) + "_result:" + str(total_rewards) + ".mp4"

            files = glob.glob('./vid/*.mp4')
            last_video = max(files, key=os.path.getctime)
            os.rename(last_video, new_file_name)

            testSumm = "Test episode: Result= " + str(total_rewards) + "\n"
            self.logFile.write(testSumm)

            self.logFile.flush()
            os.fsync(self.logFile.fileno())

        self.env.close()


######
# Main
######
if __name__ == "__main__":
    pong = CreateGame(training_number=10)

    get_terminal_input = threading.Thread(target=pong.terminal_input)
    get_terminal_input.daemon = True
    get_terminal_input.start()

    pong.training()
