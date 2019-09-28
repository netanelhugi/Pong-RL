from argparse import ArgumentParser
import gym
import os
import pickle
import numpy as np
import tensorflow as tf
import random
INF = int(1e15)



# Load the data set from file:
Q = {}
with open('dataset_size:567524.pkl', 'rb') as f:
    Q = pickle.load(f)[0]

# divide to keys(states) and values(actions):
batch_xs, batch_ys = Q.keys(),Q.values()  #MB-GD
batch_xs = np.array(list(batch_xs))
batch_ys = np.array(list(batch_ys))

# train data
xs = []
ys = []

# test data
xs_test = []
ys_test = []

count = 0

for l in batch_xs:

    # convert each state to the form: [a,b,c,d,e,f]
    array = np.array([l[0][0],l[0][1],l[1],l[2],l[3][0],l[3][1]])

    # divide the data into train and test:
    if count < int(len(batch_xs)*0.7):
        xs.append(array)
    else:
        xs_test.append(array)

    count += 1

count = 0
for y in batch_ys:

    # divide the data into train and test:
    if count < int(len(batch_ys) * 0.7):
        ys.append(y)
    else:
        ys_test.append(y)

    count += 1
    ys.append(y)

# [stay,up,down]
def array_to_action(arr):

    index = np.argmax(arr)

    if index == 0:
        return 0
    elif index == 1:
        return 2
    else:
        return 3

def results(states, realValues):
    # Testing the model:
    true = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    for i in range(len(states)):
        state = states[i]
        real = realValues[i]

        # Predicted Value:
        action = y.eval(session=sess, feed_dict={x: [state]})
        pred = array_to_action(action[0])
        if (pred == array_to_action(real)):
            true += 1

    accuracy = true / len(states)
    # recall = truePositive / (truePositive + falseNegative)
    # precision = 0
    # fMeasure = 0
    #
    # if(truePositive + falsePositive>0):
    #     precision = truePositive / (truePositive + falsePositive)
    # if(precision + recall>0):
    #     fMeasure = 2 * (precision * recall) / (precision + recall)

    return accuracy

# Neural network:

# hidden layers size:
(hidden1_size, hidden2_size) = (10, 5)

x = tf.placeholder(tf.float32, [None, 6])
y_ = tf.placeholder(tf.float32, [None, 3])
W1 = tf.Variable(tf.truncated_normal([6, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x,W1)+b1)
W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)
W3 = tf.Variable(tf.truncated_normal([hidden2_size, 3], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[3]))
y = tf.nn.softmax(tf.matmul(z2, W3) + b3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    idx = np.random.choice(len(xs), size=512, replace=False)
    batch_xs = []
    batch_ys = []
    for n in idx:
        batch_xs.append(xs[n])
        batch_ys.append(ys[n])

    if i % 1000 == 0:
        print("iter: ",i," ,curr acc: " ,results(xs_test,ys_test))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# [stay,up,down]
def array_to_action(arr):

    index = np.argmax(arr)

    if index == 0:
        return 0
    elif index == 1:
        return 2
    else:
        return 3

# Auto agent - for building the dataset.
def get_action(state):
    ball = state[0]
    p_right = state[2]
    p_size = 8
    dir = state[3]

    if dir[1] < 0:
        return 0
    if dir == (0,0):
        return 0
    elif ball[0]+dir[0] >= p_right + p_size:
        return 3 #down
    elif ball[0]+dir[0] >= p_right and ball[0]+dir[0] < ball[0] + p_size:
        return 0 #stay
    elif ball[0]+dir[0] < p_right:
        return 2 #up


# Print the action(DOWN,UP,STAY):
# Argument: action - 0/1/2/3/4/5
def action_to_string(action):
    if action == 1 or action == 0:
        return "STAY"
    elif action == 2 or action == 4:
        return "UP"
    elif action == 3 or action == 5:
        return "DOWN"


class GamePlayer:

    def __init__(self):

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
        self.env = gym.make('Pong-v0')
        return self.prepro(self.env.reset())


    # Print the action(DOWN,UP,STAY):
    # Argument: action - 0/1/2/3/4/5
    def action_to_string(self,action):
        if action == 1 or action == 0:
            print("STAY")
        elif action == 2 or action == 4:
            print("UP")
        elif action == 3 or action == 5:
            print("DOWN")

    def play(self):
        episodes = 10000
        episodeStart = 1
        scores = []

        for episode in range(episodeStart, episodes + 1):

            state = self.reset_game()
            curr_score = 0
            done = False

            while not done:

                self.env.render()

                sarr = np.asarray(state)

                s = np.array([sarr[0][0], sarr[0][1], sarr[1], sarr[2], sarr[3][0], sarr[3][1]])
                action = y.eval(session=sess,feed_dict={x: [s]})
                action = array_to_action(action[0])

                # get action from the auto agent:
                # action = get_action(state)

                newState, reward, done, _ = self.env.step(action)
                curr_score += reward
                next_state = self.prepro(newState)
                state = next_state

            scores.append(curr_score)

            to_file = "%d,%d" %(episode,curr_score)
            self.scores_log.write(to_file + "\n")

            self.scores_log.flush()
            os.fsync(self.scores_log.fileno())

            print("Episode:", episode, ", Score:",curr_score)


if __name__ == '__main__':
    gp = GamePlayer()

    print(results(xs_test,ys_test))
    # gp.play()
