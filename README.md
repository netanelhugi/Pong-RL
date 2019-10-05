# Reinforcement Learning On Atari Pong

![](https://i.ibb.co/4RTSVhn/Figure-1.png)

Atari 2600 Pong is a game environment provided on the OpenAI “Gym” platform. Pong is a two-dimensional sport game that simulates table tennis which released it in 1972 by Atari. The player controls an in-game paddle by moving it vertically across the left or right side of the screen. They can compete against another player controlling a second paddle on the opposing side. Players use the paddles to hit a ball back and forth and have three action (“stay”, “down”, and “up”). The goal is for each player to reach 21 points before the opponent, points are earned when one fails to return the ball to the other. The OpenAI “gym” platform is a toolkit for developing and comparing reinforcement learning algorithms. It support teaching agents everything from walking to playing games. The unique feature it’s has is that we can train NN (Neural network) on one game and use it to play other games in the gym environment. We create four AI agents that generates the optimal actions, two of them taking raw pixels as features. One by feeding them into a convolutional neural network (CNN), also known as deep Q-learning. One by simple Q-learning The two other use features given before training to generate the optimal actions.

**For full details, read the final report.**
