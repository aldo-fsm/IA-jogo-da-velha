import tensorflow as tf
import numpy as np
import time

class NeuralNet:
    def __init__(self, nb_inputs, nb_hidden, nb_outputs, learning_rate, logdir=None):
        self.logdir = logdir
        self.iterations = 0
        self.nb_outputs = nb_outputs
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=(None, nb_inputs), name='input')
        prev_layer = self.x
        prev_layer_size = nb_inputs
        for i in range(len(nb_hidden)):
            W = tf.Variable(0.1*np.random.randn(prev_layer_size, nb_hidden[i]), dtype=tf.float32, name='W'+str(i))
            b = tf.Variable(0.1*np.random.randn(nb_hidden[i]), dtype=tf.float32, name='b'+str(i))
            h = tf.nn.relu(tf.matmul(prev_layer, W) + b, name='hidden'+str(i))
            prev_layer = h
            prev_layer_size = nb_hidden[i]
        W = tf.Variable(0.1*np.random.randn(prev_layer_size, nb_outputs), dtype=tf.float32, name='W'+str(len(nb_hidden)))
        b = tf.Variable(0.1*np.random.randn(nb_outputs), dtype=tf.float32, name='b'+str(len(nb_hidden)))
        self.y = tf.add(tf.matmul(prev_layer, W), b, name='output')
        
        self.y_target = tf.placeholder(tf.float32, shape=(None, nb_outputs), name='target_output')
        self.loss = tf.losses.huber_loss(self.y_target, self.y)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        
        self.softmax_indexes = tf.placeholder(tf.int32, name='softmax_indexes')
        self.softmax_temperature = tf.placeholder(tf.float32, name='softmax_temperature')
        self.probs = tf.nn.softmax(tf.gather(self.y/self.softmax_temperature, self.softmax_indexes, axis=1, name='probabilities'))
        
        self.variable_initializer = tf.global_variables_initializer()
        self.session.run(self.variable_initializer)

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        if logdir:
            self.writer = tf.summary.FileWriter(logdir, self.session.graph)

    def learn(self, x, y_target):
        summary, _ = self.session.run(
            [self.loss_summary, self.train_step],
            feed_dict={
                self.x : x,
                self.y_target : y_target
            }
        )
        if self.logdir:
            self.writer.add_summary(summary, self.iterations)
        self.iterations += 1
        
    def predict_probs(self, x, softmax_temperature=1, softmax_indexes=None):
        if softmax_indexes == None:
            softmax_indexes = list(range(self.nb_outputs))
        return self.session.run(
            self.probs,
            feed_dict={
                self.x: x,
                self.softmax_temperature: softmax_temperature,
                self.softmax_indexes: softmax_indexes
            }
        )

    def predict(self, x):
        return self.session.run(
            self.y,
            feed_dict={
                self.x: x
            }
        )

class Dqn:
    def __init__(self, reward_decay, nb_hidden, learning_rate, softmax_temperature):
        self.reward_decay = reward_decay
        self.reward_window_length = 100
        self.reward_window = []
        self.softmax_temperature = softmax_temperature
        self.batch_size = 7000
        self.memory_capacity = 10000
        self.memory = ExperienceReplay(self.memory_capacity)
        self.last_state = None
        self.last_action = None
        self.brain = NeuralNet(9, nb_hidden, 9, learning_rate, logdir='/tmp/tensorboard/ia_velha/'+str(time.time()))
        # self.brain = NeuralNet(9, nb_hidden, 9, learning_rate)
        
    def encode_board(self, board):
        map_dict = {'X':-1,'':0,'O':1}
        return list(map(lambda x : map_dict[x], board))

    def choose_action(self, state):
        state = self.encode_board(state)
        available_actions = [i for i in range(len(state)) if not state[i]]
        if len(available_actions) > 0:
            probs = self.brain.predict_probs(
                [state],
                softmax_temperature=self.softmax_temperature,
                softmax_indexes=available_actions
            )[0]
            sorting_indexes = np.argsort(probs)
            index = np.argmax(np.random.multinomial(1, probs[sorting_indexes]))
            available_actions = np.array(available_actions)[sorting_indexes]
            action = available_actions[index]
            return action
        else:
            return -1

    def average_score(self):
        if len(self.reward_window) == self.reward_window_length:
            return np.mean(self.reward_window)
        else:
            return np.mean(self.reward_window+[0]*(self.reward_window_length-len(self.reward_window)))
    
    def update(self, new_state, reward):
        print(self.average_score(), len(self.memory))
        self.reward_window.append(reward)
        self.reward_window = self.reward_window[-self.reward_window_length:]
        action = self.choose_action(new_state)
        new_state = self.encode_board(new_state)
        if self.last_state:
            self.memory.append(self.last_state, new_state, self.last_action, reward)
        if len(self.memory) > 0:
            last_state_batch, new_state_batch, action_batch, reward_batch = self.memory.sample(self.batch_size)
            new_state_value = np.max(self.brain.predict(new_state_batch), axis=1)
            target_q = reward_batch + self.reward_decay*new_state_value
            current_q = self.brain.predict(last_state_batch)
            
            target_batch = []
            for i in range(len(target_q)):
                row = np.array(current_q[i])
                row[action_batch[i]] = target_q[i]
                target_batch.append(row)
            
            self.brain.learn(last_state_batch, target_batch)

        self.last_action = action
        self.last_state = new_state
        return action

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def append(self, last_state, new_state, action, reward):
        self.memory.append((last_state, new_state, action, reward))
        self.memory = self.memory[-self.capacity:]

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch = self.memory
        else:
            choices = np.random.choice(range(len(self.memory)), batch_size, replace=False)
            batch = np.array(self.memory)[choices]

        return list(zip(*batch))

    def __len__(self):
        return len(self.memory)