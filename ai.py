import time
import tensorflow as tf
import numpy as np

class NeuralNet:
    def __init__(self, session, nb_hidden, nb_outputs, learning_rate):
        self.iterations = 0
        self.nb_outputs = nb_outputs
        self.parameters = []
        self.session = session
        summaries = []
        self.x = tf.placeholder(tf.float32, shape=(None, 3, 3, 3), name='input')
        filter1 = tf.Variable(0.1*np.random.randn(2, 2, 3, 128), dtype=tf.float32, name='filter1')
        conv1 = tf.nn.relu(tf.nn.conv2d(self.x, filter1, [1,1,1,1], 'SAME'))
        max_pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,1,1,1], 'VALID')

        summaries.append(tf.summary.image('filter', tf.transpose(filter1, [3, 0, 1, 2])))
        self.parameters.append(filter1)
        prev_layer = tf.reshape(max_pool1, [-1, 512])
        prev_layer_size = int(prev_layer.shape[1])
        for i in range(len(nb_hidden)):
            sigma = np.sqrt(2/prev_layer_size)*.1
            W = tf.Variable(sigma*np.random.randn(prev_layer_size, nb_hidden[i]), dtype=tf.float32, name='W'+str(i))
            b = tf.Variable(np.zeros(nb_hidden[i]), dtype=tf.float32, name='b'+str(i))
            h = tf.nn.relu(tf.matmul(prev_layer, W) + b, name='hidden'+str(i))

            W_summary = tf.summary.histogram('weights'+str(i), W)
            b_summary = tf.summary.histogram('biases'+str(i), b)

            prev_layer = h
            prev_layer_size = nb_hidden[i]

            self.parameters.extend([W, b])
            summaries.extend([W_summary, b_summary])
            
        sigma = np.sqrt(2/prev_layer_size)
        W = tf.Variable(sigma*np.random.randn(prev_layer_size, nb_outputs), dtype=tf.float32, name='W'+str(len(nb_hidden)))
        b = tf.Variable(np.zeros(nb_outputs), dtype=tf.float32, name='b'+str(len(nb_hidden)))
        W_summary = tf.summary.histogram('weights'+str(len(nb_hidden)), W)
        b_summary = tf.summary.histogram('biases'+str(len(nb_hidden)), b)
        
        summaries.extend([W_summary, b_summary])
        self.parameters.extend([W, b])
        
        self.y = tf.add(tf.matmul(prev_layer, W), b, name='output')
        self.target_indexes = tf.placeholder(tf.int32, shape=[None], name='target_indexes')
        mask=tf.one_hot(self.target_indexes, self.y.shape[1], on_value=True, off_value=False)
        self.y_target = tf.placeholder(tf.float32, shape=(None), name='target_output')
        self.loss = tf.losses.huber_loss(self.y_target, tf.boolean_mask(self.y, mask))
        self.optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.001)
        self.train_step = self.optimizer.minimize(self.loss)
        
        self.softmax_indexes = tf.placeholder(tf.int32, name='softmax_indexes')
        self.softmax_temperature = tf.placeholder(tf.float32, name='softmax_temperature')
        self.probs = tf.nn.softmax(tf.gather(self.y/self.softmax_temperature, self.softmax_indexes, axis=1, name='probabilities'))
        
        loss_summary = tf.summary.scalar('loss', self.loss)
        summaries.append(loss_summary)
        self.summary = tf.summary.merge(summaries)

        self.writer = None

    def assignParameters(self, new_parameters):
        for i in range(len(self.parameters)):
            self.session.run(self.parameters[i].assign(new_parameters[i]))

    def learn(self, x, y_target, target_indexes):
        summary, _ = self.session.run(
            [self.summary, self.train_step],
            feed_dict={
                self.x : x,
                self.y_target : y_target,
                self.target_indexes : target_indexes
            }
        )
        if self.writer:# and self.iterations % 10 == 0:
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
        self.target_net_update_steps = 20
        self.batch_size = 500
        self.memory_capacity = 1000
        self.memory = ExperienceReplay(self.memory_capacity)
        self.last_state = None
        self.last_action = None
        self.session = tf.Session()

        self.target_network = NeuralNet(self.session, nb_hidden, 9, learning_rate)
        self.q_network = NeuralNet(self.session, nb_hidden, 9, learning_rate)

        logdir='/tmp/tensorboard/ia_velha/batch_size={}, steps_to_refresh={}, {} - {}'.format(self.batch_size, self.target_net_update_steps, [reward_decay, nb_hidden, learning_rate, softmax_temperature], time.ctime())
        self.writer = tf.summary.FileWriter(logdir, self.session.graph)
        self.q_network.writer = self.writer

        self.session.run(tf.global_variables_initializer())
        self.training_steps = 0
        
    def encode_board(self, board):
        map_dict = {'X':[1,0,0],'':[0,1,0],'O':[0,0,1]}
        return np.reshape(list(map(lambda x : map_dict[x], board)), (3,3,3))

    def choose_action(self, board):
        state = self.encode_board(board)
        available_actions = [i for i in range(len(board)) if not board[i]]
        if len(available_actions) > 0:
            # probs = self.q_network.predict_probs(
            #     [state],
            #     softmax_temperature=self.softmax_temperature,
            #     softmax_indexes=available_actions
            # )[0]
            # sorting_indexes = np.argsort(probs)
            # index = np.argmax(np.random.multinomial(1, probs[sorting_indexes]))
            # available_actions = np.array(available_actions)[sorting_indexes]
            # action = available_actions[index]
            # return action
            epsilon = 0.1
            if np.random.rand() < epsilon:
                return np.random.choice(available_actions)
            else:
                q_values = self.q_network.predict([state])[0][available_actions]
                return available_actions[np.argmax(q_values)]
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
        if self.last_state is not None:
            self.memory.append(self.last_state, new_state, self.last_action, reward)
        if len(self.memory) > 0:
            last_state_batch, new_state_batch, action_batch, reward_batch = self.memory.sample(self.batch_size)
            new_state_value = np.max(self.target_network.predict(new_state_batch), axis=1)
            target_q = reward_batch + self.reward_decay*new_state_value
            self.q_network.learn(last_state_batch, target_q, action_batch)
            if self.training_steps % self.target_net_update_steps == 0:
                self.target_network.assignParameters(self.q_network.parameters)

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