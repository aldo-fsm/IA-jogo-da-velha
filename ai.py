import tensorflow as tf
import numpy as np

class Dqn:
    def __init__(self, reward_decay, nb_hidden, learning_rate, softmax_temp=3):
        self.reward_decay = reward_decay
        self.memory = ActionReplay(1000)
        self.last_state = None
        self.last_action = None
        self.session = tf.Session()
        self.state = tf.placeholder(tf.float32, shape=(None, 9), name='input')
        
        W1 = tf.Variable(np.random.randn(9, nb_hidden), dtype=tf.float32, name='W1')
        b1 = tf.Variable(np.random.randn(nb_hidden), dtype=tf.float32, name='b1')
        h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1, name='hidden1')
        
        W2 = tf.Variable(np.random.randn(nb_hidden, 9), dtype=tf.float32, name='W2')
        b2 = tf.Variable(np.random.randn(9), dtype=tf.float32, name='b2')
        self.q_values = tf.add(tf.matmul(h1, W2), b2, name='q_values')
        self.available_actions = tf.placeholder(tf.int32, name='available_actions')
        self.probs = tf.nn.softmax(tf.gather(self.q_values/softmax_temp, self.available_actions, axis=1, name='action_probabilities'))

        self.target_q = tf.placeholder(tf.float32, shape=(None, 9), name='target_q_values')
        self.loss = tf.losses.huber_loss(self.target_q, self.q_values)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.variable_initializer = tf.global_variables_initializer()
        self.train_step = self.optimizer.minimize(self.loss)
    
        self.session.run(self.variable_initializer)

    def encode_board(self, board):
        map_dict = {'X':-1,'':0,'O':1}
        return list(map(lambda x : map_dict[x], board))

    def choose_action(self, state):
        state = self.encode_board(state)
        available_actions = [i for i in range(len(state)) if not state[i]]
        if len(available_actions) > 0:
            probs = self.session.run(
                self.probs,
                feed_dict={
                    self.state: [state],
                    self.available_actions: available_actions
                })[0]
            sorting_indexes = np.argsort(probs)
            index = np.argmax(np.random.multinomial(1, probs[sorting_indexes]))
            available_actions = np.array(available_actions)[sorting_indexes]
            action = available_actions[index]
            return action
        else:
            return -1

    def update(self, new_state, reward):
        print(len(self.memory))
        action = self.choose_action(new_state)
        new_state = self.encode_board(new_state)
        if self.last_state:
            self.memory.append(self.last_state, new_state, self.last_action, reward)
        if len(self.memory) > 0:
            last_state_batch, new_state_batch, action_batch, reward_batch = self.memory.sample(200)

            new_state_value = np.max(self.session.run(self.q_values,
                feed_dict={
                    self.state : new_state_batch
                }), axis=1)
            target_q = reward_batch + new_state_value
            current_q = self.session.run(self.q_values,
                feed_dict={
                    self.state : last_state_batch
                })
            target_batch = []
            for i in range(len(target_q)):
                row = np.array(current_q[i])
                row[action_batch[i]] = target_q[i]
                target_batch.append(row)
            self.session.run(self.train_step,
                feed_dict={
                    self.state : last_state_batch,
                    self.target_q : target_batch
                })
        self.last_action = action
        self.last_state = new_state
        return action

class ActionReplay:
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