from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.properties import ListProperty, StringProperty, NumericProperty
from kivy.uix.button import Button
from kivy.clock import Clock

import time
import numpy as np
import ai

dqn = ai.Dqn(0.9,[50,30,20],0.01, 0.5)

class BoardScreen(Screen):
    board = ListProperty(9*[''])
    turn = StringProperty('X')
    status = StringProperty('running')
    nb_matches = NumericProperty(0)
    x_wins = NumericProperty(0)
    o_wins = NumericProperty(0)
    random_play_event = None

    def toMatrix(self, board):
        return [
            board[:3],
            board[3:6],
            board[6:]
            ]
    def toList(self, board):
        return [element for row in board for element in row]

    def updateStatus(self):
        if self.status == 'running':
            lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
            board = np.array(self.board)
            for line in lines:
                j1, j2, j3 = board[line]
                if j1 == j2 == j3 and j1 != '':
                    self.status = j1
                    if j1 == 'X':
                        self.x_wins += 1
                    else:
                        self.o_wins += 1
                    return
            if '' not in self.board:
                self.status = 'tie'

    def nextTurn(self):
        self.turn = 'X' if self.turn == 'O' else 'O' 
        
    def random(self):
        if self.status == 'running':
            state = self.board
            available_actions = [i for i in range(len(state)) if not state[i]]
            self.play(np.random.choice(available_actions))
        else:
            self.restart()

    def restart(self):
        self.nb_matches += 1
        reward = {'X':-1, 'tie':0, 'O':1, 'running':0}[self.status]
        dqn.update(self.board, reward)
        dqn.last_action = None
        dqn.last_state = None
        self.board = 9*['']
        self.status = 'running'
        self.turn = 'X'
    
    def toggle_random(self):
        if self.random_play_event == None:
            self.random_play_event = Clock.schedule_interval(
                lambda dt : self.random()
            , 0.01)
        else:
            self.random_play_event.cancel()
            self.random_play_event = None

    def playAI(self):
        if self.status == 'running':
            choice = dqn.choose_action(self.board)
            if choice != -1:
                self.play(choice)
        else:
            self.restart()

    def play(self, position):
        if self.status == 'running' and not self.board[position]:
            self.board[position] = self.turn
            self.nextTurn()
        result = self.updateStatus()
        if self.status == 'running' and self.turn == 'O':
            choice = dqn.update(self.board, 0)# 0, 0.01, 0.1 ... ?
            if choice != -1:
                self.play(choice)

class GameApp(App):
    def build(self):
        return ScreenManager()


if __name__ == '__main__':
    GameApp().run()