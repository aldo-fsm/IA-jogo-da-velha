from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager

class BoardScreen(Screen):
    pass


class GameApp(App):
    def build(self):
        return ScreenManager()


if __name__ == '__main__':
    GameApp().run()