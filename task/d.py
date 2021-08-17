class GazeV1:
    def __init__(self):
        pass

    def run(self):
        pass

class GazeV2:
    def __init__(self):
        pass

    def run(self):
        pass

class CallGaze:
    def __init__(self):
        pass

    def load_gaze(self, what_gaze: str):
        if what_gaze == 'v1':
            return GazeV1()
        else:
            return GazeV2()

if __name__ == '__main__':
    call = CallGaze()

    call.load("v1").run()

