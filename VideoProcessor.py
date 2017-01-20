class VideoProcessor:

    sum = 0
    linija = None

    def __init__(self):
        pass

    def SetLiniju(self,linija):
        self.linija = linija

    def IsLinijaSet(self):
        if linija == None:
            return False
        else:
            return True

    def Process(self, brojeviSlike):
        pass