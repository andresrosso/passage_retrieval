class ObjectFactory():
    @staticmethod
    def loadDataSet(targetclass):
        return globals()[targetclass]()
