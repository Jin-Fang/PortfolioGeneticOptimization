class Asset:
    """

    """
    def __init__(self, length):
        """

        :param length:
        """
        self.length = length/3
        self.cu_weight = []
        self.se_weight = []

    def __str__(self):
        """

        :return:
        """
        return "asset length: " + str(self.length) + "\n" + str(self.cu_weight) + "\n" + str(self.se_weight)

if __name__ == "__main__":
    a = Asset(36)
    print(a)