

class Test:
    def __init__(self):
        self.a = 1

    def change_a(self, w):
        w = 2

    def fun(self):
        print(self.a)
        self.change_a(self.a)
        print(self.a)


if __name__ == '__main__':
    test = Test()
    test.fun()