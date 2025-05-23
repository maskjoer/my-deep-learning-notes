class MyClass:
    def __call__(self, x, y):
        return x + y

    # def __repr__(self):  # 描述当前对象状态
    #     return f'MyClass(x={self.x}, y={self.y})'


def check_index(key):
    if not isinstance(key, int): raise TypeError
    if key < 0:raise IndexError
class ArithmeticSequence:

    def __init__(self, start = 0, step = 1):
        self.start = start
        self.step = step
        self.changed ={}
    def __getitem__(self,key):
        try :return self.changed[key]
        except KeyError:
            return self.start+key*self.step
    def __setitem__(self, key, value):
        check_index(key)
        self.changed[key]= value
        
s = ArithmeticSequence(1,2)
print(s[32])


