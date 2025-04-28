class MyClass:
    def __call__(self, x, y):
        return x + y

    # def __repr__(self):  # 描述当前对象状态
    #     return f'MyClass(x={self.x}, y={self.y})'


obj = MyClass()
print(obj(1, 2))
