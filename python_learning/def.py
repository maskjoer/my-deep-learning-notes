class People:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print('Hello {}!'.format(self.name))


person = People('John', 23)
person.greet()