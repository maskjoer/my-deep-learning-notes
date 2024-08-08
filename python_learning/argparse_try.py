import argparse

parser = argparse.ArgumentParser(description='description')
# parser.add_argument('-gf', '--girlfriend', choices=['jingjing', 'lihuan'])
# parser.add_argument('food')
# parser.add_argument('--house', type=int, default=0)
# parser.add_argument('--model name', '-m', type=str, required=True, choices=['model_A', 'model_B'])
parser.add_argument('--ap', '-a', action='store_true')
args = parser.parse_args()
print(args)
