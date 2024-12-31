from MarineEnv3 import MarineEnv

if __name__ == '__main__':
    env = MarineEnv()
    env.reset()
    for _ in range(50):
        env.render()