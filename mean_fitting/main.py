import numpy as np


def main():
    dist = []
    for i in range(1000):
        theta1, theta2 = np.random.uniform(
            low=np.array([-1., -1.]),
            high=np.array([1., 1.]),
            size=(2, 2)
        )
        dist.append(np.linalg.norm(theta1 - theta2))

    print(np.mean(dist))


if __name__ == '__main__':
    main()
