from cunumpy.arrays import xp


def test_xp_array():

    arr = xp.array([1, 2])

    print(type(arr))
    print(xp.__name__)


if __name__ == "__main__":
    test_xp_array()
