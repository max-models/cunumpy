import cunumpy as xp


def test_xp_array():

    arr = xp.array([1, 2])
    arr *= 2

    print(f"{arr = } {type(arr) = }")

if __name__ == "__main__":
    test_xp_array()
