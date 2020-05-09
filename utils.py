from params import h_w_coef


def get_size_by_y(full_heigh: int, min_len: int, max_len: int, y: int) -> (int, int):
    """

    :param full_heigh:
    :param min_len:
    :param max_len:
    :param y:
    :return: (H, W) for current y coord
    """
    H = int(min_len + (float(max_len) - min_len) / (full_heigh - max_len) * y)
    W = int(h_w_coef * H)
    return H, W
