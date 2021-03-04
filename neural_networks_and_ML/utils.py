def create_batches(x, y, size=100):
    start, end = 0, size
    while end < x.shape[0]:
        yield x[start:end, :], y[start:end]
        start += size
        end += size
