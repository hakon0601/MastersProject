
input = [16, 32]
filters = [5, 5, 5]
channels = [32, 64, 64]
strides = [1, 1, 1]
padding = [2, 2, 2]


expected_result = 7*7*64

def calculate_size_from_layer(inn, filter, stride, padding):
    a = inn - filter
    a += 2*padding
    a /= stride
    a += 1
    return a


for i in range(len(filters)):
    width1, height1 = calculate_size_from_layer(input[0], filters[i], strides[i], padding[i]), calculate_size_from_layer(input[1], filters[i], strides[i], padding[i])
    print(width1, height1)
    width1_after_max_pool, height1_after_max_pool = calculate_size_from_layer(width1, 2, 2, 0), calculate_size_from_layer(height1, 2, 2, 0)
    print(width1_after_max_pool, height1_after_max_pool)
    input = [width1_after_max_pool, height1_after_max_pool]

