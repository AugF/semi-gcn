# rnn:   x=1*f   a=1*h  y=1*c
# a = x * w_cell_input + a * w_cell_state + b_cel
# y = a * w_output
# 则  w_cell_input: f*h    w_cell_state h*h   b_cell 1*h
# w_output h*c

import numpy as np

# h=2, f=1, c=1
X = [1, 2]
state = [0.0, 0.0]

w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])

b_cell = np.asarray([0.1, -0.1])

w_output = np.asarray([[1.0], [2.0]])

b_output = 0.1

for i in range(len(X)):
    # 计算循环体重的全脸解神经网络
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 根据当前状态计算最终输出
    final_output = np.dot(state, w_output) + b_output

    print("before activation: ", before_activation)
    print("state", state)
    print("output", final_output)
