import numpy as np

all_data = [
    [0,0,0,[0]],
    [0,0,1,[1]],
    [0,1,0,[0]],
    [0,1,1,[0]],
    [1,0,0,[0]],
    [1,0,1,[0]],
    [1,1,0,[1]],
    [1,1,1,[1]],
]
print(all_data)

np.random.shuffle(all_data)

print(all_data)


nn_input = list()
nn_class = list()

for i in range(len(all_data)):
    nn_input.append(all_data[i][0:3])
    nn_class.append(all_data[i][3])

print(nn_input)
print(nn_class)

np.save("data/nn_m_input",nn_input)
np.save("data/nn_m_class",nn_class)
# np.save("data/nn_n_input",nn_input)
# np.save("data/nn_n_class",nn_class)