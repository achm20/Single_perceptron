#import Main
import random
import numpy as np
import Main

# choose 4 random training cases of 1, 4 random training cases of 0
first_half = range(1, 7)
second_half = range(7, 13)
full_range = range(1, 13)
train_index_first = random.sample(first_half, 4)
train_index_second = random.sample(second_half, 4)
train_index = train_index_first + train_index_second
print(train_index)

# specify learning rate, alpha
alpha = 0.2

#initialise weights and bias
weights = []
for i in range (0, 25):
    n = random.random()
    weights.append(n)

weights = np.transpose(weights)
print('initial weights = ' + str(weights))
bias = random.uniform(-1, 1)
print('initial bias = ' + str(bias))

#loop training for all chosen training cases
for x in train_index:
    index = x
    index_imp = index - 1
    sign_sum = np.sign(sum(np.transpose(Main.Data[index_imp])*weights) + bias)
    weights = weights + alpha * (Main.Desired_output[index_imp] - sign_sum) \
              * Main.Data[index_imp]
    bias = bias + alpha * (Main.Desired_output[index_imp] - sign_sum)

print('trained weights = ' + str(weights))
print('trained bias = ' + str(bias))

#predict on the cases not chosen
for case in full_range:
    if case not in train_index:
        test_index = case
        case_imp = case - 1
        ytest = np.sign(sum(np.transpose(Main.Data[case_imp])*weights) + bias)
        print('test case ' + str(case))
        print(ytest)
        print(Main.Desired_output[case_imp])
