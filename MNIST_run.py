import numpy as np
import matplotlib.pyplot as plt
from neural_network_model import neural_network
from MNIST_fileHandler import pkl_read,pkl_write



### Read/write ###
#pkl_write()
flat_data_train, flat_data_test = pkl_read()
### 


### Initialize Model ###
model = neural_network()
node_structure = [28*28,50,20,10]
#node_structure = [10, 5, 2]
model.init_network(node_structure)

### 




### TESTS ###
which_example = 4
flat_ex = flat_data_train[which_example][0]
image_ex = flat_ex.reshape(28,28)
label_ex = flat_data_train[which_example][1]


#model.__cost_for_image__(flat_ex,label_ex,print_output=True)


model.__cost_for_image__(flat_ex, label_ex, True)
old = [model.__weights__[i].copy() for i in range(3)]
model.train(epochs=1000, images_per_epoch=100, data_train=flat_data_train)
model.__cost_for_image__(flat_ex, label_ex, True)
new = [model.__weights__[i].copy() for i in range(3)]
model.__forward_propagation__(flat_ex,print_output=True)
print(model.check(flat_data_test))

diff = sum([np.linalg.norm(old[i]-new[i]) for i in range(3)])
print(diff)

plt.imshow(image_ex, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

# np.set_printoptions(linewidth=200)
# print(label_ex)
# print(np.round(image_ex,1))

# i=0
# print(f"{np.round(model.__weights__[i],3)}")
# print(f"{np.round(model.__biases__[i],3)}")
# print(f"{np.round(model.test,3)}")