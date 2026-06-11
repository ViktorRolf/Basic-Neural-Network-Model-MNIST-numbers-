import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from neural_network_model import neural_network
from MNIST_fileHandler import pkl_read,pkl_write
from PIL import Image

### Options ###
train_models = False
display_models = False
test_model_on_single = True
###


### Read/write ###
#pkl_write()
flat_data_train, flat_data_test = pkl_read()
### 


### Initialize Model ###
model = neural_network()
node_structure = [28*28,50,20,10]
model.init_network(node_structure)
### 


### Train (and save) models ###
epoch_select = [1000,2000,3000,4000,5000]
if train_models:
    for epoch in epoch_select:
        model.train(epochs=epoch, images_per_epoch=100, data_train=flat_data_train, step_size=1)
        model.save_to("model_trained_"+str(epoch)+"_epochs")
        model.check(flat_data_test,print_output=False)
### 


### Load (and display) models ###
if display_models:
    prediction_per_epoch = {}
    for epoch in epoch_select:
        model.load_from("model_trained_"+str(epoch)+"_epochs")
        prediction_per_epoch[epoch] = model.check(flat_data_test,print_output=False)

    df = pd.DataFrame([prediction_per_epoch])
    print(df)
###


### Test model on singular input
if test_model_on_single:
    #Options
    image_name = "test.jpg"
    num_of_epochs = 3000

    #load image
    image_load = Image.open("DRAW_files/"+image_name).convert("L")
    image_array = np.array(image_load)
    image_flat = image_array.flatten()

    #load model
    model.load_from("model_trained_"+str(num_of_epochs)+"_epochs")
    pred = model.__forward_propagation__(image_flat)
    
    #plot
    plt.imshow(image_array, cmap='gray', interpolation='nearest')
    plt.title(f"Model predicts: {pred}")
    plt.colorbar()
    plt.show()


### TESTS ###
# which_example = 4
# flat_ex = flat_data_train[which_example][0]
# image_ex = flat_ex.reshape(28,28)
# label_ex = flat_data_train[which_example][1]


# model.train(epochs=1000, images_per_epoch=100, data_train=flat_data_train)
# model.check(flat_data_test,True)
# model.save_to("testsave")


# model.load_from("testsave")
# model.check(flat_data_test,True)


# plt.imshow(image_ex, cmap='gray', interpolation='nearest')
# plt.colorbar()
# plt.show()


# np.set_printoptions(linewidth=200)
# print(label_ex)
# print(np.round(image_ex,1))


# i=0
# print(f"{np.round(model.__weights__[i],3)}")
# print(f"{np.round(model.__biases__[i],3)}")
# print(f"{np.round(model.test,3)}")