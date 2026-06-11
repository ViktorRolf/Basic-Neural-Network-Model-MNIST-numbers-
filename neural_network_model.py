import numpy as np


class neural_network:
    def __init__(self):
        pass

    def init_network(self, number_of_nodes_per_layer: list[int]):
        """
        Initialize the neural network architecture.

        Args:
            number_of_nodes_per_layer: list[int]: List of integers, representing number of nodes in input layer, hidden layers and output layer

        Returns:
            None
        """
        
        #layer/node constants
        self.__number_of_nodes_per_layer__ = number_of_nodes_per_layer
        self.__number_of_layers__ = len(number_of_nodes_per_layer)

        #weights/biases constants
        self.__weights__ = []
        self.__biases__ = []
        
        for i in range(self.__number_of_layers__-1):
            weights_layer_i_to_iplus1 = 0.01 * np.random.randn(self.__number_of_nodes_per_layer__[i+1],self.__number_of_nodes_per_layer__[i])
            biases_layer_i            = 0.01 * np.random.randn(self.__number_of_nodes_per_layer__[i+1])

            self.__weights__.append(weights_layer_i_to_iplus1)
            self.__biases__.append(biases_layer_i)
        
        #self.test = self.__weights__[0] @ np.array([1,2,3,4,5,6,7,8,9,10]) + self.__biases__[0]

    def __sigmoid__(self, x):
        return 1/(1+np.exp(-x))
    
    def __sigmoid_prime__(self, x):
        s = self.__sigmoid__(x)
        return s*(1-s)
    
    def __forward_propagation__(self, image: np.array, print_output: bool =False):
        '''
        Run the model (forward) for one image to make a prediction. Saves node activations per layer 
        and z value per layer

        Args:
            image: np.array: Array of real numbers in [0,1] representing input image
            print_output: bool: Should the result be printed

        Returns:
            prediction: int: Which label is most likely

        '''
        node_activations = []
        node_activations.append(image)

        z_to_layer = []

        for i in range(self.__number_of_layers__-1):
            z_to_layer.append(self.__weights__[i] @ node_activations[i] + self.__biases__[i])
            node_activations.append(self.__sigmoid__(z_to_layer[i]))

        self.__node_activations__ = node_activations
        self.__z_to_layer__ = z_to_layer

        prediction = np.argmax(node_activations[-1])
        if print_output:
            print( f"Predicted: {prediction} with confidence vector: {node_activations[-1]}" )
        return prediction

    def __cost_for_image__(self, image: np.array, correct_label: int, print_output: bool):
        '''
        Calculate cost for one image.

        Args:
            image: np.array: Array of real numbers in [0,1] representing input image
            correct_label: int: The correct label for the image
            print_output: bool: Should the result be printed

        Returns: 
            cost: real: The calculated cost for given image
        '''
        
        self.__forward_propagation__(image, print_output=False)
        
        correct_activations = np.zeros(10)
        correct_activations[correct_label] = 1

        cost = np.sum((self.__node_activations__[-1]-correct_activations)**2)
        if print_output:
            print(f"Costs for image is {cost}!")
        return cost
    
    def __delCost_for_image__(self, image: np.array, correct_label: int):
        '''
        Calculate delC/delw and delC/delb for all b and w

        Args:
            image: np.array: Array of real numbers in [0,1] representing input image
            correct_label: int: The correct label for the image

        Returns: 
            delC_delb: list[np.array]: list of arrays of all delC/delb
            delC_delW: list[np.array]: list of matrixes of all delC/delw
        '''
        self.__forward_propagation__(image,False)

        #Prep delta, list of empty arrays and initialize final delta
        delta_at_layer = [np.zeros(self.__number_of_nodes_per_layer__[i]) 
                          for i in range(1,self.__number_of_layers__)]  
        correct_activations = np.zeros(self.__number_of_nodes_per_layer__[-1])
        correct_activations[correct_label] = 1
        delta_at_layer[-1] = np.multiply(2*(self.__node_activations__[-1]-correct_activations),
                                         self.__sigmoid_prime__(self.__z_to_layer__[-1]))
        
        # Calculate all delta through recursions
        for i in range(2,self.__number_of_layers__):
            layer = self.__number_of_layers__ - 1 - i
            delta_at_layer[layer] = np.multiply(self.__weights__[layer+1].transpose() @ delta_at_layer[layer+1],
                                                self.__sigmoid_prime__(self.__z_to_layer__[layer]))
        
        # Assemble by /delb and /delW arrays
        delC_delb = [arr.copy() for arr in delta_at_layer]
        delC_delW = [delta_at_layer[i][:, None] * self.__node_activations__[i][None, :] 
                     for i in range(self.__number_of_layers__-1)]
        return delC_delW, delC_delb

    def __gradient_descent__(self, data_train: list):
        '''
        Performs gradient descent for input data

        Args:
            data_train: list[tuple]: List of tuples, each tuple is (image, label) as (np.array, int).

        Returns:
            None
        '''

        W_accumulation = [np.zeros_like(self.__weights__[i]) for i in range(self.__number_of_layers__-1)]
        b_accumulation = [np.zeros_like(self.__biases__[i]) for i in range(self.__number_of_layers__-1)]
        
        for data in data_train:
            delC_delW, delC_delb = self.__delCost_for_image__(data[0], data[1])
            for i in range(self.__number_of_layers__-1):
                W_accumulation[i] += delC_delW[i]
                b_accumulation[i] += delC_delb[i]
        
        num_images = len(data_train)
        step_size = 0.1
        for i in range(self.__number_of_layers__-1):
            self.__weights__[i] -= step_size*W_accumulation[i]/num_images
            self.__biases__[i] -= step_size*b_accumulation[i]/num_images


    def train(self, epochs: int, images_per_epoch: int, data_train: list):
        for j in range(epochs):
            selected_ints = selected_ints = np.random.choice(len(data_train),size=images_per_epoch, replace=False)
            selected_data = [data_train[i] for i in selected_ints]
            self.__gradient_descent__(selected_data)
            if j%10 ==0:
                print(f"epoch {j} done")

    def check(self, data_test: list):
        correct_predictions = 0
        for data in data_test:
            prediction = self.__forward_propagation__(data[0])
            if prediction == data[1]:
                correct_predictions += 1
        part_correct = correct_predictions/len(data_test)
        return part_correct