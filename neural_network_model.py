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
    
    def predict_image(self, image: np.array, print_output: bool =False):
        '''
        Run the model (forward) for one image to make a prediction

        Args:
            image: np.array: Array of real numbers in [0,1] representing input image
            print_output: bool: Should the result be printed

        Returns: prediction, nodes_layer_i:
            prediction: int : Predicted number
            nodes_layer_i: np.array: Confidence in each number

        '''
        nodes_layer_i = image
        for i in range(self.__number_of_layers__-1):
            nodes_layer_iplus1 = self.__sigmoid__(self.__weights__[i] @ nodes_layer_i + self.__biases__[i])
            nodes_layer_i = nodes_layer_iplus1

        prediction = np.argmax(nodes_layer_i)
        if print_output:
            print( f"Predicted: {prediction} with confidence vector: {nodes_layer_i}" )
        return prediction, nodes_layer_i