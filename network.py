import numpy as np
import random
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # y er hvor mange neurons i dette laget, mens x er hvor mange i forrige

 

   
    def feedforward(self, a):
        #Går lag for lag og gjøre activation function på hver av inputsa hvor a da blir en vector med alle activasionsa
        
        for b, w in zip(self.biases, self.weights):
            #weights ganger activation pluss bias, sigmod blir gjort på hver og en neuron på grunn av numpy 
            a = sigmoid(np.dot(w,a)+ b)
            
        return a
    
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """
        Args:
            training_data : treningsdata
            epochs : hvor mange generasjoner du skal ha
            mini_batch_size : hvor store batchene vi skal kjøre gradient decent på er
            eta : learning rate
            test_data: sammenlikningsdata . Defaults to None.
        """
     
        
        n = len(training_data)

        for j in range(epochs):
            #shuffler den i starten så det blir tilfeldig pluket batcher
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            #hvor vær batch kjør gradient decent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #Bruk test dataen til å sjekke hvor mange den klassifiserer riktig
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)} ")
            else:
                print(f"Epoch {j} complete")
            
    def update_mini_batch(self, mini_batch, eta):
        #lager en tom vector med samme form som biases og weigths med bare masse nuller
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #x er inputs, og y er ønskelig output 
        for x, y in mini_batch:
           
            #bruker backpropagation til å finne ut hvilken retning vi skal dytte wheightsa og biasene
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            #lagrer en "samlet" retning fra all dataen i batchen og får da en retning basert på alle treningsdata
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        #oppdaterer wheightsa og biasene med det vi har funnet ut
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
            
        self.biases = [b-(eta/len(mini_batch))*nb 
                        for b, nb in zip(self.biases, nabla_b)]
          
          
    #alt under denne skal jeg lære mer om lengre i boken så har bare nabbet den, men skal komme tilbake til den når jeg lest meg opp på dette
    def backprop(self, x, y):
      
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        return (output_activations-y)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

