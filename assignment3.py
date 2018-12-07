import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#input is an array of features and labels
		#store data for KNN
		self.X_train = X
		self.y_train = y
		None

	def predict(self, X):
		#Return array of predictions where there is one prediction for each set of features
		#containers for predictions and training set neighbors
		predictions_list = np.empty([len(X)])
		test_neighbors = []

		for return_index, observation in enumerate(X):
		
			sorted_dist_meas = self.get_distances(observation)

			for counter in range(self.k):
				population_member = sorted_dist_meas[counter][1]
				test_neighbors.append(self.y_train[population_member])

			predictions_list[return_index] = max(set(test_neighbors),key = test_neighbors.count)			

			del test_neighbors[:]

		return predictions_list

	def get_distances(self, observation):
		#get distances of observation from entire training set and return sorted list
		distance_measure = []

		for counter, training_observation in enumerate(self.X_train):
    		#find distance from training set observations using provided dist func
			observation_distance = self.distance(observation, training_observation)

			distance_measure.append([observation_distance, counter])	

		return sorted(distance_measure)

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		print(categorical_data)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		return None

class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#input is array of features and labels
		pop_size_multip = int(steps/len(X))

		for num_of_iterations in range(pop_size_multip):

			for label_index,training_observation in enumerate(X):
    			#get weighted pred
				weighted_pred = float(np.dot(training_observation, self.w) + self.b)
				error_rate = y[label_index] - weighted_pred
				
				#update beta and weights
				self.b = self.b + (error_rate * self.lr) 
				error_rate_lr = error_rate * self.lr
				
				for feature in training_observation:
					feature = feature * error_rate_lr

				self.w = np.add(self.w, training_observation)
		None

	def predict(self, X):
		#Return array of predictions where there is one prediction for each set of features
		prediction_list = np.empty([len(X)])

		for return_index, observation in enumerate(X):
    		#iterate over entire sample and make prediction
			predict_val = np.dot(observation, self.w) + self.b
			activation_val = int(np.sum(predict_val))

			prediction_list[return_index] = 0 if activation_val > 1 else 1

		return prediction_list

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None	

class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None	