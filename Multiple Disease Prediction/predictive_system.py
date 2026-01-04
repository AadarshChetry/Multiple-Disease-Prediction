import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/V2/saved_models/diabetes_model.sav", 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

#change the input_data to numpy_array
inp_as_np_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
inp_data_reshaped = inp_as_np_array.reshape(1,-1)

prediction = loaded_model.predict(inp_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("The person is not diabetic")

else:
  print("The person is diabetic")