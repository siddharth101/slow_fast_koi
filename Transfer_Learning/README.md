The file gravityspy_model.py was used to create a CNN model with the same architecture as the one used to train current (fast scattering + low frequency blips) model. This model is then saved to the file gravityspy_model.h5.

In the script classify_noise_transfer_learn.py I am using transfer learning and loading the model stored in gravityspy_model.h5. It is important to notice that this model has not been trained earlier (the original model is created in a complicated way by adding a sequential layer to output layer), so this is not exactly transfer learning but the syntax is correct.

The loss and accuracy is then saved in history_transfer.pkl and the final model is saved in transfer_fast_slow_koi.h5
