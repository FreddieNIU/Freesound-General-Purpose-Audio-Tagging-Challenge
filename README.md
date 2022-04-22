
# Freesound General-Purpose Audio Tagging Challenge

## https://www.kaggle.com/c/freesound-audio-tagging


First put the two folders of dataset under the freesound-audio-tagging\ folder, and install all required packages using requirement.txt. 

Data Augmentation.ipynb is the notebook that augment the training data from which you can generate  audio_train_augmented\, augmented_mfcc24_npy_files\, and augmented_stft_npy_files\ 

lstm.py is the script training baseline model.

lstm_mfcc.py and lstm_mfcc.ipynb are codes for LSTM models

resize_cnn.py is the script for fine-tuned VGG16 model, and resize_cnn.ipynb is the notebook for fine-tuned VGG16 model and good initialized head fine-tuned VGG16 models (using the resizing method to unitize image size). 

padding_cnn.py is the script for fine-tuned VGG16 model, and padding_cnn.ipynb is the notebook for fine-tuned VGG16 model and good initialized head fine-tuned VGG16 models (using the padding method to unitize image size). 

CNN model is part of resize_cnn.ipynb and padding_cnn.ipynb where you just use the defined model CNN_02 instead of CNN. 

All the trained models are saved in the ckp\ folder. 

Evaluate_lstm.ipynb is the notebook used to evaluate LSTM models. To evaluate CNN models and VGG16 models, you can find the code at the end of resize_cnn.ipynb and padding_cnn.ipynb

labelencoder.pkl is the saved labelencoder to encode the audio labels. 
}
