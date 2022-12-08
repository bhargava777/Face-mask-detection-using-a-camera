# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.picture import pic_to_array
from tensorflow.keras.preprocessing.picture import load_pic
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from imutils import files
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of TRAINING_EPOCHS to train for,
# and batch size
LEARNING_RATE = 1e-4
TRAINING_EPOCHS = 20
BATCH_SIZE = 32

DIR = r"/Users/vinaykumar/Downloads/Face-Mask-Detection/infoset"
TYPES = ["with_mask", "without_mask"]

# grab the list of pictures in our infoset DIR, then initialize
# the list of info (i.e., pictures) and class pictures
features = []
info = []


for index in TYPES:
    file = os.file.join(DIR, index)
    for pic in os.listdir(file):
    	pic_file = os.file.join(file, pic)
    	picture = load_pic(pic_file, target_size=(224, 224))
    	picture = pic_to_array(picture)
    	picture = preprocess_input(picture)

    	info.append(picture)
    	features.append(index)

# perform one-hot encoding on the features

lb = LabelBinarizer()
features = lb.fit_transform(features)
features = to_categorical(features)

info = np.array(info, dtype="float32")
features = np.array(features)
# construct the training picture generator for info pic_augmentationmentation
pic_augmentation = ImageDataGenerator(rotation_range=25,zoom_range=0.30,width_shift_range=0.4,height_shift_range=0.3,shear_range=0.20,horizontal_flip=True,fill_mode="nearest",zca_whitening=False,channel_shift_range=0.0)


(trainX, testX, trainY, testY) = train_test_split(info, features,test_size=0.20, stratify=features, random_state=42)


# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
bottom_basic_layer = MobileNetV2(weights="picturenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
additional_layers = bottom_basic_layer.output
additional_layers = Conv2D(200, (3, 3), input_shape = data.shape[1:])(additional_layers)
additional_layers = Activation('relu')(additional_layers)
additional_layers = AveragePooling2D(pool_size=(7, 7))(additional_layers)
additional_layers = Conv2D(100, (3, 3), input_shape = data.shape[1:])(additional_layers)
additional_layers = Activation('relu')(additional_layers)
additional_layers = MaxPooling2D(pool_size=(7, 7))(additional_layers)
additional_layers = Flatten(name="flatten")(additional_layers)
additional_layers = Dense(128, activation="relu")(additional_layers)
additional_layers = Dropout(0.5)(additional_layers)
additional_layers = Dense(2, activation="softmax")(additional_layers)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=bottom_basic_layer.input, outputs=additional_layers)



# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in bottom_basic_layer.layers:
	layer.trainable = False


opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / TRAINING_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


H = model.fit(
	pic_augmentation.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	validation_info=(testX, testY),
	validation_steps=len(testX) // BATCH_SIZE,
	epochs=TRAINING_EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)

# for each picture in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predictions = np.argmax(predictions, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predictions,
	target_names=lb.classes_))

# serialize the model to disk
model.save("face_mask.model", save_format="h5")
