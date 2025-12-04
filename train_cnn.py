import tensorflow as tf
import pathlib
import numpy as np
import pandas as pd
import os.path
import os
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomTranslation, RandomZoom, RandomBrightness
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, VGG19, VGG16, EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB3, DenseNet121, ResNet50V2
from tensorflow.keras.optimizers import Adam, AdamW, Nadam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.regularizers import l2

train_and_val_dir = pathlib.Path("train_and_val_datasets")
# test_dir = pathlib.Path("test_dataset")

batch_size = 16
image_height = 224
image_width = 224
epochs = 20
learning_rate = 1e-3
AUTOTUNE = tf.data.AUTOTUNE

# ================================
# Function for creating dataframe out of the directory
# ================================
def build_dataframe(directory):
    filepaths = []
    labels = []

    for class_dir in directory.iterdir():
        if class_dir.is_dir():
            # directories names in train_and_val_dir are the class names
            label = class_dir.name
            for file in class_dir.glob("*.*"):
                # filepaths are the paths to the file eg. "train_and_val_datasets/no_findings/00011145_000.png"
                # labels are class names eg. "no_findings"
                filepaths.append(str(file))
                labels.append(label)
    
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    # converts the label column into a pandas categorical type 
    # eg. "0" for "covid19" class, "1" for "pneumonia" class and "3" for "no_findings" class
    df["label_index"] = df["label"].astype('category').cat.codes

    # converts each numerical label into a one-hot vector 
    # eg. "[1, 0, 0]" for "covid19" class, "[0, 1, 0]" for "pneumonia" class and "[0, 0, 1]" for "no_findings" class
    df["label_onehot"] = df["label_index"].apply(
        lambda x: tf.keras.utils.to_categorical(x, num_classes=len(df["label"].unique())))
    
    return df, df["label"].unique()

df, classes = build_dataframe(train_and_val_dir)
print("Number of loaded images: ", len(df))
print("Classes: ", classes)
print("Number of loaded images per class: ", df["label"].value_counts())

# ================================
# Function for loading images
# ================================
def load_image(path, label):
    image = tf.io.read_file(path)

    try: # JPEG decoding
        image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    except: # If JPEG decoding fails → try PNG decoding
        try:
            image = tf.image.decode_png(image, channels=3)
        except: # If both fail → return a black image
            image = tf.zeros([*(image_height, image_width), 3], dtype=tf.uint8)

    # resize the image to image_height x image_width
    image = tf.image.resize(image, [image_height, image_width], method="nearest")
    return image, label

# ================================
# Function for creating dataset from dataframe
# ================================  
def df_to_dataset(df, shuffle=True):
    paths = df["filepath"].values
    labels = np.stack(df["label_onehot"].values)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(len(df))

    # loading the images to dataset and resizing them
    ds = ds.map(lambda x, y: load_image(x, y), num_parallel_calls=AUTOTUNE)

    # buffered prefetching to load images from disk without having I/O become blocking
    # When you train a model your GPU/CPU is training the model, and your CPU is loading + preparing the next batch of data
    # If these two don’t work in parallel, the GPU ends up waiting for data — which slows training down.
    # prefetch() tells TensorFlow to prepare the next batch in the background while the model is training on the current batch
    # tf.data.AUTOTUNE means: TensorFlow, you decide how many batches to prepare ahead of time.
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# ================================
# Function for data augmentation
# ================================  

data_augmentation = tf.keras.Sequential([
    # RandomRotation(0.1),
    RandomRotation(10/360),
    # RandomTranslation(height_factor=0.1, width_factor=0.1), # more aggressive
    # RandomZoom(height_factor=0.1, width_factor=0.1), # more aggressive
    # RandomBrightness(factor=0.2, value_range=(0, 1)),
    RandomFlip(mode="horizontal")
])

# data_augmentation = tf.keras.Sequential([
#     RandomFlip("horizontal"),
#     RandomRotation(0.1),
#     RandomZoom(0.1),
#     RandomTranslation(height_factor=0.1, width_factor=0.1) # more aggressive
# ])

# applying data augmentation on same image 9 times and ploting the outcomes
# ds = df_to_dataset(df)
# for image, _ in ds.take(1):
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0] / 255)
#     plt.axis('off')
# plt.show()

# ================================
# Function for creating the CNN model
# ================================  
def build_model(num_classes):   
    # data preprocessing
    inputs = Input(shape=(image_height, image_width, 3))
    x = data_augmentation(inputs)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # preprocessing becomes an actual Keras layer in the model graph
    x = Lambda(preprocess_input, name="preprocessing_layer")(x)

    # pretrained cnn
    base_model = MobileNetV2(input_tensor=x,
                            include_top=False,
                            weights='imagenet')
    base_model.trainable = False
    # base_model.summary()

    # new classifier
    x=base_model.output
    # x = base_model(inputs, training=False)
    # x = AveragePooling2D(pool_size=(4, 4))(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.4)(x)
    # x = Dense(1024, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
    # x = Dense(256, activation="relu", kernel_initializer='he_uniform',kernel_regularizer=l2(1e-4))(x)
    # x = Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    # model.summary()

    metrics = [
        CategoricalAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
    # optimize = SGD(learning_rate=0.0001, decay=0.9 / 5, momentum=0.9, nesterov=True)
    # optimizer_conv = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9)

    model.compile(
                # optimizer=SGD(learning_rate=learning_rate,
                #                 momentum=0.9),
                #   optimizer=Adam(learning_rate=learning_rate),
                  optimizer=Nadam(learning_rate=learning_rate),
                #   optimizer=AdamW(learning_rate=learning_rate),
                loss=CategoricalCrossentropy(),
                metrics=metrics)

    # initial loss and accuracy
    # loss0, accuracy0 = model.evaluate(validation_dataset)
    # print("initial loss: {:.2f}".format(loss0))
    # print("initial accuracy: {:.2f}".format(accuracy0))

    return model, base_model

# ================================
# Callbacks
# ================================  
checkpointer = ModelCheckpoint(
    filepath="models/checkpoints/mobilenetv2_best.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_logger = CSVLogger(os.path.join('models','logs',f"MobileNetV2_1-training-{timestamp}.log"))

early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',     # metric to watch
    factor=0.5,             # reduce LR by half
    patience=2,             # wait 3 epochs before reducing
    min_lr=1e-7,            # final lowest allowed LR
    verbose=1)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=1e-7)
# Bozinovic
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
#                                             mode='min'
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.1)
# CoronaNidaan
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
#                                             patience=2, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)
# wang factor  0.7 patience  5

callbacks = [
    checkpointer,
    early_stopper,
    csv_logger,
    learning_rate_reduction
]

def save_fold_plots(history, fold):
    out_dir = "acc+loss"
    os.makedirs(out_dir, exist_ok=True)

    # ------- Accuracy plot -------
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"Fold {fold} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{out_dir}/accuracy_fold_{fold}.png")
    plt.close()

    # ------- Loss plot -------
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"Fold {fold} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/loss_fold_{fold}.png")
    plt.close()

def run_5fold_cross_validation(df, class_names):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    val_accuracy_list = []
    recall_list = []
    precision_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n==================== FOLD {fold} ====================\n")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = df_to_dataset(train_df, shuffle=True)
        val_ds = df_to_dataset(val_df, shuffle=False)

        print("Train samples:", len(train_idx))
        print("Val samples:", len(val_idx))

        model, base_model = build_model(num_classes=len(class_names))

        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=callbacks)
        
        # # Fine-tune from this layer onwards, so 20 layers are fine tuned
        # for layer in base_model.layers[-5:]:
        #     layer.trainable = True

        # for layer in base_model.layers:
        #     if isinstance(layer, tf.keras.layers.BatchNormalization):
        #         layer.trainable = False

        # fine_tuning_learning_rate = 5e-5
        # model.compile(loss=CategoricalCrossentropy(),
        #             #   optimizer = Adam(learning_rate=fine_tuning_learning_rate),
        #             #   optimizer=AdamW(learning_rate=learning_rate),
        #             #   optimizer = Nadam(learning_rate=fine_tuning_learning_rate),
        #                 optimizer=SGD(learning_rate=fine_tuning_learning_rate,
        #                             momentum=0.9),
        #             metrics=[CategoricalAccuracy(name='accuracy')])

        # fine_tune_epochs = 10
        # total_epochs =  epochs + fine_tune_epochs

        # history_fine = model.fit(train_ds,
        #                         epochs=total_epochs,
        #                         initial_epoch=len(history.epoch),
        #                         validation_data=val_ds)

        print(f"✔ Finished Fold {fold+1}\n")

        fold_val_acc = history.history["val_accuracy"][-1]
        val_accuracy_list.append(fold_val_acc)

        fold_recall = history.history["recall"][-1]
        recall_list.append(fold_recall)

        fold_precision = history.history["precision"][-1]
        precision_list.append(fold_precision)

        print(f"FINAL VAL ACCURACY = {fold_val_acc:.4f}")
        print(f"FINAL RECALL = {fold_recall:.4f}")
        print(f"FINAL PRECISION = {fold_precision:.4f}")

        # save acc and loss plots
        save_fold_plots(history, fold)
    
    print("\n================== CROSS-VAL RESULTS ==================\n")

    print("Fold accuracies:", val_accuracy_list)
    print("Mean accuracy:", np.mean(val_accuracy_list))
    print("Std deviation:", np.std(val_accuracy_list))

    print("Fold recalls:", recall_list)
    print("Mean recall:", np.mean(recall_list))
    print("Std deviation:", np.std(recall_list))

    print("Fold precisions:", precision_list)
    print("Mean precision:", np.mean(precision_list))
    print("Std deviation:", np.std(precision_list))

run_5fold_cross_validation(df=df, class_names=classes)

# df = df.sample(frac=1, random_state=123).reset_index(drop=True)
# split_index = int(len(df) * 0.8)
# train_df = df.iloc[:split_index]
# val_df   = df.iloc[split_index:]
# print("Train images:", len(train_df))
# print("Validation images:", len(val_df))

# train_dataset = df_to_dataset(train_df, shuffle=True)
# validation_dataset   = df_to_dataset(val_df, shuffle=False)

# model, base_model = build_model(len(classes))

# history = model.fit(train_dataset,
#                     epochs=epochs,
#                     validation_data=validation_dataset,
#                     callbacks=callbacks)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# # Fine-tune from this layer onwards, so 20 layers are fine tuned
# for layer in base_model.layers[-5:]:
#     layer.trainable = True

# for layer in base_model.layers:
#     if isinstance(layer, tf.keras.layers.BatchNormalization):
#         layer.trainable = False

# fine_tuning_learning_rate = 5e-5
# model.compile(loss=CategoricalCrossentropy(),
#             #   optimizer = Adam(learning_rate=fine_tuning_learning_rate),
#             #   optimizer=AdamW(learning_rate=learning_rate),
#             #   optimizer = Nadam(learning_rate=fine_tuning_learning_rate),
#                 optimizer=SGD(learning_rate=fine_tuning_learning_rate,
#                             momentum=0.9),
#               metrics=[CategoricalAccuracy(name='accuracy')])

# fine_tune_epochs = 10
# total_epochs =  epochs + fine_tune_epochs

# history_fine = model.fit(train_dataset,
#                          epochs=total_epochs,
#                          initial_epoch=len(history.epoch),
#                          validation_data=validation_dataset)

# acc += history_fine.history['accuracy']
# val_acc += history_fine.history['val_accuracy']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.4, 1])
# plt.plot([epochs-1,epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([epochs-1,epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')

# plt.savefig("plot.png")
# plt.show()