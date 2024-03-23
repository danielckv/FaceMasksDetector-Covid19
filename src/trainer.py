import os
import click
import keras
import numpy as np
import pandas as pd
import tensorflow
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from utils import classes

train_images = []
train_targets = []
train_labels = []


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli_app(ctx, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug
    ctx.obj['PWD'] = os.getcwd()


@cli_app.command
@click.pass_context
@click.option('--dataset_path', default='./datasets/train', help='The directory containing the training data.')
@click.option('--img_width', default=300, help='The width of the input images.')
@click.option('--img_height', default=300, help='The height of the input images.')
@click.option('--batch_size', default=32, help='The batch size for training.')
@click.option('--device', default='cpu', help='The device to use for training (cpu, cuda or mps).')
def train_model(ctx, dataset_path, img_width, img_height, batch_size, device):
    click.echo(f"Debug is {'on' if ctx.obj['DEBUG'] else 'off'}")
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator, validation_generator = None, None

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    if dataset_path.endswith(".csv") is False:
        train_data_dir, validation_data_dir = dataset_path + "/train", dataset_path + "/validation"

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')
    else:
        training_image_records = pd.read_csv(dataset_path)
        for index, row in training_image_records.iterrows():
            (idx, filename, width, height, class_name, xmin, ymin, xmax, ymax) = row

            train_image_fullpath = filename
            train_img = keras.preprocessing.image.load_img(train_image_fullpath, target_size=(img_height, img_width))
            train_img_arr = keras.preprocessing.image.img_to_array(train_img)

            xmin = round(xmin / img_width, 2)
            ymin = round(ymin / img_height, 2)
            xmax = round(xmax / img_width, 2)
            ymax = round(ymax / img_height, 2)

            train_images.append(train_img_arr)
            train_targets.append((xmin, ymin, xmax, ymax))

            if class_name == "mask_weared_incorrect":
                class_name = "without_mask"

            train_labels.append(classes.index(class_name))

    # Define the model architecture
    # create the common input layer
    input_shape = (img_height, img_width, 3)
    input_layer = tensorflow.keras.layers.Input(input_shape)

    # create the base layers
    base_layers = layers.Rescaling(1. / 255, name='bl_1')(input_layer)
    base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
    base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
    base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
    base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
    base_layers = layers.Flatten(name='bl_8')(base_layers)

    classifier_branch = layers.Dense(128, activation='relu', name='cl_1')(base_layers)
    classifier_branch = layers.Dense(2, name='cl_head')(classifier_branch)

    locator_branch = layers.Dense(128, activation='relu', name='bb_1')(base_layers)
    locator_branch = layers.Dense(64, activation='relu', name='bb_2')(locator_branch)
    locator_branch = layers.Dense(32, activation='relu', name='bb_3')(locator_branch)
    locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)

    model = tensorflow.keras.Model(input_layer,
                                   outputs=[classifier_branch, locator_branch])

    print("Compiling model...")
    # Compile the model
    model.compile(optimizer='adam',
                  loss={
                      "cl_head": tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      "bb_head": tensorflow.keras.losses.MSE
                  },
                  metrics=['accuracy', 'mae'])

    # Train the model
    epochs = 4

    print(f"Training model for {epochs} epochs on {device}...")

    if train_generator is None:
        # Convert the lists to numpy arrays
        np_train_images = np.array(train_images)
        np_train_targets = np.array(train_targets)
        np_train_labels = np.array(train_labels)

        model.fit(
            np_train_images,
            {
                "cl_head": np_train_labels,
                "bb_head": np_train_targets
            },
            validation_split=0.2,
            shuffle=True,
            epochs=epochs)
    else:
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size)

    # Save the model
    print("Saving model...")
    model_name = f"{ctx.obj['PWD']}/models/maskdetect-v1.12.h5"
    print(f"Model saved to: {model_name}")
    model.save(model_name)

    print("Training finished successfully!")


@cli_app.command
@click.pass_context
def train_svm(ctx):
    click.echo(f"Debug is {'on' if ctx.obj['DEBUG'] else 'off'}")
    # Train the SVM model
    pass


if __name__ == "__main__":
    cli_app(obj={})
