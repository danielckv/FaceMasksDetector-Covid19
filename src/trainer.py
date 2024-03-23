import click
from click import parser
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli_app(ctx, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug


@cli_app.command
@click.pass_context
@click.option('--train_data_dir', default='./datasets/train', help='The directory containing the training data.')
@click.option('--validation_data_dir', default='./datasets/validation',
              help='The directory containing the validation data.')
@click.option('--img_width', default=128, help='The width of the input images.')
@click.option('--img_height', default=128, help='The height of the input images.')
@click.option('--batch_size', default=32, help='The batch size for training.')
@click.option('--device', default='mps', help='The device to use for training (cpu, cuda or mps).')
def train_model(ctx, train_data_dir, validation_data_dir, img_width, img_height, batch_size):
    click.echo(f"Debug is {'on' if ctx.obj['DEBUG'] else 'off'}")
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

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

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 10
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    # Save the model
    model.save('danielckv-maskdetect-v1.12.h5')


@cli_app.command
@click.pass_context
def train_svm(ctx):
    click.echo(f"Debug is {'on' if ctx.obj['DEBUG'] else 'off'}")
    # Train the SVM model
    pass


if __name__ == "__main__":
    cli_app(obj={})
