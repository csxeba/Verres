import keras


def build(num_output_classes, onehot_y=False):
    inputs = keras.layers.Input((200, 320, 3))

    down_stage1 = keras.layers.Conv2D(8, 3, padding="same")(inputs)
    down_stage1 = keras.layers.BatchNormalization()(down_stage1)
    down_stage1 = keras.layers.ReLU()(down_stage1)

    down_stage2 = keras.layers.MaxPool2D()(down_stage1)  # 100
    down_stage2 = keras.layers.Conv2D(8, 3, padding="same")(down_stage2)
    down_stage2 = keras.layers.BatchNormalization()(down_stage2)
    down_stage2 = keras.layers.ReLU()(down_stage2)

    down_stage3 = keras.layers.MaxPool2D()(down_stage2)  # 50
    down_stage3 = keras.layers.Conv2D(16, 3, padding="same")(down_stage3)
    down_stage3 = keras.layers.BatchNormalization()(down_stage3)
    down_stage3 = keras.layers.ReLU()(down_stage3)

    down_stage4 = keras.layers.MaxPool2D()(down_stage3)  # 25
    down_stage4 = keras.layers.Conv2D(16, (2, 1), padding="valid")(down_stage4)  # 24
    down_stage4 = keras.layers.BatchNormalization()(down_stage4)
    down_stage4 = keras.layers.ReLU()(down_stage4)

    down_stage5 = keras.layers.MaxPool2D()(down_stage4)  # 12
    down_stage5 = keras.layers.Conv2D(16, 3, padding="same")(down_stage5)
    down_stage5 = keras.layers.BatchNormalization()(down_stage5)
    down_stage5 = keras.layers.ReLU()(down_stage5)

    down_stage6 = keras.layers.MaxPool2D()(down_stage5)  # 6, 10
    down_stage6 = keras.layers.Conv2D(16, 3, padding="same")(down_stage6)
    down_stage6 = keras.layers.BatchNormalization()(down_stage6)
    down_stage6 = keras.layers.ReLU()(down_stage6)

    down_stage7 = keras.layers.MaxPool2D()(down_stage6)  # 3, 5
    down_stage7 = keras.layers.Conv2D(32, 3, padding="same")(down_stage7)
    down_stage7 = keras.layers.BatchNormalization()(down_stage7)
    down_stage7 = keras.layers.ReLU()(down_stage7)

    x = keras.layers.UpSampling2D()(down_stage7)  # 6, 10
    x = keras.layers.concatenate([down_stage6, x])
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)  # 12, 20
    x = keras.layers.concatenate([down_stage5, x])
    x = keras.layers.Conv2D(16, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)  # 24, 40
    x = keras.layers.concatenate([down_stage4, x])
    x = keras.layers.Conv2DTranspose(16, (2, 1), padding="valid")(x)  # 25, 40
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)  # 50, 80
    x = keras.layers.concatenate([down_stage3, x])
    x = keras.layers.Conv2D(16, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)  # 100, 160
    x = keras.layers.concatenate([down_stage2, x])
    x = keras.layers.Conv2D(8, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.UpSampling2D()(x)  # 200, 320
    classes = keras.layers.Conv2D(num_output_classes+1, 5, activation="softmax", padding="same")(x)

    model = keras.models.Model(inputs, classes)
    loss = keras.losses.categorical_crossentropy if onehot_y else keras.losses.sparse_categorical_crossentropy
    model.compile("adam", loss)
    # keras.utils.plot_model(model, "unet.png", show_shapes=True)
    return model


if __name__ == '__main__':
    build(num_output_classes=10)
