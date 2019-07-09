import tensorflow as tf

from verres.layers import gradient_barrier, correlation, global_pooling


class LocalErrorBase:

    def __init__(self,
                 base_layer_type,
                 base_layer_kwargs,
                 base_layer_activation,
                 batch_normalization,
                 use_gradient_barrier,
                 use_similarity_loss,
                 use_label_prediction_loss,
                 projection_kwargs,
                 num_output_classes,
                 label_prediction_activation):

        self.base_layer_type = base_layer_type
        self.base_layer_type_str = base_layer_type.__name__.lower()
        self.base_layer_kwargs = base_layer_kwargs
        self.base_layer_activation = base_layer_activation
        self.batch_normalization = batch_normalization
        self.projection_kwargs = projection_kwargs
        self.use_gradient_barrier = use_gradient_barrier
        self.use_similarity_loss = use_similarity_loss
        self.use_label_prediction_loss = use_label_prediction_loss
        self.num_output_classes = num_output_classes
        self.prediction_activation = label_prediction_activation

        if use_label_prediction_loss:
            if self.num_output_classes is None or label_prediction_activation is None:
                raise ValueError(
                    "Please set the number of output classes and label prediction activation "
                    "to use the intermediate label prediction loss."
                )

        self.base_layer = None
        self.batch_norm_layer = None
        self.activation_layer = None
        self.gradient_barrier_layer = None
        self.flatten_layer = None
        self.correlation_layer = None
        self.projection_layer = None
        self.label_prediction_layer = None
        self.global_pooling_layer = None

        self.output_feature = None
        self.correlation_output = None
        self.label_prediction_output = None

        self.built = False

    def build(self):
        self.base_layer = self.base_layer_type(**self.base_layer_kwargs)
        if self.base_layer_type.__name__.lower() == "conv2d":
            self.global_pooling_layer = global_pooling.GlobalSTDPooling2D()
            self.flatten_layer = tf.keras.layers.Flatten()

        if self.batch_normalization:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        if self.base_layer_activation is not None:
            self.activation_layer = tf.keras.layers.Activation(self.base_layer_activation)
        if self.use_label_prediction_loss or self.use_similarity_loss:
            self.projection_layer = self.base_layer_type(**self.projection_kwargs)
        if self.use_label_prediction_loss:
            self.label_prediction_layer = tf.keras.layers.Dense(
                units=self.num_output_classes, activation=self.prediction_activation
            )
        if self.use_similarity_loss:
            self.correlation_layer = correlation.Correlation()
        if self.use_gradient_barrier:
            self.gradient_barrier_layer = gradient_barrier.GradientBarrier()

        self.built = True

    def call(self, inputs):
        if not self.built:
            self.build()
        self.output_feature = self.base_layer(inputs)
        if self.batch_normalization:
            self.output_feature = self.batch_norm_layer(self.output_feature)
        if self.base_layer_activation is not None:
            self.output_feature = self.activation_layer(self.output_feature)
        if self.use_label_prediction_loss or self.use_similarity_loss:
            projection = self.projection_layer(self.output_feature)
            if self.use_label_prediction_loss:
                if self.global_pooling_layer is None:
                    self.label_prediction_output = self.label_prediction_layer(projection)
                else:
                    x = self.global_pooling_layer(projection)
                    self.label_prediction_output = self.label_prediction_layer(x)
            if self.use_similarity_loss:
                if self.flatten_layer is None:
                    self.correlation_output = self.correlation_layer(projection)
                else:
                    self.correlation_output = self.correlation_layer(self.flatten_layer(projection))

        if self.use_gradient_barrier:
            self.output_feature = self.gradient_barrier_layer(self.output_feature)

        return self.output_feature

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs)


class LocalErrorConvContainer(LocalErrorBase):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 name=None,
                 batch_normalization=True,
                 use_gradient_barrier=True,
                 use_similarity_loss=True,
                 use_label_prediction_loss=True,
                 projection_trainable=False,
                 projection_dimension=1024,
                 projection_activation=None,
                 num_output_classes=None,
                 label_prediction_activation=None,
                 **conv_kwargs):

        conv_kwargs.update(dict(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding=padding, name=name
        ))
        projection_kwargs = dict(
            filters=projection_dimension, kernel_size=(3, 3),
            trainable=projection_trainable, activation=projection_activation
        )
        super().__init__(base_layer_type=tf.keras.layers.Conv2D,
                         base_layer_kwargs=conv_kwargs,
                         base_layer_activation=activation,
                         batch_normalization=batch_normalization,
                         use_gradient_barrier=use_gradient_barrier,
                         use_similarity_loss=use_similarity_loss,
                         use_label_prediction_loss=use_label_prediction_loss,
                         projection_kwargs=projection_kwargs,
                         num_output_classes=num_output_classes,
                         label_prediction_activation=label_prediction_activation)


class LocalErrorDenseContainer(LocalErrorBase):

    def __init__(self,
                 units,
                 activation=None,
                 batch_normalization=True,
                 use_gradient_barrier=True,
                 use_similarity_loss=True,
                 use_label_prediction_loss=True,
                 projection_trainable=False,
                 projection_dimension=1024,
                 projection_activation=None,
                 num_output_classes=None,
                 label_prediction_activation=None,
                 **dense_kwargs):

        dense_kwargs.update(units=units)
        projection_kwargs = dict(
            units=projection_dimension, trainable=projection_trainable, activation=projection_activation
        )
        super().__init__(
            base_layer_type=tf.keras.layers.Dense,
            base_layer_kwargs=dense_kwargs,
            base_layer_activation=activation,
            batch_normalization=batch_normalization,
            use_gradient_barrier=use_gradient_barrier,
            use_similarity_loss=use_similarity_loss,
            use_label_prediction_loss=use_label_prediction_loss,
            projection_kwargs=projection_kwargs,
            num_output_classes=num_output_classes,
            label_prediction_activation=label_prediction_activation
        )
