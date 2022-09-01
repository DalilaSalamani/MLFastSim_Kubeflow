from typing import NamedTuple, Tuple, Any, Dict


def generate(max_energy: int, checkpoint_dir: str, gen_dir: str, batch_size: int,
             original_dim: int, latent_dim: int, epsilon_std: int,
             mu: int, epochs: int, lr: float, outActiv: str, validation_split: float,
             wReco: int, wkl: float, ki: str, bi: str, earlyStop: bool) -> str:
    import tensorflow as tf
    from keras.layers import BatchNormalization
    from keras.layers import Input, Dense, Layer, concatenate
    from keras.models import Model
    from keras.optimizer_v1 import Adam
    from keras.metrics import Mean
    from keras.losses import binary_crossentropy
    def get_condition_arrays(geo, energy_particle, nb_events):
        cond_e = [energy_particle / max_energy] * nb_events
        cond_angle = [energy_particle / max_energy] * nb_events
        if geo == "SiW":
            cond_geo = [[0, 1]] * nb_events
        else:  # geo == "SciPb"
            cond_geo = [[1, 0]] * nb_events
        return cond_e, cond_angle, cond_geo

    class _Sampling(Layer):
        def call(self, inputs, **kwargs):
            z_mean, z_log_var = inputs
            z_sigma = tf.math.exp(0.5 * z_log_var)
            epsilon = tf.random.normal(tf.shape(z_sigma))
            print("done Sampling")
            return z_mean + z_sigma * epsilon

    class _Reparametrize(Layer):
        """
        Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
        distribution.
        The sampled vector z is given by sampled_z = mean + std * epsilon
        """

        def call(self, inputs, **kwargs):
            z_mean, z_log_var = inputs
            z_sigma = tf.math.exp(0.5 * z_log_var)
            epsilon = tf.random.normal(tf.shape(z_sigma))
            print("done Reparameterize")
            return z_mean + z_sigma * epsilon

    class VAE(Model):
        def get_config(self):
            config = super().get_config()
            config["encoder"] = self.encoder
            config["decoder"] = self.decoder
            print("done VAE CONFIG")
            return config

        def call(self, inputs, training=None, mask=None):
            _, e_input, angle_input, geo_input = inputs
            z, _, _ = self.encoder(inputs)
            print("done VAE CALL")
            return self.decoder([z, e_input, angle_input, geo_input])

        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self._set_inputs(inputs=self.encoder.inputs, outputs=self(self.encoder.inputs))
            self.total_loss_tracker = Mean(name="total_loss")
            self.val_total_loss_tracker = Mean(name="val_total_loss")
            print("done VAE INIT")

        @property
        def metrics(self):
            print("done VAE MERICS")
            return [self.total_loss_tracker, self.val_total_loss_tracker]

        def _perform_step(self, data: Any) -> Tuple[float, float, float]:
            # Unpack data.
            #             print("done")
            ([x_input, e_input, angle_input, geo_input],) = data
            # Encode data and get new probability distribution with a vector z sampled from it.
            z_mean, z_log_var, z = self.encoder([x_input, e_input, angle_input, geo_input])
            # Reconstruct the original data.
            reconstruction = self.decoder([z, e_input, angle_input, geo_input])

            # Reshape data.
            x_input = tf.expand_dims(x_input, -1)
            reconstruction = tf.expand_dims(reconstruction, -1)

            # Calculate reconstruction loss.
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(binary_crossentropy(x_input, reconstruction), axis=1))

            # Calculate Kullback-Leibler divergence.
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Calculated weighted total loss (ORIGINAL_DIM is a weight).
            total_loss = original_dim * reconstruction_loss + kl_loss
            print("done PERFORM STEP")
            return total_loss, reconstruction_loss, kl_loss

        def train_step(self, data: Any) -> Dict[str, object]:
            with tf.GradientTape() as tape:
                # Perform step, backpropagate it through the network and update the tracker.
                total_loss, reconstruction_loss, kl_loss = self._perform_step(data)

                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
            print("done TRAIN_STEP")
            return {"total_loss": self.total_loss_tracker.result()}

        def test_step(self, data: Any) -> Dict[str, object]:
            # Perform step and update the tracker (no backpropagation).
            val_total_loss, val_reconstruction_loss, val_kl_loss = self._perform_step(data)

            self.val_total_loss_tracker.update_state(val_total_loss)

            return {"total_loss": self.val_total_loss_tracker.result()}

    class VAEHandler:
        def __init__(self, original_dim: int = original_dim, latent_dim: int = latent_dim, batch_size: int = batch_size,
                     #                      intermediate_dims: List[int] = INTERMEDIATE_DIMS,
                     learning_rate: float = lr, epochs: int = epochs, activation: str = "relu",
                     out_activation: str = "sigmoid", validation_split: float = validation_split,
                     kernel_initializer: str = ki,
                     bias_initializer: str = bi, checkpoint_dir: str = checkpoint_dir,
                     early_stop: bool = True, save_model: bool = False, save_best: bool = True,
                     period: int = 5, patience: int = 5, min_delta: float = 0.01,
                     best_model_filename: str = "VAE-best.tf"):
            self._best_model_filename = best_model_filename
            self._min_delta = min_delta
            self._patience = patience
            self._save_best = save_best
            self._save_model = save_model
            self._original_dim = original_dim
            self.latent_dim = latent_dim
            self._intermediate_dims = [100, 50, 20, 14]
            self._batch_size = batch_size
            self._epochs = epochs
            self._activation = activation
            self._out_activation = out_activation
            self._validation_split = validation_split
            self._bias_initializer = bias_initializer
            self._kernel_initializer = kernel_initializer
            self._checkpoint_dir = checkpoint_dir
            self._early_stop = early_stop
            self._period = period

            # Build encoder and decoder.
            encoder = self._build_encoder()
            decoder = self._build_decoder()

            # Build VAE.
            self.model = VAE(encoder, decoder)
            print("done")
            # Manufacture an optimizer and compile model with.
            #             optimizer = OptimizerFactory.create_optimizer(optimizer_type, learning_rate)
            optimizer = Adam(learning_rate)
            self.model.compile(optimizer=optimizer,
                               metrics=[self.model.total_loss_tracker, self.model.val_total_loss_tracker])

        def _prepare_input_layers(self, for_encoder: bool) -> Tuple[Input, Input, Input, Input]:
            x_input = Input(shape=self._original_dim) if for_encoder else Input(shape=self.latent_dim)
            e_input = Input(shape=(1,))
            angle_input = Input(shape=(1,))
            geo_input = Input(shape=(2,))
            print("Done Prepaer Input Layer")
            return x_input, e_input, angle_input, geo_input

        def _build_encoder(self) -> Model:
            # Prepare input layer.
            x_input, e_input, angle_input, geo_input = self._prepare_input_layers(for_encoder=True)
            x = concatenate([x_input, e_input, angle_input, geo_input])
            # Construct hidden layers (Dense and Batch Normalization).
            for intermediate_dim in self._intermediate_dims:
                x = Dense(units=intermediate_dim, activation=self._activation,
                          kernel_initializer=self._kernel_initializer,
                          bias_initializer=self._bias_initializer)(x)
                x = BatchNormalization()(x)
            z_mean = Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
            # Sample a probe from the distribution.
            z = _Sampling()([z_mean, z_log_var])
            # Return model.
            print("Encoder Done")
            return Model(inputs=[x_input, e_input, angle_input, geo_input], outputs=[z_mean, z_log_var, z],
                         name="encoder")

        def _build_decoder(self) -> Model:

            # Prepare input layer.
            latent_input, e_input, angle_input, geo_input = self._prepare_input_layers(for_encoder=False)
            x = concatenate([latent_input, e_input, angle_input, geo_input])
            # Construct hidden layers (Dense and Batch Normalization).
            for intermediate_dim in reversed(self._intermediate_dims):
                x = Dense(units=intermediate_dim, activation=self._activation,
                          kernel_initializer=self._kernel_initializer,
                          bias_initializer=self._bias_initializer)(x)
                x = BatchNormalization()(x)
            # Add Dense layer to get output which shape is compatible in an input's shape.
            decoder_outputs = Dense(units=self._original_dim, activation=self._out_activation)(x)
            # Return model.
            print("Decoder Done")
            return Model(inputs=[latent_input, e_input, angle_input, geo_input], outputs=decoder_outputs,
                         name="decoder")

    import numpy as np
    energy_particle = 1
    angle_particle = 50
    geometry = "SciPb"
    nb_events = 10
    epoch_generate = "best"
    # 1. Get condition values
    cond_e, cond_angle, cond_geo = get_condition_arrays(geometry, energy_particle, nb_events)
    # 2. Load a saved model
    vae = VAEHandler()
    # Load the saved weights
    print(f"{checkpoint_dir}VAE-{epoch_generate}.tf")
    vae = tf.keras.models.load_model(f"{checkpoint_dir}VAE-{epoch_generate}.tf")
    generator = vae.decoder
    # 3. Generate showers using the VAE model by sampling from the prior (normal distribution) in d dimension
    # (d=latent_dim, latent space dimension)
    z_r = np.random.normal(loc=0, scale=1, size=(nb_events, latent_dim))
    cond_e = np.array(cond_e)
    cond_angle = np.array(cond_angle)
    cond_geo = np.array(cond_geo)

    generated_events = (generator.predict([z_r, cond_e, cond_angle, cond_geo]) * (energy_particle * 1000))
    # 4. Save the generated showers
    saved_gen = f"{gen_dir}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt"
    np.savetxt(
        f"{gen_dir}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt",
        generated_events)
    return saved_gen
