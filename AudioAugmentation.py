"""
AudioAugmentation.py
Capa de Keras para Data Augmentation en audio con audiomentations.
Adaptada para la práctica de clasificación de acento peninsular vs canario.

Instalar: pip install audiomentations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

try:
    import audiomentations
    AUDIOMENTATIONS_DISPONIBLE = True
except ImportError:
    AUDIOMENTATIONS_DISPONIBLE = False
    print("⚠️  audiomentations no instalado. Usando augmentation básica.")
    print("   Instalar con: pip install audiomentations")


# ─────────────────────────────────────────────
# CAPA DE AUGMENTATION CON AUDIOMENTATIONS
# ─────────────────────────────────────────────

class AudioAugmentation(keras.layers.Layer):
    """
    Capa de Keras que aplica data augmentation a señales de audio.
    Sólo activa durante el entrenamiento (training=True).

    Técnicas aplicadas:
        - AddGaussianNoise: ruido de fondo realista
        - TimeStretch: simula hablar más rápido/lento
        - PitchShift: simula variaciones de tono entre hablantes
        - Shift: desplazamiento temporal de la señal
        - Gain: variación de volumen
    """

    def __init__(self, sample_rate: int = 22050, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.p = p

        if AUDIOMENTATIONS_DISPONIBLE:
            self.augment = audiomentations.Compose([
                # ── Añadir ruido gaussiano ──────────────────────
                # Simula grabaciones en entornos no ideales (estudio, exterior)
                audiomentations.AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=0.5
                ),

                # ── Desplazamiento temporal ──────────────────────
                # El habla no siempre empieza al inicio del clip
                audiomentations.Shift(
                    min_shift=-0.2,
                    max_shift=0.2,
                    p=0.4
                ),

                # ── Cambio de ganancia (volumen) ─────────────────
                # Diferentes micrófonos y distancias de grabación
                audiomentations.Gain(
                    min_gain_db=-6,
                    max_gain_db=6,
                    p=0.4
                ),

                # ── Cambio de velocidad ──────────────────────────
                # Algunos hablantes son más rápidos/lentos
                audiomentations.TimeStretch(
                    min_rate=0.85,
                    max_rate=1.15,
                    leave_length_unchanged=True,
                    p=0.3
                ),

                # ── Cambio de tono ───────────────────────────────
                # Variación natural entre hablantes del mismo acento
                audiomentations.PitchShift(
                    min_semitones=-2,
                    max_semitones=2,
                    p=0.3
                ),
            ])
        else:
            self.augment = None

    def call(self, inputs, training=None):
        """Aplica augmentation sólo durante el entrenamiento."""
        if not training:
            return inputs

        if self.augment is None:
            return self._augmentation_basica(inputs)

        # Aplicar augmentation con audiomentations
        def augmentar_batch(batch):
            resultados = []
            for sample in batch:
                audio_1d = sample.numpy().flatten().astype(np.float32)
                try:
                    audio_aug = self.augment(samples=audio_1d, sample_rate=self.sample_rate)
                    resultados.append(audio_aug.reshape(sample.shape))
                except Exception:
                    resultados.append(audio_1d.reshape(sample.shape))
            return np.array(resultados, dtype=np.float32)

        return tf.py_function(augmentar_batch, [inputs], tf.float32)

    def _augmentation_basica(self, inputs):
        """Augmentation básica en TensorFlow (sin audiomentations)."""
        # Ruido gaussiano
        ruido = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0, stddev=0.005
        )
        return tf.clip_by_value(inputs + ruido, -1.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'p': self.p
        })
        return config


# ─────────────────────────────────────────────
# AUGMENTATION SOBRE ESPECTROGRAMAS (en CNN)
# ─────────────────────────────────────────────

class EspectrogramaAugmentation(keras.layers.Layer):
    """
    Augmentation directamente sobre espectrogramas Mel.
    Más eficiente que augmentar el audio crudo.

    Técnicas:
        - SpecAugment: enmascaramiento de frecuencias y tiempo
        - Jitter de amplitud
    """

    def __init__(self, freq_mask_param: int = 20, time_mask_param: int = 20,
                 n_freq_masks: int = 2, n_time_masks: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param  = freq_mask_param
        self.time_mask_param  = time_mask_param
        self.n_freq_masks     = n_freq_masks
        self.n_time_masks     = n_time_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs

        x = inputs

        # ── SpecAugment: máscaras de frecuencia ─────────────
        for _ in range(self.n_freq_masks):
            f = tf.random.uniform([], 0, self.freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, tf.shape(x)[1] - f, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([f0, tf.shape(x)[2], 1]),
                tf.zeros([f, tf.shape(x)[2], 1]),
                tf.ones([tf.shape(x)[1] - f0 - f, tf.shape(x)[2], 1])
            ], axis=0)
            x = x * mask[tf.newaxis, ...]

        # ── SpecAugment: máscaras de tiempo ─────────────────
        for _ in range(self.n_time_masks):
            t = tf.random.uniform([], 0, self.time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, tf.shape(x)[2] - t, dtype=tf.int32)
            mask = tf.concat([
                tf.ones([tf.shape(x)[1], t0, 1]),
                tf.zeros([tf.shape(x)[1], t, 1]),
                tf.ones([tf.shape(x)[1], tf.shape(x)[2] - t0 - t, 1])
            ], axis=1)
            x = x * mask[tf.newaxis, ...]

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'freq_mask_param': self.freq_mask_param,
            'time_mask_param': self.time_mask_param,
            'n_freq_masks': self.n_freq_masks,
            'n_time_masks': self.n_time_masks,
        })
        return config


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Test de AudioAugmentation")
    print("=" * 50)

    # Simular un batch de espectrogramas
    batch = np.random.rand(4, 128, 128, 1).astype(np.float32)
    print(f"Input shape: {batch.shape}")

    # Augmentation sobre espectrogramas
    aug_layer = EspectrogramaAugmentation(freq_mask_param=20, time_mask_param=20)
    output = aug_layer(batch, training=True)
    print(f"Output shape (espectrograma aug): {output.shape}")

    if AUDIOMENTATIONS_DISPONIBLE:
        print("✅ audiomentations disponible → AudioAugmentation activa")
    else:
        print("⚠️  audiomentations NO disponible → augmentation básica")

    print("\nAugmentaciones configuradas:")
    print("  Audio (audiomentations):")
    print("    • AddGaussianNoise     → ruido de grabación")
    print("    • Shift                → desplazamiento temporal")
    print("    • Gain                 → variación de volumen")
    print("    • TimeStretch          → velocidad de habla")
    print("    • PitchShift           → tono del hablante")
    print("  Espectrograma (SpecAugment):")
    print("    • Freq masking         → oculta bandas de frecuencia")
    print("    • Time masking         → oculta segmentos temporales")
