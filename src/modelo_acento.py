"""
Clasificador de Acento: Peninsular vs Canario
CNN con espectrogramas Mel y SpecAugment Integrado
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── NUEVA IMPORTACIÓN: Conectamos el archivo avanzado ──
from AudioAugmentation import EspectrogramaAugmentation

# ─────────────────────────────────────────────
# PARÁMETROS GLOBALES
# ─────────────────────────────────────────────
SAMPLE_RATE    = 22050
DURACION_SEG   = 5         
N_MELS         = 128       
HOP_LENGTH     = 512
N_FFT          = 2048
BATCH_SIZE     = 32
EPOCHS         = 50
IMG_HEIGHT     = 128
IMG_WIDTH      = 128
CLASES         = ['peninsular', 'canario']
DATA_DIR       = Path("data")
MODELS_DIR     = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. CARGA Y PREPROCESAMIENTO DE AUDIO
# ─────────────────────────────────────────────

def cargar_audio(ruta_fichero: str, sr: int = SAMPLE_RATE, duracion: float = DURACION_SEG):
    y, sr_original = librosa.load(ruta_fichero, sr=sr, duration=duracion)
    samples_necesarios = int(sr * duracion)

    if len(y) < samples_necesarios:
        y = np.pad(y, (0, samples_necesarios - len(y)))
    else:
        y = y[:samples_necesarios]

    return y, sr


def audio_a_melspectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm


def cargar_dataset(data_dir: Path):
    X, y = [], []
    for idx, clase in enumerate(CLASES):
        carpeta = data_dir / clase
        if not carpeta.exists():
            continue

        ficheros = list(carpeta.glob("*.wav")) + list(carpeta.glob("*.WAV"))
        for fichero in ficheros:
            try:
                audio, sr = cargar_audio(str(fichero))
                mel = audio_a_melspectrogram(audio, sr)
                mel_resized = tf.image.resize(mel[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH]).numpy()
                X.append(mel_resized)
                y.append(idx)
            except Exception:
                pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION EN AUDIO (Offline)
# ─────────────────────────────────────────────

def augmentar_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    augmentaciones = []
    ruido = np.random.randn(len(y)) * 0.005
    augmentaciones.append(y + ruido)
    
    shift = int(sr * 0.3)
    augmentaciones.append(np.roll(y, shift))

    try:
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        y_pitch_dn = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        augmentaciones.append(y_pitch_up)
        augmentaciones.append(y_pitch_dn)
    except Exception:
        pass

    try:
        y_fast = librosa.effects.time_stretch(y, rate=1.1)
        y_slow = librosa.effects.time_stretch(y, rate=0.9)
        samples = int(SAMPLE_RATE * DURACION_SEG)
        for ys in [y_fast, y_slow]:
            if len(ys) >= samples:
                augmentaciones.append(ys[:samples])
            else:
                augmentaciones.append(np.pad(ys, (0, samples - len(ys))))
    except Exception:
        pass

    return augmentaciones


def cargar_dataset_con_augmentation(data_dir: Path, augmentar: bool = True):
    X, y_labels = [], []
    for idx, clase in enumerate(CLASES):
        carpeta = data_dir / clase
        if not carpeta.exists():
            continue

        ficheros = list(carpeta.glob("*.wav")) + list(carpeta.glob("*.WAV"))
        print(f"📂 {clase}: {len(ficheros)} audios originales")

        for fichero in ficheros:
            try:
                audio, sr = cargar_audio(str(fichero))
                mel = audio_a_melspectrogram(audio, sr)
                mel_resized = tf.image.resize(mel[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH]).numpy()
                X.append(mel_resized)
                y_labels.append(idx)

                if augmentar:
                    for audio_aug in augmentar_audio(audio, sr):
                        mel_aug = audio_a_melspectrogram(audio_aug, sr)
                        mel_aug_resized = tf.image.resize(
                            mel_aug[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH]
                        ).numpy()
                        X.append(mel_aug_resized)
                        y_labels.append(idx)
            except Exception as e:
                pass
        print(f"   → Con augmentation: {y_labels.count(idx)} muestras para '{clase}'")

    X = np.array(X, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)
    print(f"\n✅ Dataset final: {X.shape[0]} muestras, shape={X.shape}")
    return X, y_labels


# ─────────────────────────────────────────────
# 3. ARQUITECTURA CNN (Ahora con SpecAugment integrado)
# ─────────────────────────────────────────────


def construir_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_clases=2):
    model = keras.Sequential([
        # ── 1. CAPA DE ENTRADA EXPLICITA ──
        keras.Input(shape=input_shape),

        # ── 2. CAPA DE SPECAUGMENT INTEGRADA (De AudioAugmentation.py) ──
        EspectrogramaAugmentation(freq_mask_param=20, time_mask_param=20, name="SpecAugment"),

        # ── Bloque 1 ──────────────────────────────
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloque 2 ──────────────────────────────
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloque 3 ──────────────────────────────
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloque 4 ──────────────────────────────
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # ── Clasificador ──────────────────────────
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ], name="CNN_Acento_Peninsular_vs_Canario")

    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=False
)
    return model


def construir_cnn_ligera(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_clases=2):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        EspectrogramaAugmentation(freq_mask_param=15, time_mask_param=15, name="SpecAugment_Ligero"),

        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_clases, activation='softmax')
    ], name="CNN_Acento_Ligera")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO Y VISUALIZACIÓN
# ─────────────────────────────────────────────

print("Entrenando modelo")

def entrenar_modelo(model, X_train, y_train, X_val, y_val):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "mejor_modelo.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return history

def graficar_historia(history, guardar=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Entrenamiento CNN — Acento Peninsular vs Canario', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='coral', linewidth=2)
    axes[0].set_title('Accuracy por época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='coral', linewidth=2)
    axes[1].set_title('Loss por época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if guardar:
        plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def graficar_matriz_confusion(y_true, y_pred, guardar=True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASES, yticklabels=CLASES,
                linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Confusión\nPeninsular vs Canario', fontweight='bold')
    ax.set_ylabel('Etiqueta Real')
    ax.set_xlabel('Predicción')
    plt.tight_layout()
    if guardar:
        plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualizar_espectrogramas(X, y, n=6):
    fig, axes = plt.subplots(2, n//2, figsize=(14, 6))
    fig.suptitle('Ejemplos de Espectrogramas Mel', fontsize=13, fontweight='bold')

    for clase_idx, clase_nombre in enumerate(CLASES):
        indices = np.where(y == clase_idx)[0][:n//2]
        for col, idx in enumerate(indices):
            ax = axes[clase_idx][col]
            ax.imshow(X[idx, :, :, 0], aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f'{clase_nombre}\nmuestra {col+1}', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('models/espectrogramas_ejemplo.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def pipeline_completo(usar_augmentation=True, arquitectura='completa'):
    print("=" * 60)
    print("   CNN CLASIFICADOR DE ACENTO: Peninsular vs Canario")
    print("=" * 60)

    print("\n📥 CARGANDO DATOS...")
    if usar_augmentation:
        X, y = cargar_dataset_con_augmentation(DATA_DIR, augmentar=True)
    else:
        X, y = cargar_dataset(DATA_DIR)

    if len(X) == 0:
        print("❌ No se encontraron datos. Asegúrate de tener audios en data/peninsular/ y data/canario/")
        return

    visualizar_espectrogramas(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\n📊 Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    print("\n🏗️  CONSTRUYENDO MODELO CNN CON SPECAUGMENT...")
    if arquitectura == 'ligera':
        model = construir_cnn_ligera()
    else:
        model = construir_cnn()
    model.summary()

    print("\n🚀 ENTRENANDO...")
    history = entrenar_modelo(model, X_train, y_train, X_val, y_val)

    print("\n🔍 EVALUANDO EN TEST SET...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"   Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=CLASES))

    graficar_historia(history)
    graficar_matriz_confusion(y_test, y_pred)

    model.save(str(MODELS_DIR / "modelo_acento_final.keras"))
    print(f"\n💾 Modelo guardado en {MODELS_DIR}/modelo_acento_final.keras")

    return model, history


if __name__ == "__main__":
    pipeline_completo(usar_augmentation=True, arquitectura='completa')