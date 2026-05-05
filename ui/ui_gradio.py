"""
Interfaz Gradio para el Clasificador de Acento: Peninsular vs Canario
Ejecutar con: python ui_gradio.py

Requiere: pip install gradio librosa tensorflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
from AudioAugmentation import EspectrogramaAugmentation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from pathlib import Path




# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
SAMPLE_RATE  = 22050
DURACION_SEG = 5
N_MELS       = 128
HOP_LENGTH   = 512
N_FFT        = 2048
IMG_HEIGHT   = 128
IMG_WIDTH    = 128
CLASES       = ['peninsular', 'canario']
MODELO_PATH  = Path("models/modelo_acento_final.keras")

EMOJIS_CLASE = {
    'peninsular': '🏰',
    'canario': '🌋'
}

DESCRIPCIONES = {
    'peninsular': (
        "Se detecta acento peninsular (castellano estándar). "
        "Características típicas: pronunciación de la /s/ alveolar, "
        "distinción entre /s/ y /θ/ (ceceo/seseo peninsular), "
        "entonación más plana y formal del centro-norte de España."
    ),
    'canario': (
        "Se detecta acento canario. "
        "Características típicas: seseo (la /c/ y /z/ se pronuncian como /s/), "
        "aspiración de la /s/ final de sílaba, entonación más melódica "
        "con influencia del español atlántico, léxico con americanismos."
    )
}


# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
modelo = None

def cargar_modelo():
    global modelo
    if not MODELO_PATH.exists():
        print(f"⚠️  Modelo no encontrado en {MODELO_PATH}")
        print("   Ejecuta primero: python src/modelo_acento.py")
        return False
    try:
        modelo = tf.keras.models.load_model(
        str(MODELO_PATH),
        custom_objects={"EspectrogramaAugmentation": EspectrogramaAugmentation}
        )
        print(f"✅ Modelo cargado desde {MODELO_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False


# ─────────────────────────────────────────────
# PROCESAMIENTO DE AUDIO
# ─────────────────────────────────────────────

def procesar_audio(ruta_audio: str):
    """Carga y preprocesa un audio para la predicción."""
    y, sr = librosa.load(ruta_audio, sr=SAMPLE_RATE, duration=DURACION_SEG)
    samples = int(SAMPLE_RATE * DURACION_SEG)

    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    else:
        y = y[:samples]

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_resized = tf.image.resize(mel_norm[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH]).numpy()

    return mel_resized[np.newaxis, ...], mel_db


def generar_imagen_espectrograma(mel_db: np.ndarray, clase_pred: str, confianza: float):
    """Genera una imagen del espectrograma Mel con anotaciones."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')

    # Espectrograma
    ax1 = axes[0]
    img = ax1.imshow(mel_db, aspect='auto', origin='lower',
                     cmap='magma', extent=[0, DURACION_SEG, 0, SAMPLE_RATE//2/1000])
    ax1.set_xlabel('Tiempo (s)', color='white')
    ax1.set_ylabel('Frecuencia (kHz)', color='white')
    ax1.set_title('Espectrograma Mel', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    plt.colorbar(img, ax=ax1, label='dB').ax.yaxis.set_tick_params(color='white')

    # Barra de confianza
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    colores = ['#4ecdc4' if c == clase_pred else '#444' for c in CLASES]
    barras = ax2.barh(CLASES, [0, 0], color=colores)  # placeholder

    # Obtener probabilidades reales si el modelo está cargado
    ax2.set_xlim(0, 1)
    barra_pred = [confianza if c == clase_pred else 1 - confianza for c in CLASES]
    ax2.barh(CLASES, barra_pred, color=colores, height=0.5)

    for i, (clase, prob) in enumerate(zip(CLASES, barra_pred)):
        ax2.text(prob + 0.02, i, f'{prob:.1%}', va='center',
                 color='white', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Confianza', color='white')
    ax2.set_title(f'Predicción: {EMOJIS_CLASE[clase_pred]} {clase_pred.upper()}',
                  color='#4ecdc4', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.set_xlim(0, 1.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#1a1a2e', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def predecir(audio):
    """Función principal para Gradio: recibe audio y retorna predicción."""
    if audio is None:
        return "⚠️ Por favor, sube o graba un audio.", None, ""

    if modelo is None:
        return (
            "❌ Modelo no cargado. Ejecuta primero el entrenamiento.\n"
            "   `python src/modelo_acento.py`",
            None, ""
        )

    try:
        # Procesar audio
        X, mel_db = procesar_audio(audio)

        # Predicción
        probs = modelo.predict(X, verbose=0)[0]
        idx_pred = np.argmax(probs)
        clase_pred = CLASES[idx_pred]
        confianza = float(probs[idx_pred])

        # Texto resultado
        emoji = EMOJIS_CLASE[clase_pred]
        nivel_confianza = "🟢 Alta" if confianza > 0.8 else ("🟡 Media" if confianza > 0.6 else "🔴 Baja")

        resultado_texto = (
            f"## {emoji} Acento detectado: **{clase_pred.upper()}**\n\n"
            f"**Confianza:** {confianza:.1%} — {nivel_confianza}\n\n"
            f"**Descripción:**\n{DESCRIPCIONES[clase_pred]}\n\n"
            f"---\n"
            f"🟦 Peninsular: `{probs[0]:.1%}` | 🟧 Canario: `{probs[1]:.1%}`"
        )

        # Imagen del espectrograma
        buf = generar_imagen_espectrograma(mel_db, clase_pred, confianza)
        from PIL import Image
        img = Image.open(buf)

        return resultado_texto, img, ""

    except Exception as e:
        return f"❌ Error procesando el audio: {str(e)}", None, ""


# ─────────────────────────────────────────────
# INTERFAZ GRADIO
# ─────────────────────────────────────────────

CSS = """
.gradio-container {
    max-width: 900px !important;
    background: #0f0f1a !important;
}
.resultado-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2d2d5e;
}
h1 { color: #4ecdc4 !important; }
"""

def construir_ui():
    with gr.Blocks(
        title="Clasificador de Acento: Peninsular vs Canario",
        theme=gr.themes.Base(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate"
        )
    ) as demo:

        gr.Markdown("""
        # 🎙️ Clasificador de Acento: Peninsular vs Canario
        ### Detección automática de acento mediante CNN y espectrogramas Mel
        
        Sube un audio **.wav** con voz hablando en español o grábate directamente.
        El modelo analiza el espectrograma y clasifica el acento.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="📎 Audio de entrada",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format="wav"
                )
                btn = gr.Button("🔍 Clasificar acento", variant="primary", size="lg")

                gr.Markdown("""
                **💡 Instrucciones:**
                - Sube un `.wav` de al menos 3 segundos
                - O grábate hablando 5 segundos
                - Habla de forma natural, como en una conversación
                - El audio se analiza en los **primeros 5 segundos**
                """)

            with gr.Column(scale=1):
                resultado = gr.Markdown(label="Resultado")
                espectrograma = gr.Image(label="Análisis visual", type="pil")

        gr.Markdown("---")

        with gr.Accordion("ℹ️ Sobre el modelo", open=False):
            gr.Markdown("""
            **Arquitectura:** CNN con 4 bloques convolucionales + BatchNormalization + Dropout

            **Preprocesamiento:**
            - Señal de audio → Espectrograma Mel (128 bandas, 22050 Hz)
            - Escala logarítmica (dB) → Normalización [0,1]
            - Redimensión a 128×128 píxeles

            **Data Augmentation** aplicada durante entrenamiento:
            - Ruido gaussiano, desplazamiento temporal, cambio de volumen
            - Time stretch (±15%), Pitch shift (±2 semitonos)
            - SpecAugment: enmascaramiento de frecuencias y tiempo

            **Dataset:** Clips de 5s extraídos de podcasts en español
            - Acento peninsular: Madrid/Castilla (El Hormiguero, La Resistencia...)
            - Acento canario: Radio Canarias, podcasters canarios...

            **Limitaciones:** Ver reflexión en la memoria de la práctica.
            """)

        btn.click(
            fn=predecir,
            inputs=[audio_input],
            outputs=[resultado, espectrograma, gr.Textbox(visible=False)]
        )

    return demo


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎙️  Clasificador de Acento: Peninsular vs Canario")
    print("="*55)

    cargar_modelo()

    demo = construir_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
