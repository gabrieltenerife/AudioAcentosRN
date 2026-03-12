"""
Descarga de podcasts de YouTube para el dataset.
Requiere: pip install yt-dlp pydub

Uso:
    python descargar_podcasts.py

Este script descarga fragmentos de podcasts y los corta en clips de 5 segundos.
"""

import os
import subprocess
import json
from pathlib import Path
import sys

# ─────────────────────────────────────────────
# FUENTES DE AUDIO RECOMENDADAS
# ─────────────────────────────────────────────
# 
# 🗣️ ACENTO PENINSULAR (Madrid/Castilla):
#   - El Hormiguero, La Ventana (Cadena SER), La Brújula (COPE)
#   - Podcasts de El País, El Mundo, The Wild Project (Jordi Wild)
#   - URLs de ejemplo (reemplazar con URLs reales):

FUENTES_PENINSULAR = [
    "https://www.youtube.com/watch?v=gMC6NNbxRJ0",
    "https://www.youtube.com/watch?v=L7QSthKW3IA"
]

FUENTES_CANARIO = [
    "https://www.youtube.com/watch?v=E_tWiBC9CpI",
    "https://www.youtube.com/watch?v=hoAN51vcGTc"
]

# ─────────────────────────────────────────────
# PODCASTS SUGERIDOS PARA BÚSQUEDA MANUAL
# ─────────────────────────────────────────────
SUGERENCIAS = {
    "peninsular": [
        "El Hormiguero - entrevistas en YouTube (acento madrileño)",
        "Iker Jiménez - Milenio 3 (Cuatro TV YouTube)",
        "Jordi Wild - The Wild Project podcast",
        "Luis Piedrahita - monólogos",
        "Buenafuente Late Night",
        "Nacho García - videos de stand-up",
        "La Resistencia - entrevistas cortas (YouTube)",
    ],
    "canario": [
        "Buscar: 'entrevista canaria podcast' en YouTube",
        "Radio Club Tenerife - programas en YouTube",
        "Canarias7 - entrevistas y reportajes",
        "Buscar: 'canario hablando podcast' o 'acento canario'",
        "Podcasts de youtubers canarios",
        "Entrevistas a políticos o personajes públicos canarios",
    ]
}


def verificar_dependencias():
    """Verifica que yt-dlp y ffmpeg estén instalados."""
    ok = True
    for cmd in ['yt-dlp', 'ffmpeg']:
        result = subprocess.run(['which', cmd], capture_output=True)
        if result.returncode != 0:
            print(f"❌ {cmd} no encontrado. Instalar con:")
            if cmd == 'yt-dlp':
                print("   pip install yt-dlp")
            else:
                print("   sudo apt install ffmpeg  (Ubuntu/Debian)")
                print("   brew install ffmpeg      (macOS)")
            ok = False
        else:
            print(f"✅ {cmd} disponible")
    return ok


def descargar_audio_wav(url: str, carpeta_salida: Path, nombre_base: str):
    """
    Descarga el audio de un vídeo de YouTube y lo convierte a WAV mono 22050Hz.
    Retorna la ruta del archivo descargado.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_salida = str(carpeta_salida / f"{nombre_base}.wav")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",  # mono, 22050Hz
        "-o", ruta_salida,
        "--no-playlist",
        url
    ]

    print(f"  ⬇️  Descargando: {url[:60]}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ Error: {result.stderr[:200]}")
        return None

    print(f"  ✅ Guardado en: {ruta_salida}")
    return ruta_salida


def cortar_en_clips(ruta_wav: str, carpeta_salida: Path, duracion_seg: int = 5,
                    inicio_seg: int = 30, num_clips: int = 40):
    """
    Corta un archivo WAV en clips de duración fija.
    Empieza desde inicio_seg para evitar silencios/música inicial.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    nombre_base = Path(ruta_wav).stem
    clips_creados = 0

    print(f"  ✂️  Cortando {nombre_base} en clips de {duracion_seg}s (desde seg {inicio_seg})...")

    for i in range(num_clips):
        t_inicio = inicio_seg + i * duracion_seg
        ruta_clip = str(carpeta_salida / f"{nombre_base}_clip_{i:03d}.wav")

        cmd = [
            "ffmpeg", "-i", ruta_wav,
            "-ss", str(t_inicio),
            "-t", str(duracion_seg),
            "-ar", "22050", "-ac", "1",
            "-y",  # sobrescribir
            ruta_clip
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and Path(ruta_clip).stat().st_size > 1000:
            clips_creados += 1
        else:
            break  # Fin del audio

    print(f"  ✅ {clips_creados} clips creados")
    return clips_creados


def procesar_dataset_completo(data_dir: Path = Path("data")):
    """Descarga y procesa todas las fuentes del dataset."""

    print("\n" + "="*60)
    print("  DESCARGA DE DATASET: Acento Peninsular vs Canario")
    print("="*60)

    if not verificar_dependencias():
        print("\n⚠️  Instala las dependencias antes de continuar.")
        return

    total_peninsular = 0
    total_canario = 0

    # ── Peninsular ──────────────────────────────
    if FUENTES_PENINSULAR:
        print("\n🗣️  DESCARGANDO ACENTO PENINSULAR...")
        carpeta_peninsular = data_dir / "peninsular"
        for i, url in enumerate(FUENTES_PENINSULAR):
            wav = descargar_audio_wav(url, data_dir / "_tmp", f"peninsular_{i:02d}")
            if wav:
                n = cortar_en_clips(wav, carpeta_peninsular, duracion_seg=5,
                                     inicio_seg=30, num_clips=50)
                total_peninsular += n
    else:
        print("\n⚠️  No hay URLs para peninsular. Añade URLs en FUENTES_PENINSULAR")

    # ── Canario ──────────────────────────────────
    if FUENTES_CANARIO:
        print("\n🌊 DESCARGANDO ACENTO CANARIO...")
        carpeta_canario = data_dir / "canario"
        for i, url in enumerate(FUENTES_CANARIO):
            wav = descargar_audio_wav(url, data_dir / "_tmp", f"canario_{i:02d}")
            if wav:
                n = cortar_en_clips(wav, carpeta_canario, duracion_seg=5,
                                     inicio_seg=30, num_clips=50)
                total_canario += n
    else:
        print("\n⚠️  No hay URLs para canario. Añade URLs en FUENTES_CANARIO")

    # ── Resumen ──────────────────────────────────
    print("\n" + "="*60)
    print("  RESUMEN DEL DATASET")
    print("="*60)
    print(f"  Peninsular : {total_peninsular} clips de 5s")
    print(f"  Canario    : {total_canario} clips de 5s")

    if total_peninsular < 200 or total_canario < 200:
        print(f"\n⚠️  La práctica requiere MÍNIMO 200 audios por clase.")
        print("    Añade más URLs o usa el data augmentation del modelo.")

    return total_peninsular, total_canario


def mostrar_sugerencias():
    """Muestra sugerencias de podcasts para cada acento."""
    print("\n" + "="*60)
    print("  SUGERENCIAS DE FUENTES DE AUDIO")
    print("="*60)
    for acento, fuentes in SUGERENCIAS.items():
        print(f"\n🎙️  ACENTO {acento.upper()}:")
        for f in fuentes:
            print(f"   • {f}")
    print("\n💡 Busca estos podcasts en YouTube, copia las URLs")
    print("   y pégalas en las listas FUENTES_PENINSULAR / FUENTES_CANARIO")
    print("   de este script.\n")


if __name__ == "__main__":
    if "--sugerencias" in sys.argv:
        mostrar_sugerencias()
    else:
        mostrar_sugerencias()
        procesar_dataset_completo()
