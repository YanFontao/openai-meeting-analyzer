import os
import sys
import whisper
import torch
from pyannote.audio import Pipeline
from datetime import timedelta
import tempfile
import time
from fpdf import FPDF
import zipfile
import warnings
from colorama import Fore, Style, init
import subprocess
import logging
import json
import time as tempo_exec
from tqdm import tqdm

# Iniciar colorama
init(autoreset=True)

# Suprimir avisos e logs técnicos
warnings.filterwarnings("ignore")
for lib in ["pyannote", "lightning", "torch", "speechbrain"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)

# Token Hugging Face
HUGGINGFACE_TOKEN = "INSERTAPIKEY"

# Otimiza uso da GPU
torch.backends.cudnn.benchmark = True
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Dispositivo de processamento: {Fore.GREEN}{DISPOSITIVO}{Style.RESET_ALL}\n")

# Carregar pipeline do pyannote
try:
    print(f"🔁 Iniciando diarização com pyannote...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    pipeline.to(DISPOSITIVO)
except Exception as e:
    print(f"\n❌ Não foi possível carregar o modelo 'pyannote/speaker-diarization'.")
    print("👉 Acesse https://huggingface.co/pyannote/speaker-diarization e clique em 'Access repository'")
    print("👉 Também acesse https://huggingface.co/pyannote/segmentation e clique em 'Access repository'")
    print(f"Detalhes do erro: {e}")
    sys.exit(1)

def format_tempo(seg):
    return str(timedelta(seconds=int(seg)))

def format_srt(seg):
    t = str(timedelta(seconds=int(seg)))
    if len(t.split(':')) == 2:
        t = "00:" + t
    return t.replace(".", ",") + ",000"

def extrair_trecho_ffmpeg(audio_path, start_time, end_time):
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp.close()
    output_path = temp.name
    comando = [
        "ffmpeg", "-y", "-i", audio_path, "-ss", str(start_time), "-to", str(end_time),
        "-ar", "16000", "-ac", "1", "-loglevel", "error", output_path
    ]
    subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def converter_video_para_audio(caminho):
    extensao = os.path.splitext(caminho)[1].lower()
    if extensao in ['.mp4', '.mkv', '.mov', '.avi']:
        caminho_convertido = caminho.replace(extensao, ".wav")
        comando = [
            "ffmpeg", "-y", "-i", caminho, "-ar", "16000", "-ac", "1", "-loglevel", "error", caminho_convertido
        ]
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return caminho_convertido
    return caminho

def transcrever_com_diarizacao(caminho_audio, idioma, modelo_nome, gerar_pdf=True, gerar_zip=True, gerar_json=True):
    inicio_total = tempo_exec.time()

    if not os.path.exists(caminho_audio):
        print(f"\n❌ Arquivo não encontrado: {caminho_audio}")
        return

    caminho_audio = converter_video_para_audio(caminho_audio)

    print(f"📡 Realizando diarização de falantes...")
    diarization = pipeline(caminho_audio)
    print(f"✅ Diarização concluída!\n")

    print(f"🎧 Carregando modelo Whisper ({modelo_nome})...")
    modelo = whisper.load_model(modelo_nome, device=DISPOSITIVO)
    print(f"✅ Modelo Whisper pronto para uso!\n")

    segmentos = list(diarization.itertracks(yield_label=True))
    speaker_map, speaker_count = {}, 0
    total = len(segmentos)
    resultado_final, srt_linhas, json_saida = [], [], []

    for idx, (segmento, _, speaker_id) in enumerate(tqdm(segmentos, desc="📥 Transcrevendo segmentos", colour="green")):
        start, end = segmento.start, segmento.end
        if speaker_id not in speaker_map:
            speaker_count += 1
            speaker_map[speaker_id] = f"Speaker {speaker_count}"
        speaker_nome = speaker_map[speaker_id]

        print(f"🔹 Segmento {idx+1:02}/{total} | ⏱ {format_tempo(start)} → {format_tempo(end)} | 🗣️ {speaker_nome}")
        temp_path = extrair_trecho_ffmpeg(caminho_audio, start, end)

        try:
            resultado = modelo.transcribe(temp_path, fp16=True, language=None if idioma == "auto" else idioma)
            texto = resultado["text"].strip()

            if texto:
                resultado_final.append(f"[{format_tempo(start)} - {format_tempo(end)}] {speaker_nome}:\n{texto}\n")
                srt_linhas.append(f"{len(srt_linhas)+1}\n{format_srt(start)} --> {format_srt(end)}\n{speaker_nome}: {texto}\n")
                json_saida.append({"start": float(start), "end": float(end), "speaker": speaker_nome, "text": texto})
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    time.sleep(0.5)
                    os.remove(temp_path)

    nome_base = os.path.splitext(os.path.basename(caminho_audio))[0]
    pasta_saida = os.path.dirname(caminho_audio)

    txt_path = os.path.join(pasta_saida, nome_base + "_diarizacao.txt")
    srt_path = os.path.join(pasta_saida, nome_base + "_diarizacao.srt")
    pdf_path = os.path.join(pasta_saida, nome_base + "_diarizacao.pdf")
    json_path = os.path.join(pasta_saida, nome_base + "_diarizacao.json")
    zip_path = os.path.join(pasta_saida, nome_base + "_diarizacao.zip")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(resultado_final))

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_linhas))

    if gerar_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for linha in resultado_final:
            pdf.multi_cell(0, 10, linha)
        pdf.output(pdf_path)

    if gerar_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_saida, f, ensure_ascii=False, indent=2)

    if gerar_zip:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(txt_path, arcname=os.path.basename(txt_path))
            zipf.write(srt_path, arcname=os.path.basename(srt_path))
            if gerar_pdf:
                zipf.write(pdf_path, arcname=os.path.basename(pdf_path))
            if gerar_json:
                zipf.write(json_path, arcname=os.path.basename(json_path))

        print(f"\n📦 Arquivos gerados e salvos como: {Fore.GREEN}{zip_path}{Style.RESET_ALL}")

    fim_total = tempo_exec.time()
    print(f"⏱️  Tempo total: {timedelta(seconds=int(fim_total - inicio_total))}\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"\n⚠️  Uso correto:")
        print("python transcrever.py <arquivo.mp3> <idioma: en|pt|es|auto> <modelo: tiny|base|small|medium|large> [--no-pdf] [--no-json] [--no-zip]\n")
        sys.exit(1)

    caminho = sys.argv[1]
    idioma_input = sys.argv[2].lower()
    modelo_input = sys.argv[3].lower()

    gerar_pdf = "--no-pdf" not in sys.argv
    gerar_json = "--no-json" not in sys.argv
    gerar_zip = "--no-zip" not in sys.argv

    idiomas_suportados = ["en", "pt", "es", "auto"]
    modelos_suportados = ["tiny", "base", "small", "medium", "large"]

    if idioma_input not in idiomas_suportados:
        print(f"❌ Idioma inválido. Use: en, pt, es ou auto.")
        sys.exit(1)

    if modelo_input not in modelos_suportados:
        print(f"❌ Modelo inválido. Use: tiny, base, small, medium ou large.")
        sys.exit(1)

    transcrever_com_diarizacao(caminho, idioma_input, modelo_input, gerar_pdf, gerar_zip, gerar_json)
