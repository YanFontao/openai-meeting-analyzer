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
from tqdm import tqdm
from argparse import ArgumentParser
from openai import OpenAI, OpenAIError

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
    if caminho.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')):
        wav_temp = caminho + "_temp.wav"
        subprocess.run(["ffmpeg", "-y", "-i", caminho, "-ar", "16000", "-ac", "1", wav_temp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_temp
    return caminho

def enviar_para_openai(texto):
    from openai import OpenAI

    client = OpenAI(api_key="INSERTAPIKEY")

    prompt = (
        "Faça uma interpretação geral e organizada desta transcrição de conversa, destacando os principais pontos, decisões, ideias e qualquer observação importante que possa ser útil para revisitar a conversa, incluindo também o que o cliente está pedindo, requisitos principais do projeto, objetivos mencionados, referencias mencionadas, escopo desejado, sugestões ou ideias que surgiram, qualquer dúvida ou definição pendente. Tenha em mente consideração o contexto de cliente e profissional de design ou produto conversando."
    )
    resposta = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente de análise de conversas para contextos de briefing e alinhamentos de projeto."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": texto}
        ]
    )
    return resposta.choices[0].message.content.strip()

def transcrever_com_diarizacao(caminho_audio, idioma, modelo_nome, gerar_pdf=True, gerar_zip=True, gerar_json=True, usar_gpt=True):
    inicio_total = time.time()

    caminho_convertido = converter_video_para_audio(caminho_audio)

    if not os.path.exists(caminho_convertido):
        print(f"\n❌ Arquivo não encontrado: {caminho_audio}")
        return

    print(f"📡 Realizando diarização de falantes...")
    diarization = pipeline(caminho_convertido)
    print(f"✅ Diarização concluída!\n")

    print(f"🎧 Carregando modelo Whisper ({modelo_nome})...")
    modelo = whisper.load_model(modelo_nome, device=DISPOSITIVO)
    print(f"✅ Modelo Whisper pronto para uso!\n")

    segmentos = list(diarization.itertracks(yield_label=True))
    speaker_map, speaker_count = {}, 0
    resultado_final, srt_linhas, json_saida = [], [], []

    for idx, (segmento, _, speaker_id) in enumerate(tqdm(segmentos, desc="📥 Transcrevendo segmentos", colour="green")):
        start, end = segmento.start, segmento.end
        if speaker_id not in speaker_map:
            speaker_count += 1
            speaker_map[speaker_id] = f"Speaker {speaker_count}"
        speaker_nome = speaker_map[speaker_id]

        print(f"🔹 Segmento {idx+1:02}/{len(segmentos)} | ⏱ {format_tempo(start)} → {format_tempo(end)} | 🗣️ {speaker_nome}")
        temp_path = extrair_trecho_ffmpeg(caminho_convertido, start, end)

        try:
            resultado = modelo.transcribe(temp_path, fp16=True, language=None if idioma == "auto" else idioma)
            texto = resultado["text"].strip()

            if texto:
                resultado_final.append(f"[{format_tempo(start)} - {format_tempo(end)}] {speaker_nome}:\n{texto}\n")
                srt_linhas.append(f"{len(srt_linhas)+1}\n{format_srt(start)} --> {format_srt(end)}\n{speaker_nome}: {texto}\n")
                json_saida.append({"start": float(start), "end": float(end), "speaker": speaker_nome, "text": texto})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    nome_base = os.path.splitext(os.path.basename(caminho_audio))[0]
    pasta_saida = os.path.dirname(caminho_audio)

    txt_path = os.path.join(pasta_saida, nome_base + "_transcricao.txt")
    srt_path = os.path.join(pasta_saida, nome_base + "_diarizacao.srt")
    pdf_path = os.path.join(pasta_saida, nome_base + "_diarizacao.pdf")
    json_path = os.path.join(pasta_saida, nome_base + "_diarizacao.json")
    zip_path = os.path.join(pasta_saida, nome_base + "_diarizacao.zip")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(resultado_final))

    if gerar_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_saida, f, ensure_ascii=False, indent=2)

    if gerar_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", fname="fonts/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
        for linha in resultado_final:
            pdf.multi_cell(0, 10, linha)
        pdf.output(pdf_path)

    if gerar_zip:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(txt_path, arcname=os.path.basename(txt_path))
            if gerar_pdf:
                zipf.write(pdf_path, arcname=os.path.basename(pdf_path))
            if gerar_json:
                zipf.write(json_path, arcname=os.path.basename(json_path))
        os.remove(txt_path)
        if gerar_pdf: os.remove(pdf_path)
        if gerar_json: os.remove(json_path)
        print(f"\n📦 Arquivos gerados e salvos como: {Fore.GREEN}{zip_path}{Style.RESET_ALL}")
    else:
        print(f"\n📄 Arquivo gerado: {Fore.GREEN}{txt_path}{Style.RESET_ALL}")

    if usar_gpt:
        print("\n🧠 Enviando transcrição para o ChatGPT...")
        try:
            analise = enviar_para_openai("\n".join(resultado_final))
            analise_path = os.path.join(pasta_saida, nome_base + "_analise.txt")
            with open(analise_path, "w", encoding="utf-8") as f:
                f.write(analise)
            print(f"✅ Análise salva em: {Fore.GREEN}{analise_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"❌ Erro ao enviar para o ChatGPT: {e}")

    if caminho_convertido.endswith("_temp.wav") and os.path.exists(caminho_convertido):
        os.remove(caminho_convertido)

    fim_total = time.time()
    print(f"⏱️  Tempo total: {timedelta(seconds=int(fim_total - inicio_total))}\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Transcrição com diarização e análise opcional com GPT")
    parser.add_argument("arquivos", nargs="+", help="Caminhos dos arquivos de áudio ou vídeo")
    parser.add_argument("idioma", choices=["en", "pt", "es", "auto"], help="Idioma da transcrição")
    parser.add_argument("modelo", choices=["tiny", "base", "small", "medium", "large"], help="Modelo Whisper")
    parser.add_argument("--no-pdf", action="store_true", help="Não gerar PDF")
    parser.add_argument("--no-json", action="store_true", help="Não gerar JSON")
    parser.add_argument("--no-zip", action="store_true", help="Não gerar ZIP")
    parser.add_argument("--no-gpt", action="store_true", help="Não enviar para GPT")
    args = parser.parse_args()

    for caminho in args.arquivos:
        transcrever_com_diarizacao(
            caminho,
            args.idioma,
            args.modelo,
            gerar_pdf=not args.no_pdf,
            gerar_zip=not args.no_zip,
            gerar_json=not args.no_json,
            usar_gpt=not args.no_gpt
        )
