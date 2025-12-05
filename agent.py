import time
import os
import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.retrievers import EnsembleRetriever
from croma_db_update import update_db_with_feedback
from hybrid_reranking_rag import reranked_rag_query,create_reranked_rag_chain
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import re
import os
import subprocess
import whisper
import pandas as pd
import json
from PIL import Image
import base64
import io


CLOUDFLARE_TUNNEL_URL = ".../"
OLLAMA_MODEL_ID = "gemma3:27b"
WEATHER_API = "..."


def get_question():
    print("Getting question...")
    API_URL = "https://agents-course-unit4-scoring.hf.space/random-question"
    response = requests.get(API_URL).json()

    question = response.get("question")
    return question, response


class CustomError(Exception):
    pass


@tool
def WeatherInfoTool(location: str) -> str:
    """Fetches weather information for a given location."""
    weather_start = time.time()
    url = f"https://api.weatherstack.com/current?access_key={WEATHER_API}"
    querystring = {"query": location}
    response = requests.get(url, params=querystring)
    data = response.json()
    city = data["location"]["name"]
    country = data["location"]["country"]
    temperature = data["current"]["temperature"]
    weather_description = data["current"]["weather_descriptions"][0]
    weather_stop = time.time()
    weather_time = weather_stop - weather_start
    print(f"⏱️ Çalışma Süresi (weather tool cevap süresi): {weather_time:.2f} saniye")
    return f"Weather in {location}: {weather_description}, {str(temperature)}°C"


search_tool = DuckDuckGoSearchRun(
    name="general_web_search",
    description="Used for general web searches, including current news, web information, and broad, up-to-date topics.",
)
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=3,  # Kaç sonuç dönmesini istediğinizi ayarlayabilirsiniz
    doc_content_chars_max=4000,  # Geri dönen metnin maksimum karakter sayısı
)
wiki_search_tool = WikipediaQueryRun(
    name="wikipedia_search",
    description="Used for searching encyclopedic information such as history, biographies, and general definitions from Wikipedia.",
    api_wrapper=wikipedia_wrapper,  # <-- KRİTİK EKLENTİ
)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
archive_search_tool = ArxivQueryRun(
    name="academic_search",
    description="Used for searching academic articles, theses, and scientific research...",
    api_wrapper=arxiv_wrapper,  # <-- EKLENTİ
)
python_repl_tool = PythonREPLTool()

"""calendar_creator = CalendarToolkit()

@tool
def create_calendar_event(summary: str, start_time: str, end_time: str, description: str) -> str:
    
    #Creates a new event in Google Calendar based on the user's request. 
    #The summary (title) and time information (start_time and end_time) are mandatory.
    
    # Altındaki gerçek Google API çağrısını çalıştır (This line remains the internal execution logic)
    return calendar_creator.run({
        "summary": summary,
        "start_time": start_time,
        "end_time": end_time,
        "description": description
    })
"""


@tool
def transcribe_audio_whisper(audio_path: str) -> str:
    """Extracts transcript from any voice file using Whisper"""
    print("Using audio transcriber tool...")
    model = whisper.load_model("small")  # 'tiny', 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_path)
    return result["text"]


def download_audio_from_youtube(url, output_path="audio.mp3"):

    subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bestaudio",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "-o",
            output_path,
            url,
        ]
    )


def download_video_from_youtube(url, output_path="video.mp4"):
    print("Using yt video download tool...")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo+bestaudio", "-o", output_path, url],
        capture_output=True,
        text=True,
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")


@tool
def youtube_transcript_func(url: str) -> str:
    "Extracts transcript from a YouTube video using Whisper"
    print("Using yt audio download tool...")
    if url.startswith("url="):
        url = url.split("url=")[1].strip()

    url = url.strip("'").strip('"')  # <-- YENİ EKLENTİ
    audio_path = "audio.mp3"
    download_audio_from_youtube(url, audio_path)
    transcript = transcribe_audio_whisper.run(audio_path)
    os.remove(audio_path)
    return transcript


@tool
def caption_image_func(raw_input: str) -> str:
    """
    Analyzes a local image file. The input MUST be a single string containing
    both 'image_path' and 'prompt' as key-value pairs (e.g., image_path='path', prompt='question').
    """
    print("Using image caption tool for manual parsing...")

    args = {}

    # 1. Ham girdiyi önce temizle (Baştaki ve sondaki tırnakları/boşlukları sil)
    raw_input = (
        raw_input.strip().strip("'").strip('"')
    )  # Bu, Pydantic'ten gelen dizeyi temizler

    # 2. Key-Value çiftlerini Regex ile güvenli bir şekilde ayırma
    # Bu desen, 'key=value' formatını bulur ve değerin içindeki tırnakları görmezden gelir.
    # Bu yöntem, virgül, tırnak ve boşluk sorunlarını büyük ölçüde çözer.
    matches = re.findall(r"(\w+)\s*=\s*([^\,]+)", raw_input)

    for match in matches:
        key = match[0].strip()
        value = match[1].strip().strip("'").strip('"')  # Tırnakları temizle
        args[key] = value

    try:
        image_path = args.get("image_path")
        prompt = args.get("prompt")

        if not image_path or not prompt:
            # Hata varsa, ajana neyi düzeltmesi gerektiğini söyleyin.
            return f"Error: Parsing failed. Missing 'image_path' or 'prompt'. Parsed args: {args}. Make sure the format is 'image_path=..., prompt=...'"

    except Exception as e:
        return f"Error during manual parsing: {e}. Raw input: {raw_input}"

    print("Using image caption tool...")
    global CLOUDFLARE_TUNNEL_URL
    global OLLAMA_MODEL_ID

    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = CLOUDFLARE_TUNNEL_URL + "/api/generate"

    payload = {
        "model": OLLAMA_MODEL_ID,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    }
    response = requests.post(
        url, headers={"Content-Type": "application/json"}, json=payload
    )
    response.raise_for_status()  # Hatalı HTTP durum kodu varsa Exception atar

    data = response.json()

    if "response" in data:
        print(data["response"])
        return data["response"]
    else:
        print("Image not recognized")
        return "Image not recognized"




@tool
def download_video_from_youtube(url: str, output_path="video.mp4") -> str:
    """
    Downloads a YouTube video from the provided URL using yt-dlp and saves it locally.

    Args:
        url (str): The full URL of the YouTube video to download (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').

    Returns:
        str: A message indicating success with the file path, or an error message if the download fails.
    """
    print("Using yt video download tool...")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]",
            "--merge-output-format",
            "mp4",  # video+ses -> mp4
            "-o",
            output_path,
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    return output_path



RAG_STATE = {
    "llm": None,
    "prompt": None,
    "ensemble_retriever": None,
    "is_initialized": False
}

def build_rag():
    """
    Bu fonksiyon RAG sistemini SADECE BİR KERE kurar.
    İkinci çağrışta 'Zaten kurulu' diyip geçer.
    """
    # Kontrol: Zaten yüklü mü?
    if RAG_STATE["is_initialized"]:
        return

    print("⏳ RAG Sistemi yükleniyor... (Bu işlem sadece bir kez yapılır)")

    # --- Senin yazdığın build_rag mantığı buraya giriyor ---
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 8000
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # Veritabanını ve BM25'i hazırla
    vectorstore, bm25_retriever = update_db_with_feedback(
        "./database", client, "rag_test_data", bm_ret=True
    )
    
    if bm25_retriever == False:
        print("HATA: BM25 Retriever oluşturulamadı!")
        return

    # LLM ve Prompt
    llm_model, rag_prompt = create_reranked_rag_chain(vectorstore)
    
    # Retriever'ları birleştir
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], 
        weights=[0.5, 0.5]
    )
    
    # --- SONUÇLARI GLOBAL DEPOYA KAYDET ---
    RAG_STATE["llm"] = llm_model
    RAG_STATE["prompt"] = rag_prompt
    RAG_STATE["ensemble_retriever"] = ensemble_retriever
    RAG_STATE["is_initialized"] = True
    
    print("✅ RAG Sistemi hazır ve belleğe alındı!")


@tool
def rag_tool(question:str)->str:
    """
    Use this tool to answer factual questions about general knowledge, history, science, geography, famous people, and specific events.
    The database contains a vast encyclopedic collection (based on Wikipedia) covering diverse topics like the Normans, Physics, Sports events (e.g., Super Bowl), and more.
    Capable of answering in both Turkish and English.
    Always use this tool when the user asks for specific information, definitions, or historical details.
    Do NOT use for personal questions like 'How are you?'.
    """
    print("Using Rag tool")
    try:
        # Eğer Agent çalıştı ama RAG henüz yüklenmediyse, burada yükle
        if not RAG_STATE["is_initialized"]:
            build_rag()
            
        # Global depodan hazır malzemeleri çek
        llm = RAG_STATE["llm"]
        prompt = RAG_STATE["prompt"]
        retriever = RAG_STATE["ensemble_retriever"]
    
        response,elapsed_time=reranked_rag_query(
        llm,
        prompt,
        retriever,
        query=question,
        )
        return response

    except Exception as e:
        return f"RAG Hatası: {str(e)}"



def build_agent():
    global CLOUDFLARE_TUNNEL_URL
    global OLLAMA_MODEL_ID
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("system_prompt.txt dosyası bulunamadı, varsayılan prompt kullanılacak.")
        system_prompt = """You are a helpful assistant tasked with answering questions using a set of tools. Never return your answer in dictionary, list, or JSON format. Your output must be a single string, integer, or float. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. 
        [ARGUMENT FORMATTING RULE]: When calling any tool that requires multiple arguments (e.g., 'a' and 'b' for math functions, or 'image_path' and 'prompt' for image tools), you MUST use a clean Python-like key-value format separated by commas (e.g., 'a=174.5, b=93.2'). DO NOT include quotation marks around numbers.
        [ACTION INPUT FORMATTING RULES] 
        1. **STRICT KEY-VALUE FORMAT**: When using any tool that requires arguments (e.g., 'image_path', 'prompt', 'location'), the Action Input MUST be a clean, comma-separated list of key-value pairs, strictly adhering to Python dictionary syntax. 
        2. **MANDATORY QUOTES**: All string values (paths, URLs, names) MUST be enclosed in single or double quotes (e.g., 'Ankara', "https://..."). 
        3. **NO EXTRA TEXT**: Do not include any text, reasoning, or explanation before or after the Action Input. 
        4. **EXAMPLE FOR MULTI-ARGUMENT TOOL**: 
            - CORRECT: Action Input: image_path='/home/user/file.png', prompt='What is the object?' 
            - WRONG: Action Input: I will use the tool on image_path: '/home/user/file.png'"""
        # ^^^ Dizeyi kapatan üç tırnak işareti (son satırda)
    except Exception as e:
        print(f"system_prompt.txt okunurken hata oluştu: {e}")
        system_prompt = """You are a helpful assistant tasked with answering questions using a set of tools. Never return your answer in dictionary, list, or JSON format. Your output must be a single string, integer, or float. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. 
        [ARGUMENT FORMATTING RULE]: When calling any tool that requires multiple arguments (e.g., 'a' and 'b' for math functions, or 'image_path' and 'prompt' for image tools), you MUST use a clean Python-like key-value format separated by commas (e.g., 'a=174.5, b=93.2'). DO NOT include quotation marks around numbers.
        [ACTION INPUT FORMATTING RULES] 
        1. **STRICT KEY-VALUE FORMAT**: When using any tool that requires arguments (e.g., 'image_path', 'prompt', 'location'), the Action Input MUST be a clean, comma-separated list of key-value pairs, strictly adhering to Python dictionary syntax. 
        2. **MANDATORY QUOTES**: All string values (paths, URLs, names) MUST be enclosed in single or double quotes (e.g., 'Ankara', "https://..."). 
        3. **NO EXTRA TEXT**: Do not include any text, reasoning, or explanation before or after the Action Input. 
        4. **EXAMPLE FOR MULTI-ARGUMENT TOOL**: 
            - CORRECT: Action Input: image_path='/home/user/file.png', prompt='What is the object?' 
            - WRONG: Action Input: I will use the tool on image_path: '/home/user/file.png'"""
        # ^^^ Dizeyi kapatan üç tırnak işareti (son satırda)
    base_prompt = hub.pull("hwchase17/react")

    ### Building rag
    print("Building Rag...")
    build_rag()
    ###

    # 2. ReAct prompt'unun sistem mesajını, kendi sistem dizenizle değiştirin

    final_prompt = base_prompt.partial(
        system_message=system_prompt  # Kendi sistem açıklamanızı ReAct şablonuna yerleştirin
    )

    # VEYA, daha basit bir ReAct şablonu oluşturun:

    model = ChatOllama(
        model=OLLAMA_MODEL_ID,
        temperature=0,
        base_url=CLOUDFLARE_TUNNEL_URL,
    )

    tools = [
        WeatherInfoTool,
        transcribe_audio_whisper,
        youtube_transcript_func,
        caption_image_func,
        rag_tool,
        search_tool,
        wiki_search_tool,
        archive_search_tool,
        python_repl_tool,
    ]
    agent = create_react_agent(model, tools, final_prompt)
    Arxivangelist = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,  # Model format hatası yaparsa düzeltmeye çalışır
    )
    return Arxivangelist


if __name__ == "__main__":
    # tool_test()
    Arxivangelist = build_agent()
    start_time = time.time()
    question, response = get_question()
    try:
        answer = Arxivangelist.invoke(
            {"input": "Şu an konyada hava durumu nedir?", "chat_history": []}
        )
    except Exception as e:
        print(f"Hata: {e}")
    duration = time.time() - start_time
    print(f"   ✅ BAŞARILI ({duration:.2f}sn)")
    print(f"   🤖 Cevap: {str(answer)}")
    print(answer)
