from agent import *

TEST_CASES = [
    (
        "Benim için './cca530fc-4052-43b2-b130-b30968d8aa44.png' konumundaki görseli incele. Sıranın siyah taşlarda olduğunu düşünerek olası en iyi sonraki hamleyi verirmisin? ",
        "caption_image_func",
    ),
    (
        "İstanbul'daki hava kirliliği son durumu nedir?,İnternette araştırıp bulduğun sonuçları kısaca değerlendirirmisin",
        "general_web_search",
    ),
    (
        "En son 2024'te yayınlanan biyolojik yapay zeka makaleleri hakkında bilgi ver.",
        "academic_search",
    ),
    (
        "Avrupa Birliği'nin kurucusu kimdir ve kaç yılında kurulmuştur? Bu konudaki Wikipedia bilgilerini özetlermisin",
        "wikipedia_search",
    ),
    ("Tokyo'da şu an hava nasıl?", "WeatherInfoTool"),
    (
        "Büyük bir matematik sorusu: 174.5 ile 93.2 sayılarının çarpımı kaçtır?",
        "multiply_func",
    ),
    (
        "Büyük bir hesaplama sorusu: 5000 sayısını 125'e bölüp, sonucu 17 ile topla.",
        "div_func",
    ),
    (
        "Şu Python kodunu çalıştır ve sonucu söyle: print(len(['a', 'b', 'c'] * 5))",
        "python_repl_tool",
    ),
    (
        "Bu YouTube URL'sindeki (https://www.youtube.com/watch?v=dQw4w9WgXcQ) videosunun transcriptini çıkar ve metni kısaca yorumla.",
        "youtube_transcript_func",
    ),
    (
        "Sürtünme gibi muhafazakar olmayan kuvvetler, neden aslında mikroskobik potansiyellerin sonuçları olarak kabul edilir?",
        "rag tool",
    )
]


def tool_test_loop(agent_executor, test_cases):
    """Verilen test senaryolarını AgentExecutor üzerinde çalıştırır ve süreleri ölçer."""
    print("=" * 60)
    print("🤖 TOOL FONKSİYONEL VE PERFORMANS TESTİ BAŞLIYOR ⏱️")
    print("=" * 60)

    for i, (question, expected_tool) in enumerate(test_cases):
        print(f"\n--- TEST {i+1}/{len(test_cases)} ---")
        print(f"❓ SORU: {question}")
        print(f"🎯 BEKLENEN TOOL: {expected_tool}")

        start_time = time.time()

        try:
            # AgentExecutor'ı çağırıyoruz. chat_history boş bir liste olarak gönderilmeli.
            result = agent_executor.invoke({"input": question, "chat_history": []})

            duration = time.time() - start_time

            # Agent'ın yanıtını ve süresini yazdır
            print(f"   ✅ BAŞARILI. Süre: {duration:.2f} saniye")
            # ReAct çıktısını temizlemek için sadece FINAL ANSWER'ı yazdırma
            print(f"   🤖 Cevap Özeti: {result.get('output', 'Yanıt Bulunamadı')}...")

        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ HATA! Süre: {duration:.2f} saniye")
            print(f"   Hata Detayı: {type(e).__name__}: {str(e)[:100]}...")

    print("=" * 60)
    print("TESTLER TAMAMLANDI. HATA ALAN TOLLARI KONTROL EDİN.")
    print("=" * 60)


# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":

    Arxivangelist = build_agent()
    """    raw_input_string = (
        f"image_path='{os.path.abspath('/home/mbaloglu/langchain_llm/cca530fc-4052-43b2-b130-b30968d8aa44.png')}', "
        f"prompt='What is the best move in this chess position?'"
    )

    transcript = caption_image_func.invoke({
        "raw_input": raw_input_string # <-- KRİTİK: Anahtar adı artık 'raw_input'
    })
    print(f"Transcript (invoke): {transcript}")"""

    # Testi başlat
    tool_test_loop(Arxivangelist, TEST_CASES)
