from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
import chromadb
from langchain_classic.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_classic.chains.llm import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from croma_db_update import update_db_with_feedback
import time
import os


CLOUDFLARE_TUNNEL_URL = ".../"
OLLAMA_MODEL_ID = "gemma3:27b"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2"
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_test_data"
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# 1. Veritabanını Güncelle ve VectorStore'u Al
print("--- 1. Veritabanı Güncelleme ---")

TOP_K_RETRIEVAL = 20  # ChromaDB'den çekilecek toplam parça sayısı
TOP_K_RERANK = 5  # LLM'e gönderilecek nihai parça sayısı


def get_hybrid_reranked_docs(ensemble_retriever, query):
    """ChromaDB'den çok sayıda parça çeker ve bunları Cross-Encoder ile yeniden sıralar."""

    # 3. İlk Geniş Havuz (Keyword + Vector sonuçlarının karışımı)

    first_pass_docs = ensemble_retriever.invoke(query)

    # 2. Yeniden Sıralama için Veri Hazırlama

    # Cross-Encoder, (sorgu, metin) çiftleri listesi ister.

    sentences = [(query, doc.page_content) for doc in first_pass_docs]

    # 3. Puanlama (Score)

    # Model, her parça için 0-1 arasında bir alaka puanı hesaplar.

    scores = reranker.predict(sentences)

    # 4. Parçaları Puanlarla Birleştirme ve Sıralama

    doc_scores = sorted(zip(first_pass_docs, scores), key=lambda x: x[1], reverse=True)

    # 5. En iyi (TOP_K_RERANK) parçayı seçme

    final_docs = [doc_score[0] for doc_score in doc_scores[:TOP_K_RERANK]]

    return final_docs


def create_reranked_rag_chain(vectorstore: Chroma):
    """
    Özelleştirilmiş Yeniden Sıralama (Re-ranking) mantığını kullanarak
    RAG zincirini oluşturur.
    """
    llm = OllamaLLM(model=OLLAMA_MODEL_ID, base_url=CLOUDFLARE_TUNNEL_URL)

    template = """
    You are an expert Question-Answering system. Your sole instruction is to generate a final answer 
    BASED ONLY on the information provided in the CONTEXT below. The CONTEXT is your only source of truth.

    The CONTEXT contains snippets, each starting with a source tag like [KAYNAK: filename].
    When answering, you MUST cite the source filename for the information you use.
    
    Example format: "According to [KAYNAK: document.pdf], the sky is blue." or "The sky is blue (Source: document.pdf)."

    You MUST answer the question in the SAME LANGUAGE as the question was asked. 
    (E.g., if the question is in Turkish, the answer MUST be in Turkish.)

    If the CONTEXT is insufficient to fully answer the question, you MUST NOT add external information. 
    Instead, summarize what IS available and mention the source.

    CONTEXT:
    {context}

    QUESTION: {question}

    Answer:
    """
    RAG_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # RetrievalQA zincirini, LLM'e doğrudan Parçaları besleyen 'stuff' tipiyle oluşturuyoruz.
    # Ancak retriever'ı, yeniden sıralama mantığımızı uygulayacak özel bir fonksiyondan almalıyız.
    # LangChain'in RetrievalQA'sı esnek olmadığı için, sorgu mantığını manuel olarak uygulamamız en temiz yol olacaktır.

    # Bu adımı basitleştirmek adına, ana çalışma bloğunda manuel Retrieval/Reranking yapıp,
    # ardından LLM'i çağıracak bir fonksiyon oluşturalım.

    # RetrievalQA yerine, sadece LLM'in kendisini döndürelim ve ana döngüde manuel olarak kullanalım:
    return llm, RAG_PROMPT


def reranked_rag_query(
    llm,
    prompt,
    ensemble_retriever,
    query: str,
):
    """Yeniden sıralama mantığını uygulayarak sorguyu çalıştırır ve sonuçları yazdırır."""
    start_time = time.time()
    print(f"\nSorgu: {query}")
    print("-" * 50)

    # 1. Yeniden Sıralamayı Uygula (Retrieval + Reranking)
    final_docs = get_hybrid_reranked_docs(ensemble_retriever, query=query)

    # 2. Bağlamı Hazırla
    context_list = []
    for doc in final_docs:
        # Dosya yolunu temizle (örn: /home/user/data/train.json -> train.json)
        full_path = doc.metadata.get("source", "Unknown source")
        filename = os.path.basename(full_path)

        # Başlık varsa ekle (Opsiyonel)
        title = doc.metadata.get("title", "")

        # Formatı Oluştur:
        # [Kaynak: dosya_adı | Başlık]
        # İçerik...
        formatted_chunk = f"[Source: {filename} | {title}]\n{doc.page_content}"
        context_list.append(formatted_chunk)

    # Listeyi birleştir
    context = "\n\n---\n\n".join(context_list)

    # 3. LLM Chain'i Oluştur ve Çalıştır
    LLMChain(prompt=prompt, llm=llm)

    # Final Prompt'u oluştur
    final_prompt = prompt.format(context=context, question=query)

    # Cevabı al
    response = llm.invoke(final_prompt)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    # Cevabı Yazdırma
    return response,elapsed_time



if __name__ == "__main__":
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedding_function = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    # 3. Var olan koleksiyona bağlanın (Yükleme yapmadan)
    vectorstore, bm25_retriever = update_db_with_feedback(
        "./database/", client, "rag_test_data", bm_ret=True
    )
    if bm25_retriever == False:
        print("HATA!!!!!")

    print("Mevcut veritabanı koleksiyonuna başarıyla bağlanıldı.")
    # 2. LLM ve Prompt'u Al
    print("\n--- 2. RAG Bileşenleri Kurulumu ---")
    llm_model, rag_prompt = create_reranked_rag_chain(vectorstore)
    print("Yeniden Sıralama (Re-ranking) Modeli ve LLM hazır.")

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )

    # 1. Türkçe Factual Sorgu (Bu zaten Türkçe kaldı)
    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="Rollo'nun Vikinglerinin torunları hangi dili ve dini benimsedi?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")

    # 2. İNGİLİZCE Multi-Hop Sorgu (KUvvet Birimleri)
    # (Önceki: The Newton'dan daha az kullanılan metrik terim nedir ve bazen ne olarak adlandırılır?)
    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="What is the metric term less used than the Newton, and what is it sometimes referred to?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")

    # 3. İNGİLİZCE Factual/Detay Sorgu (Notre Dame)
    # (Önceki: University of Notre Dame'deki Grotto'nun bir kopyası olduğu, Virgin Mary'nin 1858'de göründüğü iddia edilen yer neresidir?)
    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="What is the location of the grotto that the University of Notre Dame's grotto is a replica of, where the Virgin Mary allegedly appeared in 1858?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")
    # 4. Türkçe Çıkarım/Neden Sorgusu (Bu zaten Türkçe kaldı)
    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="Normanların eski İskandinav dinini ve dilini bırakıp, yerel halkın dinini ve dilini benimsemesindeki temel kültürel adaptasyon süreci nasıldı?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")

    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="Sürtünme gibi muhafazakar olmayan kuvvetler, neden aslında mikroskobik potansiyellerin sonuçları olarak kabul edilir?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")    
    # 5. İNGİLİZCE Multi-Hop/Detay Sorgu (Super Bowl)
    # (Önceki: Super Bowl 50'de, geleneksel Romen rakamları (L) neden kullanılmamıştır?)
    response,elapsed_time=reranked_rag_query(
        llm_model,
        rag_prompt,
        ensemble_retriever,
        query="Why were the traditional Roman numerals (L) not used for Super Bowl 50?",
    )
    print(f"\n✅ Gemma Cevabı: {response}")
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")
