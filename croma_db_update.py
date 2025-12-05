import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.errors import NotFoundError
from langchain_community.retrievers import BM25Retriever
import hashlib
import os
import json

# --- Ayarlar ---
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000  # Sadece port numarası
COLLECTION_NAME = "rag_test_data"
FILE_PATH = "./database/"  # Kendi dosyanızın adı
MAX_BATCH_SIZE = 5000


def db_update():

    # 4. ChromaDB İstemcisi Oluşturma ve Bağlanma
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(f"Eski koleksiyon ({COLLECTION_NAME}) siliniyor...")
    try:
        # Silme işlemini dene
        client.delete_collection(COLLECTION_NAME)
        print("Silme başarılı.")
    except NotFoundError:
        # Eğer koleksiyon yoksa (ilk çalıştırma), hatayı yoksay ve devam et
        print("Koleksiyon zaten mevcut değil. Yeni koleksiyon oluşturulacak.")
    # 5. Chroma Veri Deposu Oluşturma (Bağlantılı)
    # Vektörler bu koleksiyona yüklenir.
    vectorstore, bm25_retriever = update_db_with_feedback(
        FILE_PATH, client=client, collection_name=COLLECTION_NAME, bm_ret=False
    )

    print(
        f"Vektörler başarıyla Docker sunucusundaki '{COLLECTION_NAME}' koleksiyonuna yüklendi."
    )
    # 5. Sorgu Hazırlama (Retrieval)
    return vectorstore


def create_id(chunk):
    """Metin içeriği, kaynak dosyası ve metin sırasını kullanarak benzersiz ID oluşturur."""
    # Kaynak dosyası (SQuAD verisinde: dosya adı + başlık)
    source = chunk.metadata.get("source", "") + chunk.metadata.get("title", "")

    # Metin içeriği
    content = chunk.page_content

    # Metin + Kaynak bilgisini birleştirip hash alıyoruz
    unique_string = f"{source}_{content}"

    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


def get_chunks_with_ids(
    file_dir_path: str, chunk_size: int = 500, chunk_overlap: int = 50
):
    """Veri dizinindeki tüm JSON dosyalarını okur, SQuAD formatını çözer ve parçalara ayırır."""

    all_documents = []

    for filename in os.listdir(file_dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(file_dir_path, filename)
            print(f"-> {filename} dosyası okunuyor...")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # SQuAD JSON yapısını gezme
            # Yapı: data -> paragraphs -> context
            for item in data.get("data", []):
                for paragraph_data in item.get("paragraphs", []):
                    # Her bir paragrafı (context), tüm SQuAD veri setinin ana bağlamını temsil eden
                    # tek bir belge olarak alıyoruz.
                    context_text = paragraph_data.get("context")
                    title = item.get("title", "Bilinmeyen Başlık")

                    if context_text:
                        # LangChain Document nesnesi oluşturma
                        all_documents.append(
                            Document(
                                page_content=context_text,
                                metadata={"source": file_path, "title": title},
                            )
                        )

    # 2. Bölme (Chunking) - Artık parçalama işlemini büyük paragraflar üzerinde yapıyoruz
    print(
        f"Toplam {len(all_documents)} paragraf/belge hazırlandı. Parçalara ayrılıyor..."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    all_chunks = text_splitter.split_documents(all_documents)

    # 3. ID Oluşturma ve Ayırma
    ids = []
    for i, chunk in enumerate(all_chunks):  # enumerate ile sırasını (i) alıyoruz
        # Metin içeriğine sadece i (index) ekleyerek benzersizliği garanti ediyoruz
        unique_string = f"{chunk.page_content}_{i}"
        chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

        ids.append(chunk_id)
        chunk.metadata["ids"] = chunk_id

    return all_chunks, ids


def update_db_with_feedback(
    file_dir_path: str, client: chromadb.HttpClient, collection_name: str, bm_ret=False
):
    """
    Veritabanını günceller, mevcut parçaları kontrol eder ve geri bildirim sağlar.
    (Mevcut parçalar overwrite edilir, silinmez).
    """

    # Veriyi hazırlama
    all_chunks, ids = get_chunks_with_ids(file_dir_path)
    print(f"Hazırlanan toplam parça sayısı: {len(all_chunks)}")

    embedding_function = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    # Vektör deposunu al (Eğer koleksiyon yoksa oluşturur)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    # ------------------ DURUM KONTROLÜ VE YÜKLEME ------------------

    # ChromaDB'ye parça parça ekleme/güncelleme (Daha fazla kontrol için)
    # LangChain'in add_documents'u arka planda ID'ye göre güncelleme (overwrite) yapar.

    # Mevcut tüm ID'leri çekelim (Sadece durum kontrolü için)
    try:
        existing_ids = client.get_collection(collection_name).get(include=[])["ids"]
        existing_ids_set = set(existing_ids)
    except Exception:
        # Koleksiyon ilk kez oluşturuluyorsa
        existing_ids_set = set()

    chunks_to_add = []
    ids_to_add = []

    # Sayaçlar
    new_count = 0
    skipped_count = 0  # Güncelleme yerine "Atlanan" sayacı

    for i, chunk_id in enumerate(ids):
        if chunk_id in existing_ids_set:
            # HATANIN KAYNAĞI BURASIYDI:
            # Eskiden buraya ekliyordun, şimdi sadece sayacı artırıp geçiyoruz.
            skipped_count += 1
            continue  # Listeye eklemeden bir sonraki döngüye geç
        else:
            # ID yok, bu gerçekten yeni bir veri
            new_count += 1
            chunks_to_add.append(all_chunks[i])
            ids_to_add.append(chunk_id)

    # Yükleme Kısmı
    # Eğer eklenecek yeni parça varsa batch işlemine gir
    if chunks_to_add:
        print(f"🚀 {len(chunks_to_add)} yeni parça tespit edildi, yükleniyor...")

        # ... Batch döngüsü (senin kodunla aynı) ...
        num_chunks = len(chunks_to_add)
        for i in range(0, num_chunks, MAX_BATCH_SIZE):
            end_index = min(i + MAX_BATCH_SIZE, num_chunks)
            batch_chunks = chunks_to_add[i:end_index]
            batch_ids = ids_to_add[i:end_index]

            print(
                f"   -> Batch {int(i/MAX_BATCH_SIZE) + 1}: {len(batch_chunks)} parça yükleniyor..."
            )
            try:
                vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
            except Exception as e:
                print(f"!!! HATA: {e}")
                break
    else:
        print("✨ Eklenecek yeni veri yok. Veritabanı güncel.")

    print(f"\n--- Yükleme Özeti ---")
    print(f"Toplam Kaynak Parça: {len(all_chunks)}")
    print(f"⏭️  Atlanan (Zaten Var): {skipped_count}")
    print(f"✅ Yeni Eklenen: {new_count}")
    if bm_ret == True:
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 10
    else:
        bm25_retriever = False

    return vectorstore, bm25_retriever


if __name__ == "__main__":
    vectorstore = db_update()
