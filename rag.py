"""
RAG модуль для работы с PDF-документами по термодинамике
С поддержкой кэширования FAISS индекса
"""

import os
import logging
import warnings
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Конфигурация
BOOKS_DIR = Path("./books")
CACHE_DIR = Path("./cache")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
K_RETRIEVAL = 5

# Модель эмбеддингов
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Создаём директорию для кэша
CACHE_DIR.mkdir(exist_ok=True)


class ThermodynamicsKnowledgeBase:
    """Управление базой знаний из PDF и текстовых документов."""
    
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = Path(books_dir)
        self.vectorstore = None
        self.retriever = None
        self._embeddings = None
        self._loaded = False
        self._stats = {"total_pages": 0, "total_chunks": 0, "files": []}
        self._cache_path = CACHE_DIR / "faiss_index.pkl"
    
    def load(self, force_reload: bool = False) -> bool:
        """Загружает документы и создает векторное хранилище с кэшированием."""
        if self._loaded and not force_reload:
            logger.info("База знаний уже загружена")
            return True
        
        # Пробуем загрузить из кэша
        if not force_reload and self._cache_path.exists():
            try:
                with open(self._cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.vectorstore = cache_data['vectorstore']
                    self._stats = cache_data['stats']
                    self._loaded = True
                    print(f"✅ Загружено из кэша: {self._stats['vectors']} векторов")
                    return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки кэша: {e}")
                # Если кэш повреждён, удаляем его и пересоздаём
                try:
                    self._cache_path.unlink()
                except:
                    pass
        
        # Если кэша нет, загружаем документы и создаём индекс заново
        if not self.books_dir.exists():
            logger.warning(f"Папка {self.books_dir} не найдена")
            print(f"⚠️ Папка {self.books_dir} не найдена.")
            return False
        
        # Поиск всех документов
        pdf_files = list(self.books_dir.glob("**/*.pdf"))
        all_files = pdf_files
        
        if not all_files:
            logger.warning(f"Документы не найдены в {self.books_dir}")
            print(f"⚠️ Документы не найдены в {self.books_dir}")
            return False
        
        print(f"\n📚 Загрузка документов из {self.books_dir}...")
        print(f"   Найдено файлов: {len(all_files)}")
        
        # Загрузка всех документов с сохранением источников
        all_docs = []
        
        for file_path in all_files:
            try:
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                
                # Добавляем метаданные с именем файла и номером страницы
                for i, page in enumerate(pages):
                    page.metadata['source_file'] = file_path.name
                    page.metadata['page_num'] = i + 1
                    page.metadata['source'] = str(file_path)
                
                all_docs.extend(pages)
                self._stats["total_pages"] += len(pages)
                self._stats["files"].append({"name": file_path.name, "pages": len(pages)})
                print(f"   • {file_path.name}: {len(pages)} стр.")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_path.name}: {e}")
        
        if not all_docs:
            return False
        
        print(f"✅ Загружено страниц: {self._stats['total_pages']}")
        
        # Разбивка на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)
        self._stats["total_chunks"] = len(chunks)
        print(f"🔪 Создано чанков: {len(chunks)}")
        
        # Создание эмбеддингов
        print(f"⏳ Загрузка модели эмбеддингов...")
        
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Создание векторного хранилища
        print("⏳ Создание векторного хранилища...")
        self.vectorstore = FAISS.from_documents(chunks, self._embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
        
        self._stats["vectors"] = self.vectorstore.index.ntotal
        print(f"✅ Векторное хранилище создано: {self._stats['vectors']} векторов")
        
        # Сохраняем в кэш
        try:
            cache_data = {
                'vectorstore': self.vectorstore,
                'stats': self._stats
            }
            with open(self._cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Кэш сохранён: {self._cache_path}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить кэш: {e}")
        
        self._loaded = True
        return True
    
    def get_relevant_chunks(self, query: str, k: int = None) -> List[str]:
        """Возвращает релевантные фрагменты из базы знаний (только текст)."""
        if not self._loaded or self.vectorstore is None:
            return []
        
        k = k or K_RETRIEVAL
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    def get_relevant_chunks_with_sources(self, query: str, k: int = None) -> List[Dict]:
        """Возвращает релевантные фрагменты с метаданными (источник, страница)."""
        if not self._loaded or self.vectorstore is None:
            return []
        
        k = k or K_RETRIEVAL
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source_file": doc.metadata.get("source_file", "unknown"),
                    "page_num": doc.metadata.get("page_num", 0),
                    "source": doc.metadata.get("source", "unknown"),
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """Поиск с возвратом метаданных."""
        return self.get_relevant_chunks_with_sources(query, k)
    
    def get_stats(self) -> Dict:
        """Возвращает статистику базы знаний."""
        if not self._loaded or self.vectorstore is None:
            return {"loaded": False, "vectors": 0}
        
        return {
            "loaded": self._loaded,
            "vectors": self._stats.get("vectors", 0),
            "embedding_model": EMBEDDING_MODEL,
            "total_chunks": self._stats.get("total_chunks", 0),
            "total_pages": self._stats.get("total_pages", 0),
            "files": self._stats.get("files", []),
        }