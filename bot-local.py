"""
Локальная RAG система для работы с PDF-документами по термодинамике
С поддержкой DuckDuckGo поиска и кэшированием FAISS индекса
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple
import time
import re

from dotenv import load_dotenv

# LangChain для Ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Tavily для веб-поиска
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# DuckDuckGo поиск
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Langfuse для наблюдаемости
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Наш RAG модуль (обновлённый)
from rag import ThermodynamicsKnowledgeBase

# Отключаем предупреждения
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================================
# Конфигурация
# ============================================================================

BOOKS_DIR = Path("./books")

# Ollama (локальная модель)
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED = False

# Параметры RAG
K_RETRIEVAL = 5
MAX_HISTORY = 10
SEARCH_MAX_RESULTS = 3

# ============================================================================
# УЛУЧШЕННЫЙ СИСТЕМНЫЙ ПРОМПТ
# ============================================================================

SYSTEM_PROMPT = """Ты — преподаватель по технической термодинамике (ТТД) и тепломассообмену (ТМО).

ОБЛАСТЬ ЗНАНИЙ:
- Техническая термодинамика (ТТД): циклы, процессы, законы термодинамики
- Тепломассообмен (ТМО): теплопроводность, конвекция, излучение, массообмен

ПРИОРИТЕТ ИСТОЧНИКОВ:
1. В ПЕРВУЮ ОЧЕРЕДЬ используй материал из PDF-документов в папке books/
2. Если информации недостаточно — используй веб-поиск
3. Не придумывай факты. Если ответа нет — честно скажи об этом

ПРАВИЛА ФОРМАТИРОВАНИЯ ФОРМУЛ:
1. Все формулы ОБЯЗАТЕЛЬНО заключай в $$...$$ для отдельных формул
2. Для формул в тексте используй $...$
3. Используй \\frac{}{} для дробей, \\cdot для умножения
4. Греческие буквы: \\alpha, \\beta, \\gamma, \\Delta, \\pi

Пример: "Первый закон термодинамики: $$\\Delta U = Q - A$$"

Отвечай на русском языке. Будь полезным и точным.
"""


# ============================================================================
# ВЕБ-ПОИСК (Tavily + DuckDuckGo)
# ============================================================================

class WebSearch:
    """Универсальный веб-поиск с поддержкой Tavily и DuckDuckGo"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_client = None
        self.use_tavily = False
        self.use_duckduckgo = False
        self.search_engine = "none"
        
        # Инициализация Tavily (если есть ключ)
        if tavily_api_key and TAVILY_AVAILABLE:
            try:
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                self.use_tavily = True
                self.search_engine = "tavily"
                logger.info("✅ Веб-поиск: Tavily (API ключ найден)")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации Tavily: {e}")
        
        # Инициализация DuckDuckGo (если Tavily не доступен)
        if not self.use_tavily and DDGS_AVAILABLE:
            self.use_duckduckgo = True
            self.search_engine = "duckduckgo"
            logger.info("✅ Веб-поиск: DuckDuckGo (бесплатный, без ключа)")
        elif not self.use_tavily and not DDGS_AVAILABLE:
            logger.warning("⚠️ Веб-поиск недоступен. Установите: pip install duckduckgo-search")
    
    def search(self, query: str, max_results: int = SEARCH_MAX_RESULTS) -> Optional[str]:
        """Выполняет поиск с использованием доступного поискового движка"""
        if self.use_tavily:
            return self._search_tavily(query, max_results)
        elif self.use_duckduckgo:
            return self._search_duckduckgo(query, max_results)
        return None
    
    def _search_tavily(self, query: str, max_results: int) -> Optional[str]:
        """Поиск через Tavily API"""
        try:
            response = self.tavily_client.search(
                query,
                search_depth="basic",
                include_answer=False,
                max_results=max_results,
            )
            results = response.get("results", [])
            if not results:
                return None
            
            formatted = []
            for r in results[:max_results]:
                title = r.get("title", "Без названия")
                content = r.get("content", "")
                url = r.get("url", "")
                score = r.get("score", 0)
                formatted.append(
                    f"📄 **{title}** (релевантность: {score:.2f})\n"
                    f"{content[:500]}\n🔗 {url}"
                )
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            logger.error(f"Ошибка Tavily поиска: {e}")
            return None
    
    def _search_duckduckgo(self, query: str, max_results: int) -> Optional[str]:
        """Поиск через DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
                if not results:
                    return None
                
                formatted = []
                for r in results[:max_results]:
                    title = r.get('title', 'Без названия')
                    body = r.get('body', '')
                    href = r.get('href', '')
                    
                    body = re.sub(r'\s+', ' ', body).strip()
                    if len(body) > 500:
                        body = body[:500] + "..."
                    
                    formatted.append(
                        f"📄 **{title}**\n"
                        f"{body}\n"
                        f"🔗 {href}"
                    )
                
                logger.info(f"DuckDuckGo: найдено {len(results)} результатов")
                return "\n\n---\n\n".join(formatted)
                
        except Exception as e:
            logger.error(f"Ошибка DuckDuckGo поиска: {e}")
            return None
    
    def is_available(self) -> bool:
        return self.use_tavily or self.use_duckduckgo
    
    def get_engine_name(self) -> str:
        if self.use_tavily:
            return "Tavily"
        elif self.use_duckduckgo:
            return "DuckDuckGo"
        return "Недоступен"


# ============================================================================
# Инициализация Langfuse
# ============================================================================

langfuse = None
if LANGFUSE_AVAILABLE and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        langfuse.auth_check()
        LANGFUSE_ENABLED = True
        print("✅ Langfuse инициализирован")
    except Exception as e:
        print(f"⚠️ Langfuse не инициализирован: {e}")


# ============================================================================
# Инициализация компонентов
# ============================================================================

# LLM
llm = ChatOpenAI(
    openai_api_key="fake_key",
    openai_api_base=OLLAMA_BASE,
    model_name=OLLAMA_MODEL,
    temperature=0.7,
    max_tokens=2048,
)

# Веб-поиск
web_search = WebSearch(TAVILY_API_KEY)

# База знаний (с поддержкой кэширования)
knowledge_base = ThermodynamicsKnowledgeBase(BOOKS_DIR)
knowledge_base.load()


# ============================================================================
# Вспомогательные функции
# ============================================================================

def is_educational_query(question: str) -> bool:
    """Проверка, является ли запрос образовательным"""
    educational_keywords = [
        "термодинамик", "тепломассообмен", "энтропи", "энтальпи",
        "формула", "расчет", "закон", "цикл", "кпд", "нуссельт",
        "лабораторн", "экзамен", "помоги", "объясни", "расскажи",
        "thermodynamics", "heat transfer", "entropy", "enthalpy",
        "nusselt", "reynolds", "prandtl", "fourier"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in educational_keywords)


def format_sources(sources: list) -> str:
    """Форматирует список источников для вывода"""
    if not sources:
        return ""
    unique_sources = []
    for s in sources:
        src = f"{s['source_file']}, стр. {s['page_num']}"
        if src not in unique_sources:
            unique_sources.append(src)
    return "\n\n📚 *Источники:* " + ", ".join(unique_sources[:3])


# ============================================================================
# Функции ответов
# ============================================================================

def answer_from_pdf(question: str) -> Optional[Tuple[str, str]]:
    """
    Отвечает на вопрос из PDF-документов.
    Возвращает (ответ, строка_с_источниками)
    """
    if not knowledge_base.vectorstore:
        return None
    
    # Получаем чанки с метаданными
    results = knowledge_base.get_relevant_chunks_with_sources(question, k=K_RETRIEVAL)
    if not results:
        return None
    
    context = "\n\n---\n\n".join([r["content"] for r in results])
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"\n\n--- ИЗ PDF-ДОКУМЕНТОВ ---\n{context}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        sources_text = format_sources(results)
        return response.content, sources_text
    except Exception as e:
        logger.error(f"Ошибка RAG: {e}")
        return None


def answer_from_web(question: str) -> Optional[Tuple[str, str]]:
    """Отвечает на вопрос через веб-поиск."""
    if not web_search.is_available():
        return None
    
    search_results = web_search.search(question, max_results=SEARCH_MAX_RESULTS)
    if not search_results:
        return None
    
    engine_name = web_search.get_engine_name()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"\n\n--- ВЕБ-ПОИСК ({engine_name}) ---\n{search_results}"),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content, f"\n\n🌐 *Источник:* {engine_name}"
    except Exception as e:
        logger.error(f"Ошибка веб-ответа: {e}")
        return None


def answer_direct(question: str) -> Tuple[str, str]:
    """Отвечает без контекста (только LLM)."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content, ""
    except Exception as e:
        logger.error(f"Ошибка LLM: {e}")
        return f"Извините, произошла ошибка: {e}", ""


def get_answer(question: str, session_id: str = None) -> Tuple[str, str]:
    """Получает ответ с указанием источника."""
    
    # Проверка на образовательный запрос
    if not is_educational_query(question) and len(question) > 15:
        return """📚 Я специализируюсь на вопросах по **технической термодинамике (ТТД)** и **тепломассообмену (ТМО)**.

**Примеры вопросов:**
• Объясни первый закон термодинамики
• Что такое число Нуссельта?
• Как рассчитать КПД цикла Карно?
• Напиши формулу теплопроводности

Задайте конкретный вопрос по этим темам!""", "🎓 Совет"
    
    print(f"  🔍 Поиск в PDF: {question[:50]}...")
    result = answer_from_pdf(question)
    if result:
        answer, sources = result
        print("  ✅ Найдено в PDF")
        return answer + sources, "📚 PDF"
    
    if web_search.is_available():
        print(f"  🌐 Поиск в интернете ({web_search.get_engine_name()})...")
        result = answer_from_web(question)
        if result:
            answer, sources = result
            print(f"  ✅ Найдено через {web_search.get_engine_name()}")
            return answer + sources, f"🌐 {web_search.get_engine_name()}"
    
    print("  🤖 Использование LLM...")
    answer, _ = answer_direct(question)
    return answer, "🎓 LLM"


# ============================================================================
# Консольный интерфейс
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🔥 ИИ-преподаватель по ТТД и ТМО")
    print("   Техническая термодинамика & Тепломассообмен")
    print("="*60)
    
    # Статистика
    stats = knowledge_base.get_stats()
    print(f"📚 База знаний: {'загружена' if stats.get('loaded') else 'не загружена'}")
    print(f"   Векторов: {stats.get('vectors', 0)}")
    print(f"   Файлов: {len(stats.get('files', []))}")
    print(f"🌐 Веб-поиск: {web_search.get_engine_name()}")
    print(f"   Доступен: {'✅' if web_search.is_available() else '❌'}")
    print(f"🤖 Модель: {OLLAMA_MODEL}")
    print("="*60)
    print("Введите вопрос или 'quit' для выхода\n")
    
    while True:
        try:
            question = input("❓ Вопрос: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            print("  ⏳ Думаю...")
            start = time.time()
            answer, source = get_answer(question)
            elapsed = time.time() - start
            
            print(f"\n{source} ОТВЕТ ({elapsed:.1f}с):")
            print(answer)
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    print("\n👋 До свидания!")


if __name__ == "__main__":
    main()