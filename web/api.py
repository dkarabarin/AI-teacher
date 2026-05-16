"""
FastAPI-сервер для веб-интерфейса ИИ преподавателя по ТТД и ТМО
"""

from __future__ import annotations

import logging
import sys
import time
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag import ThermodynamicsKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Конфигурация
# ============================================================================

BOOKS_DIR = ROOT_DIR / "books"
OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:4b"

# ============================================================================
# Глобальная инициализация базы знаний (один раз)
# ============================================================================

print("\n" + "="*50)
print("🚀 Инициализация ИИ-преподавателя")
print("="*50)

print("📚 Загрузка базы знаний...")
knowledge_base = ThermodynamicsKnowledgeBase(BOOKS_DIR)
knowledge_base.load()
print("✅ База знаний готова\n")

# ============================================================================
# Загрузка бота
# ============================================================================

get_answer = None
get_answer_stream = None
web_search = None

bot_local_path = ROOT_DIR / "bot-local.py"

if bot_local_path.exists():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bot_local", bot_local_path)
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        
        get_answer = getattr(bot_module, 'get_answer', None)
        get_answer_stream = getattr(bot_module, 'get_answer_stream', None)
        web_search = getattr(bot_module, 'web_search', None)
        
        logger.info("✅ Загружен bot-local.py")
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")

if get_answer is None:
    def get_answer(question, session_id=None):
        return f"📚 **Вопрос:** {question}\n\n**Ответ:** Это тестовый режим. Для работы установите Ollama.", "⚠️ Тест"

if get_answer_stream is None:
    async def get_answer_stream(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
        answer, source = get_answer(question, session_id)
        words = answer.split()
        for i in range(0, len(words), 3):
            chunk = ' '.join(words[i:i+3])
            yield json.dumps({"chunk": chunk + " ", "source": source if i == 0 else None, "done": False}) + "\n"
            await asyncio.sleep(0.03)
        yield json.dumps({"chunk": "", "source": source, "done": True}) + "\n"

# ============================================================================
# Система безопасности
# ============================================================================

class SecurityGuard:
    """Упрощённая система безопасности"""
    
    # Академические паттерны (блокировка списывания)
    ACADEMIC_PATTERNS = [
        (r'(?i)(готовые? ответы?|списать|сдуть|срисовать)', 'cheating'),
        (r'(?i)(сделай за меня|напиши за меня|реши за меня)', 'cheating'),
        (r'(?i)(лабораторную за меня|курсовую за меня)', 'cheating'),
        (r'(?i)(ответы на экзамен)', 'cheating'),
    ]
    
    def check_academic(self, text: str) -> tuple[bool, str]:
        """Проверка академической честности"""
        for pattern, _ in self.ACADEMIC_PATTERNS:
            if re.search(pattern, text):
                return False, "📚 Запрос отклонён. Я помогаю учиться, но не даю готовые ответы."
        return True, ""
    
    def check(self, message: str) -> tuple[bool, str]:
        """Комплексная проверка"""
        is_safe, msg = self.check_academic(message)
        if not is_safe:
            return False, msg
        return True, ""


security = SecurityGuard()

# ============================================================================
# Проверка Ollama
# ============================================================================

def check_ollama() -> bool:
    """Проверка доступности Ollama"""
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE}/models", timeout=3)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# Модели
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str
    stream: bool = True

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    source: str
    processing_time: float

class ClearRequest(BaseModel):
    session_id: str

# ============================================================================
# Хранилище сессий
# ============================================================================

class SessionStore:
    def __init__(self):
        self.sessions = {}
        self.history = {}
    
    def get_or_create(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"created": datetime.now(), "count": 0}
            self.history[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, source: str = None):
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append({
            "role": role, "content": content, "time": datetime.now().isoformat(), "source": source
        })
        if session_id in self.sessions:
            self.sessions[session_id]["count"] += 1
    
    def clear(self, session_id: str):
        if session_id in self.history:
            self.history[session_id] = []
        if session_id in self.sessions:
            self.sessions[session_id]["count"] = 0

store = SessionStore()

# ============================================================================
# FastAPI
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск веб-сервера")
    stats = knowledge_base.get_stats()
    logger.info(f"📚 База знаний: {stats.get('vectors', 0)} векторов")
    yield
    logger.info("👋 Остановка сервера")

app = FastAPI(title="ИИ преподаватель по ТТД и ТМО", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API
# ============================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/api/stats")
def stats():
    kb_stats = knowledge_base.get_stats()
    web_avail = web_search and hasattr(web_search, 'is_available') and web_search.is_available()
    ollama_ok = check_ollama()
    
    return {
        "knowledge_base_loaded": kb_stats.get("loaded", False),
        "knowledge_base_vectors": kb_stats.get("vectors", 0),
        "web_search_available": web_avail,
        "web_search_engine": web_search.get_engine_name() if web_search else "Нет",
        "ollama_model": OLLAMA_MODEL,
        "ollama_available": ollama_ok,
        "total_sessions": len(store.sessions)
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Обычный endpoint без стриминга"""
    logger.info(f"Запрос: {req.message[:50]}...")
    
    # Проверка безопасности
    is_safe, error_msg = security.check(req.message)
    if not is_safe:
        return ChatResponse(
            reply=error_msg,
            session_id=req.session_id,
            source="🛡️ Безопасность",
            processing_time=0.0
        )
    
    store.get_or_create(req.session_id)
    store.add_message(req.session_id, "user", req.message)
    
    start = time.time()
    
    try:
        result = get_answer(req.message, session_id=req.session_id)
        
        if isinstance(result, tuple):
            answer, source = result
        else:
            answer = result
            source = "🤖 Бот"
        
        elapsed = time.time() - start
        
        store.add_message(req.session_id, "assistant", answer, source)
        
        return ChatResponse(
            reply=answer,
            session_id=req.session_id,
            source=source,
            processing_time=elapsed
        )
    except Exception as e:
        logger.exception("Ошибка")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Endpoint с поддержкой стриминга"""
    logger.info(f"Stream запрос: {req.message[:50]}...")
    
    # Проверка безопасности
    is_safe, error_msg = security.check(req.message)
    if not is_safe:
        async def error_stream():
            yield json.dumps({"chunk": error_msg, "source": "🛡️ Безопасность", "done": True}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/x-ndjson")
    
    store.get_or_create(req.session_id)
    store.add_message(req.session_id, "user", req.message)
    
    async def generate():
        full_answer = ""
        source = None
        
        try:
            async for chunk_data in get_answer_stream(req.message, session_id=req.session_id):
                if isinstance(chunk_data, str):
                    try:
                        data = json.loads(chunk_data)
                        chunk = data.get("chunk", "")
                        source = data.get("source", source)
                        is_done = data.get("done", False)
                    except:
                        chunk = chunk_data
                        is_done = False
                else:
                    chunk = chunk_data.get("chunk", "")
                    source = chunk_data.get("source", source)
                    is_done = chunk_data.get("done", False)
                
                full_answer += chunk
                yield json.dumps({"chunk": chunk, "source": source, "done": is_done}) + "\n"
                
                if is_done:
                    store.add_message(req.session_id, "assistant", full_answer, source)
                    
        except Exception as e:
            logger.exception(f"Stream ошибка: {e}")
            yield json.dumps({"chunk": f"\n\n❌ Ошибка: {str(e)}", "source": "⚠️ Ошибка", "done": True}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/api/clear")
def clear(req: ClearRequest):
    store.clear(req.session_id)
    return {"status": "ok"}


# ============================================================================
# HTML интерфейс
# ============================================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ИИ преподаватель по ТТД и ТМО</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #0f3460);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .header h1 { font-size: 1.2rem; }
        .status { display: flex; gap: 8px; flex-wrap: wrap; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
        }
        .clear-btn {
            background: rgba(255,255,255,0.2);
            margin-left: 8px;
            cursor: pointer;
        }
        .clear-btn:hover { background: rgba(255,255,255,0.3); }
        .chat {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f0f2f5;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message { display: flex; gap: 10px; }
        .message.user { justify-content: flex-end; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
        }
        .message.user .avatar { background: #e94560; }
        .message.bot .avatar { background: #0f3460; }
        .bubble {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 18px;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message.user .bubble { background: #e94560; color: white; }
        .bubble-text { line-height: 1.5; }
        .bubble-source {
            font-size: 0.7rem;
            margin-top: 6px;
            opacity: 0.7;
        }
        .input-area {
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
            display: flex;
            gap: 10px;
        }
        textarea {
            flex: 1;
            padding: 10px 15px;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            resize: none;
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
        }
        textarea:focus { border-color: #e94560; }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.95rem;
        }
        button:hover { background: #c73e56; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .typing {
            padding: 10px 15px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            display: flex;
            gap: 8px;
            color: #666;
            font-style: italic;
        }
        .dot {
            width: 6px;
            height: 6px;
            background: #666;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%,60%,100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        @media (max-width: 768px) {
            .bubble { max-width: 85%; }
            .header h1 { font-size: 1rem; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔥 ИИ преподаватель по ТТД и ТМО</h1>
        <div class="status">
            <div class="badge" id="badge-kb">📚 RAG</div>
            <div class="badge" id="badge-web">🌐 Поиск</div>
            <div class="badge" id="badge-guard">🛡️ Guardrails</div>
            <div class="badge clear-btn" id="btn-clear">🗑️ Очистить</div>
        </div>
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-area">
        <textarea id="input" placeholder="Введите вопрос по термодинамике..." rows="1"></textarea>
        <button id="btn-send">📤 Отправить</button>
    </div>
</div>

<script>
var sessionId = localStorage.getItem("session_id");
if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("session_id", sessionId);
}
var isLoading = false;
var chat = document.getElementById("chat");
var input = document.getElementById("input");
var sendBtn = document.getElementById("btn-send");
var clearBtn = document.getElementById("btn-clear");
var badgeKb = document.getElementById("badge-kb");
var badgeWeb = document.getElementById("badge-web");
var badgeGuard = document.getElementById("badge-guard");

function formatText(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>").replace(/\\n/g, "<br>");
}

function addUserMessage(text) {
    var div = document.createElement("div");
    div.className = "message user";
    div.innerHTML = '<div class="avatar">👤</div><div class="bubble"><div class="bubble-text">' + formatText(text) + '</div></div>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function addBotMessage(text, source) {
    var div = document.createElement("div");
    div.className = "message bot";
    var sourceHtml = source ? '<div class="bubble-source">' + source + '</div>' : "";
    div.innerHTML = '<div class="avatar">🤖</div><div class="bubble"><div class="bubble-text">' + formatText(text) + '</div>' + sourceHtml + '</div>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    if (window.MathJax) MathJax.typesetPromise([div]).catch(console.error);
}

function showTyping() {
    var div = document.createElement("div");
    div.id = "typing";
    div.className = "typing";
    div.innerHTML = '<div style="display:flex;gap:4px"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div><span>🤖 Преподаватель печатает...</span>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
    var el = document.getElementById("typing");
    if (el) el.remove();
}

function sendMessage() {
    var message = input.value.trim();
    if (!message || isLoading) return;
    input.value = "";
    input.style.height = "auto";
    addUserMessage(message);
    isLoading = true;
    sendBtn.disabled = true;
    showTyping();
    fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message, session_id: sessionId })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
        hideTyping();
        addBotMessage(data.reply, data.source);
    })
    .catch(function(error) {
        hideTyping();
        addBotMessage("❌ Ошибка: " + error.message);
    })
    .finally(function() {
        isLoading = false;
        sendBtn.disabled = false;
        input.focus();
    });
}

function clearChat() {
    if (!confirm("Очистить историю диалога?")) return;
    fetch("/api/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
    })
    .then(function() {
        chat.innerHTML = "";
        addBotMessage("🧹 История диалога очищена. Задайте новый вопрос!");
    })
    .catch(function(err) {
        addBotMessage("❌ Ошибка очистки");
    });
}

function loadStats() {
    fetch("/api/stats")
    .then(function(res) { return res.json(); })
    .then(function(stats) {
        if (badgeKb) badgeKb.innerHTML = stats.knowledge_base_loaded ? "📚 RAG ✅" : "📚 RAG ❌";
        if (badgeWeb) badgeWeb.innerHTML = stats.web_search_available ? "🌐 Поиск ✅" : "🌐 Поиск ❌";
        if (badgeGuard) badgeGuard.innerHTML = "🛡️ Guardrails ✅";
    })
    .catch(function(e) { console.log("Stats error:", e); });
}

function addWelcomeMessage() {
    addBotMessage("👋 Здравствуйте! Я ИИ-преподаватель по технической термодинамике (ТТД) и тепломассообмену (ТМО).\\n\\nЗадайте мне вопрос по:\\n• 📚 Лабораторным работам\\n• 📊 Обработке данных\\n• 📖 Теоретическому материалу\\n• 🔬 Подготовке к экзаменам");
}

sendBtn.onclick = sendMessage;
clearBtn.onclick = clearChat;
input.onkeydown = function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
};
input.oninput = function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
};

loadStats();
setInterval(loadStats, 30000);
addWelcomeMessage();
input.focus();
</script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)