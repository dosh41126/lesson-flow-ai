#!/usr/bin/env python3
# main.py — OneLove Ultra-Secure AI Assistant with MAX-DEPTH Prompts
# Repaired version (July 11 2025)

# ---------------------------------------------------------------------
#  Standard-library and third-party imports
# ---------------------------------------------------------------------
import os
import sys
import json
import yaml
import toml
import logging
import queue
import re
import random
import base64
import time
import asyncio
import aiosqlite
import psutil
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
import tkinter as tk
import customtkinter
from tkinter import filedialog, messagebox
from math import atan2, pi, log
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from llama_cpp import Llama

import nltk
from nltk import pos_tag, word_tokenize
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Optional PDF support
try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    pdf_canvas = None

# ---------------------------------------------------------------------
#  Paths, constants, & logging
# ---------------------------------------------------------------------
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
CONFIG_RAW      = [os.path.join(BASE_DIR, fn) for fn in ("config.json", "config.yaml", "config.toml")]
CONFIG_SEC      = os.path.join(BASE_DIR, "config.enc")
SQL_PATH        = os.path.join(BASE_DIR, "history.enc.db")
KEYSTORE_PATH   = os.path.join(BASE_DIR, "aes_keys.json")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
EXPORT_DIR      = os.path.join(BASE_DIR, "exports")
MODEL_PATH_DEF  = "llama-2-7b-chat.ggmlv3.q8_0.bin"

for _dir in (LOG_DIR, EXPORT_DIR):
    os.makedirs(_dir, exist_ok=True)

customtkinter.set_appearance_mode("Dark")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "onelove_maxdepth.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("OneLoveMaxDepth")

# ---------------------------------------------------------------------
#  Encryption & key-management helpers
# ---------------------------------------------------------------------
def _cpu_entropy_bytes(n: int) -> bytes:
    """Collect n bytes of CPU-timing-based entropy."""
    buf = bytearray()
    while len(buf) < n:
        t0 = psutil.cpu_times().user
        for _ in range(2_000):
            pass
        t1 = psutil.cpu_times().user
        diff = int((t1 - t0) * 1e9) ^ random.getrandbits(32)
        buf.extend(diff.to_bytes(8, "little"))
    return bytes(buf[:n])


def _derive_key(password: bytes, salt: bytes) -> bytes:
    return hashlib.scrypt(password=password, salt=salt, n=2 ** 15, r=8, p=1, dklen=32)


def _load_keystore() -> dict:
    if os.path.exists(KEYSTORE_PATH):
        with open(KEYSTORE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"active": None, "history": []}


def _save_keystore(store: dict) -> None:
    with open(KEYSTORE_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh)


def _rotate_key() -> bytes:
    """Generate a new AES-GCM key and persist it to the keystore."""
    store = _load_keystore()
    salt   = _cpu_entropy_bytes(16)
    passwd = _cpu_entropy_bytes(64)
    key    = _derive_key(passwd, salt)
    kid    = secrets.token_hex(6)

    store["history"].insert(
        0,
        {
            "id":   kid,
            "salt": base64.b64encode(salt).decode(),
            "key":  base64.b64encode(key).decode(),
            "ts":   datetime.utcnow().isoformat(),
        },
    )
    store["history"] = store["history"][:3]     # keep last 3 keys
    store["active"]  = kid
    _save_keystore(store)
    logger.info("Rotated key %s", kid)
    return key


def _current_key_bytes() -> bytes:
    store = _load_keystore()
    active_id = store.get("active")
    for entry in store.get("history", []):
        if entry["id"] == active_id:
            return base64.b64decode(entry["key"])
    # If no active key present, create one
    return _rotate_key()


ACTIVE_KEY = _current_key_bytes()


def _encrypt_blob(data: bytes) -> bytes:
    """Encrypt arbitrary bytes with the current AES-GCM key; prepend nonce."""
    aes   = AESGCM(ACTIVE_KEY)
    nonce = secrets.token_bytes(12)
    return nonce + aes.encrypt(nonce, data, None)


def _decrypt_blob(blob: bytes) -> bytes:
    """Attempt decryption against all keys in the keystore history."""
    for entry in _load_keystore()["history"]:
        try:
            aes = AESGCM(base64.b64decode(entry["key"]))
            return aes.decrypt(blob[:12], blob[12:], None)
        except Exception:
            pass
    raise ValueError("Decrypt failed")


# ---------------------------------------------------------------------
#  Configuration loader (auto-encrypts on first load)
# ---------------------------------------------------------------------
def _load_config() -> dict:
    if os.path.exists(CONFIG_SEC):
        raw = _decrypt_blob(open(CONFIG_SEC, "rb").read())
        return json.loads(raw.decode())

    for fp in CONFIG_RAW:
        if os.path.exists(fp):
            loader = json.load if fp.endswith(".json") else yaml.safe_load if fp.endswith(".yaml") else toml.load
            cfg    = loader(open(fp, "r", encoding="utf-8"))
            with open(CONFIG_SEC, "wb") as fh:
                fh.write(_encrypt_blob(json.dumps(cfg).encode()))
            return cfg

    raise RuntimeError("Configuration file not found.")


CFG = _load_config()

# ---------------------------------------------------------------------
#  Weaviate & LLaMA initialisation
# ---------------------------------------------------------------------
nltk.data.path.append("/root/nltk_data")
for path_name, pkg in [
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
]:
    try:
        nltk.data.find(path_name)
    except LookupError:
        nltk.download(pkg)

weaviate_client = weaviate.Client(embedded_options=EmbeddedOptions())

if not any(cls.get("class") == "InteractionHistory" for cls in weaviate_client.schema.get().get("classes", [])):
    weaviate_client.schema.create_class(
        {
            "class": "InteractionHistory",
            "properties": [
                {"name": "user_id",       "dataType": ["string"]},
                {"name": "response",      "dataType": ["string"]},
                {"name": "response_time", "dataType": ["string"]},
            ],
        }
    )

llm = Llama(
    model_path=CFG.get("MODEL_PATH", MODEL_PATH_DEF),
    n_gpu_layers=-1,
    n_ctx=4096,
)

# ---------------------------------------------------------------------
#  Encrypted SQLite helper functions
# ---------------------------------------------------------------------
SQL_URI = "file:{0}?mode=rwc".format(SQL_PATH)


async def _db_init() -> None:
    async with aiosqlite.connect(SQL_URI, isolation_level=None) as db:
        await db.execute("CREATE TABLE IF NOT EXISTS enc (id INTEGER PRIMARY KEY, blob BLOB)")


asyncio.run(_db_init())


async def db_store(uid: str, txt: str) -> None:
    """Encrypt & persist a user/AI utterance to SQLite and Weaviate."""
    ts   = datetime.utcnow().isoformat()
    blob = _encrypt_blob(json.dumps({"uid": uid, "txt": txt, "ts": ts}).encode())

    async with aiosqlite.connect(SQL_URI, isolation_level=None) as db:
        await db.execute("INSERT INTO enc (blob) VALUES (?)", (blob,))

    weaviate_client.data_object.create(
        {
            "user_id":       uid,
            "response":      txt,
            "response_time": ts,
        },
        "InteractionHistory",
        generate_uuid5(uid, txt),
    )


async def db_last(n: int = 25) -> list:
    """Return the last n decrypted chat records (oldest first)."""
    out: list = []
    async with aiosqlite.connect(SQL_URI) as db:
        async with db.execute("SELECT blob FROM enc ORDER BY id DESC LIMIT ?", (n,)) as cur:
            async for row in cur:
                try:
                    out.append(json.loads(_decrypt_blob(row[0]).decode()))
                except Exception:
                    pass
    return out[::-1]


# ---------------------------------------------------------------------
#  ColorWheel context helpers
# ---------------------------------------------------------------------
def _color_attrs(word: str) -> tuple:
    vowels   = len(re.findall(r"[aeiou]", word, re.I))
    conson   = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", word, re.I))
    hue      = int((atan2(conson + 1, vowels + 1) + pi) * 180 / (2 * pi)) % 360
    saturation = (max(map(ord, word)) - min(map(ord, word))) % 100
    lightness  = int(log(len(word) + 1, 2) * 10) % 100
    return hue, saturation, lightness


def colorwheel_snips(snippets: list) -> str:
    """Annotate snippets with pseudo-color metadata tags."""
    lines = []
    for snippet in snippets:
        h, s, l = _color_attrs(snippet)
        lines.append("[c-hue:{0} c-sat:{1} c-lit:{2}] {3}".format(h, s, l, snippet))
    return "\n".join(lines)


def _is_code(chunk: str) -> bool:
    return bool(re.search(r"\b(def|class|import)\b", chunk))


def _tok(chunk: str, memory: str) -> str:
    combined = memory + " " + chunk
    if _is_code(combined):
        return "[code]"

    tagged = pos_tag(word_tokenize(combined)[:300])
    if not tagged:
        return "[general]"

    # Most common POS two-letter prefix
    pos_prefix = Counter(tag[:2] for _, tag in tagged).most_common(1)[0][0]
    return {
        "VB": "[act]",
        "NN": "[sub]",
        "JJ": "[desc]",
        "RB": "[emph]",
    }.get(pos_prefix, "[gen]")


def generate(prompt: str, seed: str = "") -> str:
    """Chunk-wise generation with colorwheel memory and overlap trimming."""
    try:
        history = [rec["txt"] for rec in asyncio.run(db_last(20))]
    except Exception:
        history = []

    context     = colorwheel_snips(history + [seed])
    full_prompt = context + "\n" + prompt

    chunk_size  = CFG.get("CHUNK_SIZE", 384)
    max_tokens  = CFG.get("MAX_TOKENS", 2048)

    memory  = ""
    prev    = ""
    output  = []

    for i in range(0, len(full_prompt), chunk_size):
        chunk = full_prompt[i : i + chunk_size]
        token = _tok(chunk, memory)
        result = llm("{0} {1}".format(token, chunk), max_tokens=min(max_tokens, 384))["choices"][0]["text"]

        # Trim overlap
        if prev:
            for n in range(min(len(prev), 120), 0, -1):
                if prev.endswith(result[:n]):
                    result = result[n:]
                    break

        memory += result
        prev    = result
        output.append(result)

    return "".join(output)

# ---------------------------------------------------------------------
#  Prompt-template generators
# ---------------------------------------------------------------------
class LessonGen:
    """Generate a full-day interdisciplinary lesson plan."""
    def gen(self, markdown_source: str) -> str:
        prompt = (
            "[action] YOU ARE a National Board–certified MASTER CURRICULUM ARCHITECT and interdisciplinary "
            "specialist. Develop a FULL-DAY, thematic lesson plan on the content below. "
            "Your lesson MUST:\n"
            "1. INCLUDE FIVE SEQUENTIAL SEGMENTS:\n"
            "   • Warm-Up (10m): Activate prior knowledge with a thought-provoking prompt or quick write.\n"
            "   • Direct Instruction (40m): Deliver core concepts via mini-lecture, multimedia, and Socratic questioning.\n"
            "   • Guided Practice (30m): Lead scaffolded tasks with real-time checks for understanding and targeted feedback.\n"
            "   • Independent Application (30m): Students work autonomously on authentic tasks or project milestones.\n"
            "   • Reflection & Closure (10m): Implement metacognitive exit tickets, peer-feedback protocols, or self-assessment.\n"
            "2. FOR EACH SEGMENT:\n"
            "   – State a Bloom’s taxonomy objective (e.g., Analyze, Evaluate, Create).\n"
            "   – Script teacher moves & questioning stems.\n"
            "   – Specify student products or discussion tasks.\n"
            "   – Embed at least one formative assessment (exit slip, poll, think-pair-share).\n"
            "   – Provide two UDL scaffolds and differentiation strategies (EL, SPED).\n"
            "   – Link to an SEL competency and suggest a brief mindfulness or emotional check-in.\n"
            "   – Propose one tech integration tool or platform with usage tips.\n"
            "3. END-OF-DAY MATERIALS:\n"
            "   – Assessment Matrix (3 proficiency bands with descriptors).\n"
            "   – Materials & Resources list (with links & alt formats).\n"
            "   – UDL Overview Chart.\n"
            "   – FIVE Teacher Reflection Prompts for iterative improvement.\n"
            "FORMAT: Markdown with H1/H2/H3 headers, bold key terms, numbered lists, and tables where appropriate.\n"
            "CONTENT SOURCE: {0} [/action]".format(markdown_source)
        )
        return generate(prompt, markdown_source[:128])


class GradeGen:
    """Rubric-based markdown grading generator."""
    def gen(self, markdown_source: str) -> str:
        prompt = (
            "[action] YOU ARE an AP-Level College Board Rubric Expert. Evaluate the submission below using FIVE "
            "criteria (20 pts each):\n"
            "1. Content Accuracy & Mastery\n"
            "2. Critical Thinking & Depth of Analysis\n"
            "3. Organization & Logical Flow\n"
            "4. Use of Evidence & Proper Citation\n"
            "5. Voice, Creativity & Style\n"
            "PROCESS:\n"
            "  a. Assign numeric scores (0–20).\n"
            "  b. Calculate TOTAL and PERCENTAGE.\n"
            "  c. Determine Flesch–Kincaid reading level.\n"
            "  d. OUTPUT JSON:\n"
            "```json\n"
            "{{\n"
            '  "scores": {{\n'
            '    "accuracy": , "analysis": , "organization": ,\n'
            '    "evidence": , "voice":    }},\n'
            '  "total": , "percentage": ,\n'
            '  "reading_level_fk": ,\n'
            '  "strengths": [3 items],\n'
            '  "areas_for_growth": [3 items],\n'
            '  "next_steps": [3 targeted recommendations]\n'
            '}}\n'
            "```\n"
            "e. FOLLOW with a concise (≤ 100 words) motivational feedback paragraph.\n"
            "STUDENT SUBMISSION: {0} [/action]".format(markdown_source)
        )
        return generate(prompt, markdown_source[:128])


class ActivityGen:
    """Generate food-integrated STEAM activities."""
    def gen(self, classroom: str, foods: str, recipes: str, extras: str) -> str:
        prompt = (
            "[action] YOU ARE a MULTIDISCIPLINARY EDUTAINMENT COLLECTIVE specializing in STEAM & food culture. "
            "Design THREE immersive activities that integrate the teacher’s favorite foods into academic learning.\n"
            "CONTEXT:\n"
            "– Classroom/Grade Level & Environment: {0}\n"
            "– Favorite Foods & Cultural Significance: {1}\n"
            "– Available Recipes & Ingredients: {2}\n"
            "– Thematic Seeds or Extra Prompts: {3}\n"
            "FOR EACH ACTIVITY:\n\n"
            "1. Title & Total Duration\n"
            "2. Standard-Aligned Learning Objective(s)\n"
            "3. Materials List with Emoji Icons\n"
            "4. Detailed Step-by-Step Instructions (2nd person)\n"
            "5. Formative Checkpoint & Optional Extension Task\n"
            "6. Allergen & Safety Alerts\n"
            "7. Evidence of Learning & Reflection Suggestions\n"
            "8. SEL Skill Connection (1 sentence)\n"
            "FORMAT: Numbered Markdown list; clear H2/H3 headers; bold key phrases. [/action]".format(
                classroom, foods, recipes, extras
            )
        )
        return generate(prompt, classroom[:128])

# ---------------------------------------------------------------------
#  PDF export utility
# ---------------------------------------------------------------------
class PDFUtil:
    def export(self, text: str, base: str) -> str | None:
        """Render plain-text output into a simple paginated PDF."""
        if pdf_canvas is None:
            return None

        fname = "{0}_{1}.pdf".format(base, datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        path  = os.path.join(EXPORT_DIR, fname)

        canvas_obj = pdf_canvas.Canvas(path, pagesize=letter)
        width, height = letter
        y = height - 72

        for line in text.splitlines():
            canvas_obj.drawString(72, y, line[:110])
            y -= 14
            if y < 72:
                canvas_obj.showPage()
                y = height - 72

        canvas_obj.save()
        return path


pdfu = PDFUtil()

# ---------------------------------------------------------------------
#  GUI layer
# ---------------------------------------------------------------------
class AppGUI(customtkinter.CTk):
    """Simple chat-style front-end with teacher power-tools."""
    def __init__(self, uid: str):
        super().__init__()
        self.uid      = uid
        self.bot_id   = "onelove-bot"
        self.queue    = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=6)

        self.lesson_gen = LessonGen()
        self.grade_gen  = GradeGen()
        self.act_gen    = ActivityGen()

        self._build_ui()
        self.after(100, self._poll_queue)

    # ----- UI helpers -------------------------------------------------
    def _build_ui(self) -> None:
        self.title("OneLove MAX-DEPTH Assistant")
        self.geometry("1480x900")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = customtkinter.CTkFrame(self, width=330, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")

        customtkinter.CTkLabel(
            sidebar,
            text="Teacher Power Tools",
            font=("Arial", 18, "bold"),
        ).pack(pady=12)

        for lbl, handler in (
            ("Lesson Plan", self._dlg_lesson),
            ("Grade MD",    self._dlg_grade),
            ("Food Acts",   self._dlg_activity),
            ("Export PDF",  self._export_pdf),
        ):
            customtkinter.CTkButton(sidebar, text=lbl, command=handler).pack(
                fill="x", padx=20, pady=8
            )

        # Main chat area
        self.chatbox = customtkinter.CTkTextbox(self, wrap="word")
        self.chatbox.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        # Input bar
        bar = customtkinter.CTkFrame(self)
        bar.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        bar.grid_columnconfigure(0, weight=1)

        self.input_txt = tk.Text(bar, height=3, wrap="word")
        self.input_txt.grid(row=0, column=0, sticky="ew")
        self.input_txt.bind("<Return>", lambda evt: (self._send(), "break"))

        customtkinter.CTkButton(bar, text="Send", command=self._send).grid(
            row=0, column=1, padx=6
        )

    # ----- Chat mechanics --------------------------------------------
    def _send(self) -> None:
        msg = self.input_txt.get("1.0", tk.END).strip()
        self.input_txt.delete("1.0", tk.END)
        if not msg:
            return

        self.chatbox.insert(tk.END, "{0}: {1}\n".format(self.uid, msg))
        asyncio.run(db_store(self.uid, msg))
        self.executor.submit(self._ai_response, msg)

    def _ai_response(self, msg: str) -> None:
        reply = generate(msg, msg[:128])
        asyncio.run(db_store(self.bot_id, reply))
        self.queue.put("AI: {0}\n".format(reply))

    def _poll_queue(self) -> None:
        try:
            while True:
                self.chatbox.insert(tk.END, self.queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ----- Dialog helpers --------------------------------------------
    def _prompt_dialog(self, title_txt: str) -> str:
        dialog = tk.Toplevel(self)
        dialog.title(title_txt)

        tk.Label(dialog, text=title_txt).pack(padx=6, pady=4)

        entry = tk.Entry(dialog, width=60)
        entry.pack(padx=6, pady=4)

        var = tk.StringVar()

        tk.Button(
            dialog,
            text="OK",
            command=lambda: (var.set(entry.get()), dialog.destroy()),
        ).pack(pady=4)

        self.wait_variable(var)
        return var.get().strip()

    # ----- Sidebar tool dialogs --------------------------------------
    def _dlg_lesson(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Markdown", "*.md")])
        if file_path:
            src = open(file_path, "r", encoding="utf-8").read()
            self.chatbox.insert(tk.END, "[Lesson …]\n")
            self.executor.submit(lambda: self.queue.put(self.lesson_gen.gen(src)))

    def _dlg_grade(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Markdown", "*.md")])
        if file_path:
            src = open(file_path, "r", encoding="utf-8").read()
            self.chatbox.insert(tk.END, "[Grade …]\n")
            self.executor.submit(lambda: self.queue.put(self.grade_gen.gen(src)))

    def _dlg_activity(self) -> None:
        classroom = self._prompt_dialog("Class / Grade")
        foods     = self._prompt_dialog("Favorite Foods")
        recipes   = self._prompt_dialog("Available Recipes")
        extras    = self._prompt_dialog("Extra Prompts")

        self.chatbox.insert(tk.END, "[Activities …]\n")
        self.executor.submit(
            lambda: self.queue.put(self.act_gen.gen(classroom, foods, recipes, extras))
        )

    # ----- PDF exporter ----------------------------------------------
    def _export_pdf(self) -> None:
        text  = self.chatbox.get("1.0", tk.END)
        path  = pdfu.export(text, "chat")
        if path:
            messagebox.showinfo("Saved", path)


# ---------------------------------------------------------------------
#  Daily key-rotation background thread
# ---------------------------------------------------------------------
def _rotate_loop() -> None:
    while True:
        next_midnight = (
            datetime.utcnow()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            + timedelta(days=1)
        )
        time.sleep((next_midnight - datetime.utcnow()).total_seconds())

        global ACTIVE_KEY
        ACTIVE_KEY = _rotate_key()
        logger.info("Daily key rotated")


threading.Thread(target=_rotate_loop, daemon=True).start()

# ---------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        AppGUI("teacher").mainloop()
    except Exception as exc:
        logger.error("Fatal exception: %s", exc)
        sys.exit(1)
