# OpenAI GPT Store executor

import os
import time
import logging
import json
from pathlib import Path

import io
import base64
from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from asbench.generation.responses.base import PlatformExecutor


logger = logging.getLogger(__name__)

class OpenAIExecutor(PlatformExecutor):
    def __init__(
        self,
        credential: str,
        debug: bool = False,
    ):
        super().__init__(None, debug)
        self.session_file = credential
        self.headless = False
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._is_logged_in = False

    def setup(self):
        """Initialize browser and handle authentication"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        # load existing session (login info) if available
        if os.path.exists(self.session_file):
            with open(self.session_file, "r") as f:
                storage_state = json.load(f)
            self.context = self.browser.new_context(storage_state=storage_state)
        else:
            self.context = self.browser.new_context()

        self.page = self.context.new_page()
        if not self._check_logged_in():
            self._login_manual()

    def execute(self, agent_metadata: dict, query: str) -> tuple[str, float, int]:
        if not self._is_logged_in:
            return "Error: Not logged in to ChatGPT", 0.0, 400

        gpt_url = agent_metadata.get("agent_url", "")
        if not gpt_url:
            return "Error: No GPT URL found in metadata", 0.0, 400

        if self.debug:
            logger.info(f"Executing agent {agent_metadata['agent_id']}")

        self.page.goto(gpt_url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)

        if gpt_url not in self.page.url:
            return f"Error - GPT not found or inaccessible at {gpt_url}", 0.0, 404

        start_time = time.time()
        try:
            response = self._send_message(query)

            while response.get("text", "") == "Our systems have detected unusual activity coming from your system. Please try again later.":
                logger.warning("Rate limited by ChatGPT, waiting 5 mins before retrying...")
                time.sleep(60 * 5)
                response = self._send_message(query)

            response_str = f"Text: {response.get('text', '')}, Files: {response.get('files', [])}, Images: {response.get('images', [])}"
            return response_str, time.time() - start_time, 200

        except Exception as e:
            if "modal-conversation-history-rate-limit" in str(e).lower():
                logger.warning("Rate limited by ChatGPT, waiting 5 mins before retrying...")
                time.sleep(60 * 5)
                return self.execute(agent_metadata, query)

            error = f"Exception during web automation: {e}"
            logger.error(error)
            return error, time.time() - start_time, 500

    def teardown(self):
        """Clean up browser resources"""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _check_logged_in(self) -> bool:
        """Check if already logged in to ChatGPT"""
        try:
            self.page.goto("https://chatgpt.com/", wait_until="networkidle")
            time.sleep(2)

            if "chatgpt.com/auth" in self.page.url or self.page.query_selector(
                "button:has-text('Log in')"
            ):
                return False
            self._is_logged_in = True
            return True
        except Exception as _:
            return False

    def _login_manual(self):
        """Navigate to ChatGPT and wait for manual login"""
        print("\n" + "=" * 60)
        print("CHATGPT LOGIN REQUIRED")
        print("=" * 60)
        print("Opening browser for manual login...")
        self.page.goto("https://chatgpt.com/")
        print("Please log in to ChatGPT in the browser window.")
        print("Press Enter here after you've logged in successfully...")
        print("=" * 60 + "\n")

        input()

        # save session meta
        storage_state = self.context.storage_state()
        with open(self.session_file, "w") as f:
            json.dump(storage_state, f)
        print(f"✓ Session saved to {self.session_file}")
        self._is_logged_in = True

    def _send_message(self, message: str, timeout: int = 120) -> str | None:
        """Send message and wait for response"""
        message = message.strip()
        message = message.replace("\n", " ")  # avoid multi-line input issues
        # Find input element
        input_selectors = [
            'div.ProseMirror[contenteditable="true"]#prompt-textarea',
            'textarea[name="prompt-textarea"]',
            'div[contenteditable="true"][id="prompt-textarea"]',
        ]

        input_element = None
        for selector in input_selectors:
            try:
                self.page.wait_for_selector(selector, timeout=10000)
                input_element = self.page.query_selector(selector)
                if input_element:
                    break
            except PlaywrightTimeout:
                continue

        if not input_element:
            logger.error("Could not find message input box")
            return None

        # Type message
        if input_element.get_attribute("contenteditable") == "true":
            input_element.click()
            time.sleep(0.3)
            self.page.keyboard.press("Control+A")
            self.page.keyboard.press("Backspace")
            self.page.keyboard.type(message, delay=50)
        else:
            self.page.fill(input_selectors[1], message)

        time.sleep(0.5)

        # Click send button
        send_button_selector = (
            'button#composer-submit-button[data-testid="send-button"]'
        )
        send_button = self.page.query_selector(send_button_selector)

        if send_button:
            send_button.click()
        else:
            self.page.press(input_selectors[0], "Enter")

        # Wait for response
        return self._wait_for_response(timeout)

    def _wait_for_response(self, timeout: int = 120) -> None:
        """
        Wait until the assistant is fully finished
        (voice button is visible again).
        """

        # Wait until generation actually starts (best effort)
        try:
            self.page.wait_for_selector(
                'button:has-text("Stop")',
                state="visible",
                timeout=10_000
            )
        except Exception:
            pass

        # Now wait for the composer voice button to appear
        try:
            self.page.wait_for_selector(
                'button[aria-label="Start Voice"]',
                state="visible",
                timeout=int(timeout * 1000)
            )
        except Exception:
            pass

        return self._extract_last_turn()

    def _extract_last_turn(self):
        selectors = [
            'section[data-turn="assistant"]:last-of-type',   # scopes to full section
            '[data-message-author-role="assistant"]',
        ]

        turn = None
        for selector in selectors:
            result = self.page.query_selector_all(selector)
            if result:
                turn = result[-1]
                break

        if not turn:
            logger.error("No assistant turn found with any known selector")
            return None

        return {
            "text": self._extract_turn_text(turn),
            "images": self._extract_turn_images(turn),
            "files": self._extract_turn_files(turn),
        }
    
    def _extract_turn_text(self, turn):
        texts = []
        # Scope down to message role div for text only
        message_div = turn.query_selector('[data-message-author-role="assistant"]')
        search_root = message_div if message_div else turn

        for md in search_root.query_selector_all('.markdown'):
            t = md.inner_text().strip()
            if t:
                texts.append(t)

        if not texts and search_root:
            fallback = search_root.inner_text().strip()
            if fallback:
                texts.append(fallback)
                logger.warning("Used inner_text() fallback for text extraction")

        return "\n\n".join(texts)

    def _extract_turn_images(self, turn):
        seen = set()

        for img in turn.query_selector_all('img[src]'):
            src = img.get_attribute("src")
            if not src or "/backend-api/estuary/content" not in src:
                continue
            if src in seen:
                continue
            seen.add(src)

            r = self.page.context.request.get(src)
            if not r.ok:
                continue

            buf = io.BytesIO()
            image = Image.open(io.BytesIO(r.body()))
            image.thumbnail((128, 128), Image.LANCZOS)
            image.convert("RGB").save(buf, format="JPEG", quality=10)

            b64 = base64.b64encode(buf.getvalue()).decode()
            return [f"data:image/jpeg;base64,{b64}"] # only return first image
        return []

    def _extract_turn_files(self, turn):
        files = []
        KNOWN_EXTENSIONS = {".xlsx", ".pdf", ".csv", ".docx", ".pptx", ".zip", ".png", ".jpg"}
        mime_map = {
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "pdf":  "application/pdf",
            "csv":  "text/csv",
            "zip":  "application/zip",
        }

        for card in turn.query_selector_all('div.group'):
            filename_el = card.query_selector('div.truncate')
            if not filename_el:
                continue

            filename = filename_el.inner_text().strip()
            if not filename or Path(filename).suffix.lower() not in KNOWN_EXTENSIONS:
                continue

            download_btn = card.query_selector('button')
            if not download_btn:
                continue

            with self.page.expect_download(timeout=15_000) as download_info:
                download_btn.click()

            download = download_info.value
            tmp_path = f"/tmp/{filename}"
            download.save_as(tmp_path)

            with open(tmp_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            ext = Path(filename).suffix.lstrip(".").lower()
            mime = mime_map.get(ext, "application/octet-stream")

            files.append({
                "filename": filename,
                "data": f"data:{mime};base64,{b64}",
                "extension": ext
            })

        return files