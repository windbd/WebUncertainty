"""Small Playwright wrapper used by the ReAct agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


_AGENT_ID_ATTRIBUTE = "data-react-agent-id"
_INTERACTIVE_SELECTOR = ",".join(
    (
        "a[href]",
        "button",
        "input",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
        "[role='textbox']",
        "[role='checkbox']",
        "[role='radio']",
        "[role='combobox']",
        "[contenteditable='true']",
    )
)
_RISKY_LABEL_RE = re.compile(
    r"(?:\bbuy\s+now\b|\bplace\s+order\b|\bpay(?:\s+now)?\b|"
    r"\bpurchase\b|\bsend\b|\bpublish\b|\bpost\s+(?:comment|reply)\b|"
    r"\bdelete\b|\bremove\s+(?:account|profile)\b|"
    r"\bconfirm\s+(?:booking|purchase|order|payment)\b|"
    r"\bbook\s+now\b|\breserve\b|\btransfer\b|\blog\s*in\b|"
    r"\bsign\s*in\b|付款|支付|购买|下单|发送|发布|删除|登录|预订|转账)",
    re.IGNORECASE,
)


class BrowserError(RuntimeError):
    """Raised when a browser action cannot be executed safely."""


@dataclass(frozen=True)
class PageElement:
    """One interactive element exposed to the model."""

    element_id: int
    tag: str
    role: str
    input_type: str
    name: str
    href: str
    disabled: bool

    def render(self) -> str:
        parts = [f"[{self.element_id}]", self.tag]
        if self.role:
            parts.append(f"role={self.role}")
        if self.input_type:
            parts.append(f"type={self.input_type}")
        if self.name:
            parts.append(f"name={json.dumps(self.name, ensure_ascii=False)}")
        if self.href:
            parts.append(f"url={self.href}")
        if self.disabled:
            parts.append("disabled=true")
        return " ".join(parts)


@dataclass(frozen=True)
class PageObservation:
    """A bounded text observation of the current page."""

    url: str
    title: str
    text: str
    elements: tuple[PageElement, ...]

    @property
    def element_ids(self) -> frozenset[int]:
        return frozenset(item.element_id for item in self.elements)

    @property
    def grounded_urls(self) -> frozenset[str]:
        return frozenset(item.href for item in self.elements if item.href)

    def element(self, element_id: int) -> PageElement:
        for item in self.elements:
            if item.element_id == element_id:
                return item
        raise BrowserError(f"element [{element_id}] is not in the current page snapshot")

    def render(self) -> str:
        controls = "\n".join(item.render() for item in self.elements)
        if not controls:
            controls = "(none)"
        return (
            f"URL: {self.url}\n"
            f"Title: {self.title}\n\n"
            f"Visible page text:\n{self.text or '(none)'}\n\n"
            f"Interactive elements:\n{controls}"
        )


class BrowserSession:
    """Own one Playwright browser context and execute grounded actions."""

    def __init__(
        self,
        *,
        headless: bool = True,
        browser_name: str = "chromium",
        slow_mo: int = 0,
        action_wait_ms: int = 500,
        timeout_ms: int = 10_000,
        cdp_endpoint: str | None = None,
        reuse_tab: bool = False,
    ) -> None:
        if browser_name not in {"chromium", "msedge"}:
            raise ValueError("browser_name must be 'chromium' or 'msedge'")
        cdp_endpoint = cdp_endpoint.strip() if cdp_endpoint else ""
        if cdp_endpoint:
            _validate_http_url(cdp_endpoint)
        self.headless = headless
        self.browser_name = browser_name
        self.slow_mo = max(0, slow_mo)
        self.action_wait_ms = max(0, action_wait_ms)
        self.timeout_ms = max(1, timeout_ms)
        self.cdp_endpoint = cdp_endpoint
        self.reuse_tab = reuse_tab
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._owns_browser = True
        self.page: Page | None = None
        self._last_url = ""

    def __enter__(self) -> BrowserSession:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def current_url(self) -> str:
        return self.page.url if self.page is not None else self._last_url

    def open(self) -> None:
        if self.page is not None:
            return
        self._playwright = sync_playwright().start()
        try:
            if self.cdp_endpoint:
                self._attach_cdp()
            else:
                self._launch()
            self.page.set_default_timeout(self.timeout_ms)
        except Exception:
            self.close()
            raise

    def _launch(self) -> None:
        channel = "msedge" if self.browser_name == "msedge" else None
        self._browser = self._playwright.chromium.launch(
            channel=channel,
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self._owns_browser = True
        self._context = self._browser.new_context(
            viewport={"width": 1440, "height": 1000},
        )
        self.page = self._context.new_page()

    def _attach_cdp(self) -> None:
        """Attach to an already-running Edge/Chromium instead of launching one.

        Reuses that browser's existing context, so its cookies and login
        state carry over. The instance must already be running with
        ``--remote-debugging-port``; a normally-launched Edge cannot have
        remote debugging turned on after the fact, and ``edge://inspect``'s
        dynamic port is not attachable this way.
        """
        try:
            self._browser = self._playwright.chromium.connect_over_cdp(
                self.cdp_endpoint,
                timeout=self.timeout_ms,
            )
        except Exception as exc:
            raise BrowserError(
                f"could not attach to {self.cdp_endpoint!r}: {exc}. Close every "
                "Edge window first, then relaunch with "
                "'msedge.exe --remote-debugging-port=9222'."
            ) from exc
        self._owns_browser = False
        if not self._browser.contexts:
            raise BrowserError(
                "the attached browser has no open window/context to reuse"
            )
        self._context = self._browser.contexts[0]
        if self.reuse_tab and self._context.pages:
            self.page = self._context.pages[-1]
        else:
            self.page = self._context.new_page()

    def close(self) -> None:
        if self.page is not None:
            self._last_url = self.page.url
        if self._owns_browser:
            if self._context is not None:
                self._context.close()
            if self._browser is not None:
                self._browser.close()
        elif self.page is not None and not self.reuse_tab:
            # Attached to someone else's already-running browser: never close
            # their shared context or the browser itself, only the fresh tab
            # this session opened.
            try:
                self.page.close()
            except Exception:
                pass
        if self._playwright is not None:
            self._playwright.stop()
        self.page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._owns_browser = True

    def start(self, url: str) -> None:
        page = self._require_page()
        _validate_http_url(url)
        page.goto(url, wait_until="domcontentloaded")
        self._settle()

    def observe(
        self,
        *,
        max_text_chars: int = 8_000,
        max_elements: int = 80,
    ) -> PageObservation:
        page = self._require_page()
        max_text_chars = max(1, max_text_chars)
        max_elements = max(1, max_elements)
        raw = page.evaluate(
            """
            ({ attribute, selector, maxElements }) => {
              document.querySelectorAll(`[${attribute}]`).forEach(
                element => element.removeAttribute(attribute)
              );

              const visible = element => {
                const style = window.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.visibility !== 'hidden' &&
                  style.display !== 'none' &&
                  rect.width > 0 && rect.height > 0;
              };

              const clean = value => String(value || '')
                .replace(/\\s+/g, ' ')
                .trim()
                .slice(0, 300);

              const elements = [];
              for (const element of document.querySelectorAll(selector)) {
                if (!visible(element) || elements.length >= maxElements) continue;
                const id = elements.length + 1;
                element.setAttribute(attribute, String(id));
                const tag = element.tagName.toLowerCase();
                const type = clean(element.getAttribute('type')).toLowerCase();
                let name = element.getAttribute('aria-label') ||
                  element.getAttribute('placeholder') ||
                  element.getAttribute('title') ||
                  element.innerText ||
                  element.getAttribute('name') || '';
                if (!name && type !== 'password') name = element.value || '';
                const href = tag === 'a' ? element.href : '';
                elements.push({
                  element_id: id,
                  tag,
                  role: clean(element.getAttribute('role')).toLowerCase(),
                  input_type: type,
                  name: clean(name),
                  href: clean(href),
                  disabled: Boolean(element.disabled) ||
                    element.getAttribute('aria-disabled') === 'true',
                });
              }

              return {
                title: document.title || '',
                text: document.body ? document.body.innerText || '' : '',
                elements,
              };
            }
            """,
            {
                "attribute": _AGENT_ID_ATTRIBUTE,
                "selector": _INTERACTIVE_SELECTOR,
                "maxElements": max_elements,
            },
        )
        text = _clean_text(str(raw.get("text") or ""), max_text_chars)
        elements = tuple(PageElement(**item) for item in raw.get("elements", []))
        return PageObservation(
            url=page.url,
            title=_clean_text(str(raw.get("title") or ""), 500),
            text=text,
            elements=elements,
        )

    def execute(self, action, observation: PageObservation) -> None:
        """Execute one already validated agent action."""
        page = self._require_page()
        action_name = action.action

        if action_name in {"click", "type", "press", "select"}:
            if action.target is None:
                raise BrowserError(f"{action_name} requires an element target")
            element = observation.element(action.target)
            self._enforce_read_only(action_name, element, action.value)
            locator = page.locator(
                f'[{_AGENT_ID_ATTRIBUTE}="{action.target}"]'
            )
            if locator.count() != 1:
                raise BrowserError(
                    f"element [{action.target}] changed after the page snapshot"
                )

            if action_name == "click":
                locator.click()
            elif action_name == "type":
                locator.fill(action.value)
            elif action_name == "press":
                locator.press(action.value)
            else:
                try:
                    locator.select_option(label=action.value)
                except Exception:
                    locator.select_option(value=action.value)
        elif action_name == "scroll":
            delta = 700 if action.value == "down" else -700
            page.mouse.wheel(0, delta)
        elif action_name == "goto":
            _validate_http_url(action.value)
            if action.value not in observation.grounded_urls:
                raise BrowserError("goto URL is not present in the current snapshot")
            page.goto(action.value, wait_until="domcontentloaded")
        elif action_name == "back":
            page.go_back(wait_until="domcontentloaded")
        elif action_name == "wait":
            page.wait_for_timeout(1_000)
        else:
            raise BrowserError(f"browser cannot execute action {action_name!r}")
        self._settle()

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._require_page().screenshot(path=str(path), full_page=False)

    def _settle(self) -> None:
        page = self._require_page()
        try:
            page.wait_for_load_state("domcontentloaded", timeout=3_000)
        except PlaywrightTimeoutError:
            pass
        if self.action_wait_ms:
            page.wait_for_timeout(self.action_wait_ms)

    @staticmethod
    def _enforce_read_only(
        action_name: str,
        element: PageElement,
        value: str,
    ) -> None:
        if element.disabled:
            raise BrowserError(f"element [{element.element_id}] is disabled")
        if element.input_type in {"password", "file"}:
            raise BrowserError(
                f"{element.input_type} inputs are outside the read-only MVP"
            )
        label = f"{element.name} {value if action_name == 'press' else ''}"
        if action_name in {"click", "press"} and _RISKY_LABEL_RE.search(label):
            raise BrowserError(
                "blocked a potentially consequential action in read-only mode"
            )

    def _require_page(self) -> Page:
        if self.page is None:
            raise BrowserError("browser session is not open")
        return self.page


def _validate_http_url(url: str) -> None:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrowserError("URL must be an absolute HTTP(S) URL")


def _clean_text(value: str, limit: int) -> str:
    lines = []
    for raw_line in value.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    cleaned = "\n".join(lines)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n...[truncated]"
