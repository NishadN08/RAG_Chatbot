import base64
import io
import json
import os
import pickle
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import pdfplumber
import requests
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_JSONL = "crawl_3-19-26(2).jsonl"

START_URL = "https://www.sc.fsu.edu"

LOGIN_URL1 = "https://cas.fsu.edu/cas/login?service=https%3a%2f%2fwww.sc.fsu.edu%2flogin"
LOGIN_URL2 = "https://www.sc.fsu.edu/login"

# If your login requires manual 2FA or non-standard steps:
HEADLESS = False                 # set to False to see the browser for manual steps
WAIT_FOR_MANUAL_LOGIN = True     # set True if you want to login manually (requires HEADLESS=False)
MANUAL_LOGIN_MAX_SECONDS = 300   # how long to wait for you to finish manual login

# Domains (hostnames) that are allowed to crawl
ALLOWED_NETLOCS = {"www.sc.fsu.edu", "sc.fsu.edu"}

# Optional: restrict to certain path prefixes on the same site (keeps root '/')
ALLOWED_PATH_PREFIXES = [
    "/people", "/people/faculty", "/people/students", "/people/post-docs", "/people/administration",
    "/graduate", "/graduate/phd", "/graduate//ms/computational-science", "/graduate/application",
    "/undergraduate",
    "/research/faculty", "/research",
    "/undergraduate/bachelor-of-science",
    "/undergraduate/minor", "/undergraduate/courses",
    "/computing","/computing/tech-docs",
    "/",  # include root
]

MUST_CRAWL = [
    "https://www.sc.fsu.edu/people/students",
    "https://www.sc.fsu.edu/faq",
    "https://www.sc.fsu.edu/computing/tech-docs",
    "https://www.sc.fsu.edu/courses/",
    "https://www.sc.fsu.edu/news-and-events/newsletter",
    "https://www.sc.fsu.edu/images/newsletter"
    "https://www.sc.fsu.edu/news-and-events"
    "https://www.sc.fsu.edu/xpos"
    "https://www.sc.fsu.edu/news-and-events/colloquium"
]

ERROR_PATTERNS = [
    "ERR_TOO_MANY_REDIRECTS",
    "This page isn’t working",
    "redirected you too many times",
    "site not working",
    "try deleting your cookies",
]

# EXCLUDE_PARAMS = ["?start=", "?view=", "&view="]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tiff")

MAX_PAGES = 5000
MAX_DEPTH = 20
REQUEST_DELAY = 1.0
PAGE_LOAD_TIMEOUT = 30
MAX_TEXT_CHARS_PER_PAGE = 50000

HEADER_TITLE = "FSU Department of Scientific Computing — Site Archive"

# Crawl only links found inside the body container to avoid nav recursion
CRAWL_LINKS_FROM_BODY_ONLY = True

MAIN_CANDIDATE_SELECTORS = [
    "main", "[role=main]",
    "#content", "#main", "#primary", "#g-main", "#contentarea", "#component", "#page-content",
    ".profile-content", ".content", ".main", ".main-content", ".page-content", ".article", ".article-content",
    ".entry-content", ".item-page", ".component", ".region-content"
]

NOISE_ID_CLASS_RE = re.compile(
    r"(menu|breadcrumb|breadcrumbs|footer|header|sidebar|side-bar|"
    r"topbar|toolbar|sidenav|offcanvas|pager|pagination|tabs|share|social|"
    r"skip|cookie|consent|advert|ad-)",
    re.I,
)
ROLE_NOISE = {"banner", "contentinfo", "complementary"}

# global set to collect unique external links encountered during extraction
unique_links = set()

# ----------------------------
# LOGIN HELPERS
# ----------------------------
def load_cookies(driver, path="cookies.pkl"):
    """
    Load cookies previously saved with save_cookies() into the Selenium driver.
    This helps reuse authenticated sessions between runs.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                # Some cookies contain attributes Selenium rejects (e.g. sameSite); skip them defensively
                continue
        driver.refresh()
    except Exception:
        # non-fatal: continue without cookies
        pass


def save_cookies(driver, path="cookies.pkl"):
    """
    Persist current Selenium cookies to a file.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(driver.get_cookies(), f)
    except Exception:
        pass


def is_logged_in_heuristic(driver):
    """
    Heuristic to determine whether the user is logged in:
    - Checks page HTML for 'logout'/'sign out'
    - Detects presence of a login form
    - Falls back to checking current URL for 'login'
    """
    try:
        html = driver.page_source.lower()
        if "logout" in html or "sign out" in html or "log out" in html:
            return True
        if "login" in html and ("password" in html or 'type="password"' in html):
            return False
    except Exception:
        pass
    return "login" not in driver.current_url.lower()


def perform_login(driver, login_url):
    """
    Navigate to login_url and wait for manual login (DUO/SSO).
    The function is intentionally simple: it loads the page and then waits up to
    MANUAL_LOGIN_MAX_SECONDS for the is_logged_in_heuristic() to become True.
    """
    driver.get(login_url)
    wait_for_ready(driver)

    print(f"[Login] Please log in manually (DUO/SSO). Waiting up to {MANUAL_LOGIN_MAX_SECONDS}s...")
    end = time.time() + MANUAL_LOGIN_MAX_SECONDS
    while time.time() < end:
        if is_logged_in_heuristic(driver):
            print("[Login] Login successful.")
            return
        time.sleep(2)
    raise RuntimeError("Manual login timed out. Increase MANUAL_LOGIN_MAX_SECONDS if DUO takes longer.")





# ----------------------------
# URL / Text Utilities
# ----------------------------
def normalize_url(base: str, link: str) -> str | None:
    """
    Normalize a link relative to `base`:
    - join relative links with base using urljoin
    - drop fragment identifiers (#...)
    - return None for empty link inputs
    """
    if not link:
        return None
    try:
        abs_url = urljoin(base, link)
        parsed = urlparse(abs_url)._replace(fragment="")
        return parsed.geturl()
    except Exception:
        return None


def is_internal(url: str) -> bool:
    """
    Return True if the URL's netloc is within ALLOWED_NETLOCS.
    Defensive: returns False for malformed URLs.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc in ALLOWED_NETLOCS
    except Exception:
        return False


def clean_text(s: str) -> str:
    """
    Collapse whitespace and trim. Returns empty string for None input.
    """
    return re.sub(r"\s+", " ", s or "").strip()


def looks_like_error_page(text: str) -> bool:
    """
    Return True if the page text contains known error patterns (configurable above).
    Treats missing/empty text as suspicious.
    """
    if not text:
        return True
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in ERROR_PATTERNS)


# ----------------------------
# DOM cleaning helpers
# ----------------------------
def remove_nodes_by_selectors(root: Tag, selectors: list[str]):
    """
    Remove nodes matching CSS selectors from root tag (defensive: ignores errors).
    """
    if not isinstance(root, Tag):
        return
    to_remove = []
    for sel in selectors:
        try:
            for t in root.select(sel):
                if isinstance(t, Tag):
                    to_remove.append(t)
        except Exception:
            continue
    for t in to_remove:
        try:
            t.decompose()
        except Exception:
            pass


def remove_nodes_by_role(root: Tag, roles: set[str]):
    """
    Remove nodes whose role attribute matches any role in `roles`.
    """
    if not isinstance(root, Tag):
        return
    to_remove = []
    for t in list(root.find_all(True)):
        if not isinstance(t, Tag):
            continue
        role_attr = t.attrs.get("role", None)
        if not role_attr:
            continue
        role = " ".join(role_attr) if isinstance(role_attr, (list, tuple)) else str(role_attr)
        if role and role.lower() in roles:
            to_remove.append(t)
    for t in to_remove:
        try:
            t.decompose()
        except Exception:
            pass


def remove_nodes_by_id_class_pattern(root: Tag, pattern: re.Pattern):
    """
    Remove nodes whose id or class matches `pattern`.
    """
    if not isinstance(root, Tag):
        return
    to_remove = []
    for el in list(root.find_all(True)):
        if not isinstance(el, Tag):
            continue
        try:
            classes = " ".join(el.get("class", [])) if isinstance(el.get("class"), list) else ""
        except Exception:
            classes = ""
        try:
            el_id = el.get("id") or ""
        except Exception:
            el_id = ""
        test = f"{classes} {el_id}"
        if pattern.search(test):
            to_remove.append(el)
    for el in to_remove:
        try:
            el.decompose()
        except Exception:
            pass


def clean_main_container(main: Tag) -> Tag:
    """
    Remove scripts, styles, small nav elements, and hidden elements from the main container.
    Returns the cleaned Tag (or the same object if nothing to do).
    """
    if not main:
        return main
    remove_nodes_by_selectors(main, ["script", "style", "noscript", "template", "iframe"])
    for nav in main.select("nav, .breadcrumb, .breadcrumbs, .sidebar, .top, .prefooter"):
        try:
            nav.decompose()
        except Exception:
            pass
    for el in main.select('[style*="display:none"], [aria-hidden="true"]'):
        try:
            el.decompose()
        except Exception:
            pass
    return main


def remove_html_comments(soup: BeautifulSoup):
    """
    Remove HTML comments from the BeautifulSoup tree.
    """
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        try:
            comment.extract()
        except Exception:
            pass


def remove_navigation_blocks(root: Tag):
    """
    Try to remove common navigation module blocks or other repeated modules identified
    by known comment text or selectors.
    """
    if not isinstance(root, Tag):
        return root

    for comment_text in ["module breadcrumbs", "module sidebar", "module top", "module prefooter"]:
        for comment in root.find_all(string=lambda text: isinstance(text, type(root.string)) and comment_text in text):
            try:
                parent = comment.find_parent()
                if parent:
                    parent.decompose()
            except Exception:
                pass

    for sel in ["#content .s3-w", ".s3-nv", ".mod-custom"]:
        for el in root.select(sel):
            try:
                el.decompose()
            except Exception:
                pass

    return root


def strip_noise(root: Tag) -> Tag:
    """
    High-level noise-stripping (header/footer/roles/id/class patterns).
    """
    if not isinstance(root, Tag):
        return root
    remove_nodes_by_selectors(root, ["script", "style", "noscript", "template", "svg", "form", "iframe"])
    remove_nodes_by_selectors(root, ["header", "footer", "aside"])
    remove_nodes_by_role(root, ROLE_NOISE)
    remove_nodes_by_id_class_pattern(root, NOISE_ID_CLASS_RE)
    return root


# ----------------------------
# Picking and extracting main content
# ----------------------------
def pick_main_container(soup: BeautifulSoup) -> Tag:
    """
    Choose the most likely 'main' content container:
    1) match known candidate selectors
    2) choose the largest article/section/div content block
    3) fallback to <body> or first tag
    Always returns a Tag (never None).
    """
    candidates = []
    for sel in MAIN_CANDIDATE_SELECTORS:
        try:
            found = soup.select(sel)
        except Exception:
            found = []
        for el in found:
            if isinstance(el, Tag) and el.get_text(strip=True):
                candidates.append(el)
    if candidates:
        return max(candidates, key=lambda e: len(e.get_text(" ", strip=True)))

    best = None
    best_len = 0
    for el in soup.find_all(["article", "section", "div"]):
        if not isinstance(el, Tag):
            continue
        try:
            classes = " ".join(el.get("class", [])) if isinstance(el.get("class"), list) else ""
        except Exception:
            classes = ""
        try:
            el_id = el.get("id") or ""
        except Exception:
            el_id = ""
        if NOISE_ID_CLASS_RE.search(f"{classes} {el_id}"):
            continue
        tlen = len(el.get_text(" ", strip=True))
        if tlen > best_len:
            best, best_len = el, tlen
    if isinstance(best, Tag):
        return best

    if isinstance(soup.body, Tag):
        return soup.body
    first_tag = soup.find(True)
    if isinstance(first_tag, Tag):
        return first_tag
    return soup.new_tag("div")


def is_valid_link(href: str) -> bool:
    """
    Lightweight filter that keeps full http(s) links while excluding common unwanted patterns.
    """
    if not href:
        return False
    href = href.strip().lower()
    if not href.startswith("http"):
        return False
    blocked_patterns = ["ldap", "login", "logout", "cmd=", "session", "token", "phpldapadmin"]
    return not any(b in href for b in blocked_patterns)


def extract_text_with_links(node: Tag) -> str:
    """
    Extracts text from a node while replacing <a> text content with their URLs where appropriate,
    and also collects unique external links into the global unique_links set.
    Uses recursion but avoids repeated inclusion of the same block-level text.
    """
    if not isinstance(node, Tag):
        return ""
    if node.name in ["script", "style", "noscript", "template", "svg", "iframe", "form", "img"]:
        return ""

    parts = []
    
    # Process block-level tags to avoid duplication
    if node.name in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "code"]:
        for a in node.find_all("a", href=True):
            href = a["href"].strip()
            if is_valid_link(href):
                try:
                    a.string = href
                except Exception:
                    pass
        
        text = clean_text(node.get_text(separator=" "))
        if text:
            parts.append(text)

                # Collect unique links but DO NOT recurse further into this block
        for a in node.find_all("a", href=True):
            href = a["href"].strip()
            if is_valid_link(href) and href not in unique_links:
                unique_links.add(href)
        
        return "\n".join(parts)

    for child in node.children:
        if isinstance(child, Tag):
            txt = extract_text_with_links(child)
            if txt:
                parts.append(txt)
        else:
            # raw text node
            if child.string:
                clean = clean_text(child.string)
                if clean:
                    parts.append(clean)

    return "\n".join(parts)

# ----------------------------
# Joomla & Cloudflare email decoding
# ----------------------------
def decode_cloudflare_email(cf_str: str) -> str | None:
    """
    Decode Cloudflare's email obfuscation (data-cfemail).
    Returns decoded email or None on failure.
    """
    if not cf_str:
        return None
    try:
        r = bytes.fromhex(cf_str)
        key = r[0]
        decoded = bytes([b ^ key for b in r[1:]]).decode("utf-8")
        return decoded
    except Exception:
        return None


def decode_joomla_email(tag: Tag) -> str | None:
    """
    Decode Joomla's hidden email constructs that may embed base64/hex attributes.
    Attempts multiple heuristics (attribute names in your original script).
    """
    if not tag:
        return None

    first = tag.get("d2VibWFzdGVy")
    last = tag.get("c2MuZnN1LmVkdQ==")
    if first and last:
        try:
            local = base64.b64decode(first).decode("utf-8")
            domain = base64.b64decode(last).decode("utf-8")
            return f"{local}@{domain}"
        except Exception:
            pass

    text_val = tag.get("d2VibWFzdGVyQHNjLmZzdS5lZHU=") or tag.string
    if text_val:
        try:
            b = bytes.fromhex(text_val)
            key = b[0]
            decoded = bytes([x ^ key for x in b[1:]]).decode("utf-8")
            return decoded
        except Exception:
            try:
                decoded = base64.b64decode(text_val).decode("utf-8")
                return decoded
            except Exception:
                return None

    return None


# ----------------------------
# PDF text extraction
# ----------------------------
def extract_pdf_text(url: str) -> str:
    """
    Download a PDF and extract text using pdfplumber.
    Returns empty string on any failure (non-fatal).
    """
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting PDF from {url}: {e}")
        return ""

# ----------------------------
# CONTENT EXTRACTION
# ----------------------------


def extract_content(html: str, base_url: str) -> dict:
    """
    Parse HTML (raw), select the main container, clean it up, and extract:
    - title
    - combined text (HTML main + any linked PDF text)
    - anchor text list (from the main)
    - emails discovered (Cloudflare + mailto + Joomla)
    - links for BFS crawling (internal links found inside main or whole page)
    - external profile links (LinkedIn / Google Scholar)
    - pdf links discovered on page
    """
    soup = BeautifulSoup(html, "lxml")
    remove_html_comments(soup)

    title = clean_text(soup.title.get_text()) if soup.title else ""

    mailto_emails = []
    for a in soup.find_all("a", href=True):
        try:
            href = a["href"]
        except Exception:
            continue
        if isinstance(href, str) and href.lower().startswith("mailto:"):
            mail = href.split("mailto:", 1)[1].split("?")[0]
            if mail:
                mailto_emails.append(mail)

    emails = sorted(set(mailto_emails))

    # Main container selection and cleanup
    main = pick_main_container(soup)
    main = clean_main_container(main)
    main = remove_navigation_blocks(main)

    # Body text and anchors (body-only)
    page_text = extract_text_with_links(main)

    anchor_texts = []
    for a in main.find_all("a"):
        if not isinstance(a, Tag):
            continue
        t = clean_text(a.get_text(" "))
        href = a.get("href", "")
        if t and href and not NOISE_ID_CLASS_RE.search(" ".join(a.get("class", []))):
            anchor_texts.append(t)
    anchor_texts = list(dict.fromkeys(anchor_texts))

    # Collect links for BFS (respect toggle to only use body links)
    link_scope = main if CRAWL_LINKS_FROM_BODY_ONLY else soup
    internal_links = set()
    for a in link_scope.find_all("a", href=True):
        try:
            href = normalize_url(base_url, a["href"])
        except Exception:
            href = None
        if href and is_internal(href):
            internal_links.add(href)

    # Scan whole page for PDFs and external profiles (LinkedIn / Google Scholar)
    pdf_links = []
    external_profile_links = []
    pdf_texts = []

    for a in soup.find_all("a", href=True):
        try:
            abs_url = normalize_url(base_url, a["href"].strip())
        except Exception:
            abs_url = None
        if not abs_url:
            continue

        if abs_url == "https://www.linkedin.com/edu/school?id=18100":
            continue

        if abs_url.lower().endswith(".pdf"):
            pdf_links.append(abs_url)
            pdf_text = extract_pdf_text(abs_url)
            if pdf_text:
                pdf_texts.append(pdf_text)

        netloc = urlparse(abs_url).netloc.lower()
        if "scholar.google.com" in netloc or "linkedin.com" in netloc:
            external_profile_links.append(abs_url)

    combined_text = page_text #" ".join([page_text] + pdf_texts).strip()

    return {
        "title": title,
        "text": combined_text,
        "anchor_texts": anchor_texts,
        "emails": emails,
        "links": sorted(internal_links),
        "external_profile_links": sorted(set(external_profile_links)),
        "pdf_links": sorted(set(pdf_links)),
    }


def wait_for_ready(driver):
    """
    Wait for document.readyState == 'complete' and add a small sleep for JS to settle.
    """
    WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(0.5)


def allowed_by_robots(url: str, rp: robotparser.RobotFileParser) -> bool:
    """
    Check robots.txt using a RobotFileParser if available. If parser fails, default to True.
    """
    try:
        return rp.can_fetch("*", url)
    except Exception:
        return True


# ----------------------------
# MAIN CRAWLER (BFS)
# ----------------------------
def should_skip_url(url: str) -> bool:
    """
    Additional site-specific skip rules (photo gallery, user directories, web links component).
    Only apply these when the netloc contains sc.fsu.edu.
    """
    skip_prefixes = [
        "sc.fsu.edu/news-and-events/photo-gallery/",
        "sc.fsu.edu/~dduke/",
        "sc.fsu.edu/component/weblinks/",
    ]
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    if "sc.fsu.edu" in netloc:
        for prefix in skip_prefixes:
            if path.startswith("/" + prefix.split("/", 1)[1]):
                return True
    return False



def extract_pagination_links(base_url: str, html: str) -> list[str]:
    """
    Extract pagination links from Joomla-style pages (e.g., /computing/tech-docs).
    These links typically appear as '?start=15', '?start=30', etc.
    """
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("div.pagination a, ul.pagination a"):
        href = a.get("href")
        if href and not href.startswith("#"):
            links.append(urljoin(base_url, href))
    return links

def remove_prefix(url: str) -> str:
    """Normalize URL by removing protocol, www, and trailing slash."""
    if not url:
        return ""
    url = re.sub(r"^https?://", "", url.strip())
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/").lower()


def crawl(driver, start_url: str):
    """
    Breadth-first crawl starting from start_url, respecting MAX_PAGES, MAX_DEPTH and robots.txt.
    Writes JSONL records to OUTPUT_JSONL. Uses Selenium driver to render JS pages.
    """
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass

    seen = set()
    q = deque()
    q.append((start_url, 0))

    for link in MUST_CRAWL:
        q.append((link, 0))

    out_f = open(OUTPUT_JSONL, "w", encoding="utf-8")
    pages_crawled = 0

    try:
        while q and pages_crawled < MAX_PAGES:
            url, depth = q.popleft()
            norm_url = remove_prefix(url)
        
            # Skip duplicates based on normalized form
            if norm_url in seen:
                continue
            seen.add(norm_url)

            if depth > MAX_DEPTH:
                continue
            if not is_internal(url):
                continue

            if not allowed_by_robots(url, rp):
                print(f"[robots.txt] Skipping: {url}")
                continue

            try:
                time.sleep(REQUEST_DELAY)
                # Attempt to reuse cookies (if present)
                load_cookies(driver)

                driver.get(url)
                wait_for_ready(driver)
                html = driver.page_source

                # Pagination link extraction
                pagination_links = extract_pagination_links(url, html)
                for plink in pagination_links:
                    norm_plink = normalize_url(url, plink)
                    if norm_plink and norm_plink not in seen:
                        q.append((norm_plink, depth))  # same depth
                        print(f"[Pagination] Queued next page: {norm_plink}")

                data = extract_content(html, url)

                # If the URL itself is a PDF, override extracted text with PDF content
                if url.lower().endswith(".pdf"):
                    pdf_text = extract_pdf_text(url)
                    if pdf_text:
                        data["text"] = pdf_text
                
                
                record = {
                    "url": url,
                    "depth": depth,
                    "title": data["title"],
                    "text": data["text"],
                    "emails": data["emails"],
                    "anchor_texts": data["anchor_texts"],
                    "out_links": data["links"],
                    "external_profile_links": data["external_profile_links"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pages_crawled += 1
                print(f"[{pages_crawled}] {url} (links: {len(data['links'])}, emails: {len(data['emails'])})")

                # Enqueue discovered links
                for link in data["links"]:
                    if link.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    if should_skip_url(link):
                        continue
                    if looks_like_error_page(data["text"]):
                        print(f"[Skipping site: Error page] {url}")
                        continue
                    if link not in seen:
                        q.append((link, depth + 1))

            except (TimeoutException, WebDriverException) as e:
                print(f"[ERROR] {url}: {e}")
                continue
            except Exception as e:
                print(f"[UNEXPECTED ERROR] {url}: {e}")
                continue

    finally:
        out_f.close()
        driver.quit()

    print(f"\nDone. Saved {pages_crawled} pages to {OUTPUT_JSONL}")


if __name__ == "__main__":
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
        # keep the browser open after script ends (helpful while debugging)
    chrome_options.add_experimental_option("detach", True)

    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1400,1000")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--lang=en-US,en")

    # Optional: reduce "automation" detection noise
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    
    try:
        # Login first
        perform_login(driver, LOGIN_URL1)
        if is_logged_in_heuristic(driver):
            input("After you have manually logged in successfully, press Enter to continue...")
            print("[Login] CAS and sc.fsu.edu login successful ✅")
            
        # Crawl using logged-in session
        crawl(driver, START_URL)

    finally:
        driver.quit()    
        
