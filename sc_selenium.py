import json
import re
import requests
import time
import pickle
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from datetime import datetime
import base64
import os
import io

from bs4 import BeautifulSoup , Comment
from bs4.element import Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF
from pdfminer.high_level import extract_text  
# from pdf2image import convert_from_path
import pdfplumber


###############################################################################
# CONFIG
###############################################################################
START_URL = "https://www.sc.fsu.edu"



LOGIN_URL1 = "https://cas.fsu.edu/cas/login?service=https%3a%2f%2fwww.sc.fsu.edu%2flogin"   # update if your login page differs
LOGIN_URL2 = "https://www.sc.fsu.edu/login"


# If your login requires manual 2FA or non-standard steps:
HEADLESS = False                 # set to False to see the browser for manual steps
WAIT_FOR_MANUAL_LOGIN = True   # set True if you want to login manually (requires HEADLESS=False)
MANUAL_LOGIN_MAX_SECONDS = 300  # how long to wait for you to finish manual login



# Domains (hostnames) that are allowed
ALLOWED_NETLOCS = {"www.sc.fsu.edu", "sc.fsu.edu"}
# If you want to include subdomains (e.g., people.sc.fsu.edu), add them:
# ALLOWED_NETLOCS.update({"people.sc.fsu.edu"})

# Optional: restrict to certain path prefixes on the same site
ALLOWED_PATH_PREFIXES = [
    "/people","/people/faculty","/people/students","/people/post-docs","/people/administration",
    "/graduate","/graduate/phd","/graduate//ms/computational-science","/graduate/application",
    "/undergraduate",
    "/research/faculty","/research",
    "/undergraduate/bachelor-of-science",
    "/undergraduate/minor","/undergraduate/courses",
    "/research",
    "/",  # keep root so the homepage and general pages are included
]

MUST_CRAWL = [
    "https://www.sc.fsu.edu/people/students",
    "https://www.sc.fsu.edu/faq",
]

ERROR_PATTERNS = [
    "ERR_TOO_MANY_REDIRECTS",
    "This page isn’t working",
    "redirected you too many times",
    "site not working",
    "try deleting your cookies",
]

               
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".tiff")

OUTPUT_JSONL = "sc_test_3.jsonl"
OUTPUT_PDF = "sc_test_2.pdf"

MAX_PAGES = 5000
MAX_DEPTH = 20
REQUEST_DELAY = 1.0
PAGE_LOAD_TIMEOUT = 30
MAX_TEXT_CHARS_PER_PAGE = 50000
MAX_PDF_CHARS = 30000   # truncate extracted PDF text to avoid huge JSON


UNICODE_TTF_PATH = "DejaVuSans.ttf"  # must exist next to this script
HEADER_TITLE = "FSU Department of Scientific Computing — Site Archive"

# Crawl only links found inside the body container to avoid nav recursion
CRAWL_LINKS_FROM_BODY_ONLY = True


###############################################################################
# BODY EXTRACTION HEURISTICS
###############################################################################
MAIN_CANDIDATE_SELECTORS = [
    "main",
    "[role=main]",
    "#content", "#main", "#primary", "#g-main", "#contentarea", "#component", "#page-content",
    ".profile-content",".content", ".main", ".main-content", ".page-content", ".article", ".article-content",
    ".entry-content", ".item-page", ".component", ".region-content"
]

NOISE_ID_CLASS_RE = re.compile(
    r"(nav|menu|breadcrumb|breadcrumbs|footer|header|sidebar|side-bar|"
    r"topbar|toolbar|sidenav|offcanvas|pager|pagination|tabs|share|social|"
    r"skip|cookie|consent|advert|ad-)",
    re.I,
)
ROLE_NOISE = {"navigation","banner", "contentinfo", "complementary", "search"}


###############################################################################
# LOGIN HELPERS
###############################################################################

def load_cookies(driver, path="cookies.pkl"):
    cookies = pickle.load(open(path, "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)


def is_logged_in_heuristic(driver):
    """Heuristic: check for 'logout' text or absence of 'login' markers on START_URL."""
    try:
        html = driver.page_source.lower()
        if "logout" in html or "sign out" in html or "log out" in html:
            return True
        # If page obviously still a login page
        if "login" in html and ("password" in html or 'type="password"' in html):
            return False
    except Exception:
        pass
    # Fallback: if we're not stuck on a login URL and we can access START_URL.
    return "login" not in driver.current_url.lower()


def perform_login(driver, login_url):
    """Attempt automated login, then verify; supports manual flow if configured."""
    # Go to login page
    driver.get(login_url)
    wait_for_ready(driver)

        # Manual login flow
    print(f"[Login] Please log in manually (DUO/SSO). "
          f"Waiting up to {MANUAL_LOGIN_MAX_SECONDS}s...")

    end = time.time() + MANUAL_LOGIN_MAX_SECONDS
    while time.time() < end:
        if is_logged_in_heuristic(driver):
            print("[Login] Login successful.")
            break
        time.sleep(2)
    else:
        raise RuntimeError("Manual login timed out. Increase MANUAL_LOGIN_MAX_SECONDS if DUO takes longer.")


    # wait_for_ready(driver)
    # if not is_logged_in_heuristic(driver):
    #     raise RuntimeError("Post-login access to START_URL still looks unauthenticated.")
    

###############################################################################
# UTILITIES
###############################################################################
def normalize_url(base, link):
    if not link:
        return None
    abs_url = urljoin(base, link)
    normalized = urlparse(urlparse(abs_url)._replace(fragment="").geturl())
    return normalized.geturl()


def is_internal(url):
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc in ALLOWED_NETLOCS
    except Exception:
        return False


def clean_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def decode_cloudflare_email(cf_str):
    if not cf_str:
        return None
    try:
        r = bytes.fromhex(cf_str)
        key = r[0]
        decoded = bytes([b ^ key for b in r[1:]]).decode("utf-8")
        return decoded
    except Exception:
        return None

def looks_like_error_page(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in ERROR_PATTERNS)

def remove_nodes_by_selectors(root: Tag, selectors: list[str]):
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
    if not isinstance(root, Tag):
        return
    to_remove = []
    for t in list(root.find_all(True)):
        if not isinstance(t, Tag):
            continue
        try:
            role_attr = t.attrs.get("role")
        except Exception:
            role_attr = None
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

def should_skip_url(url: str) -> bool:
    """Return True if the URL should be skipped."""
    skip_prefixes = [
        "sc.fsu.edu/news-and-events/photo-gallery/",
        "sc.fsu.edu/~dduke/",
        "sc.fsu.edu/component/weblinks/",
        # "sc.fsu.edu/links/"
    ]
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()

    # Only skip if it's under sc.fsu.edu
    if "sc.fsu.edu" in netloc:
        for prefix in skip_prefixes:
            if path.startswith("/" + prefix.split("/", 1)[1]):  # match path after domain
                return True
    return False


###############################################################################
# Strip_noise or Clean_main
###############################################################################
def clean_main_container(main: Tag) -> Tag:
    if not main:
        return main
    # remove scripts/styles/templates/iframes inside main
    remove_nodes_by_selectors(main, ["script", "style", "noscript", "template", "iframe"])
    # remove small navigation elements inside main
    for nav in main.select("nav, .breadcrumb, .breadcrumbs, .sidebar, .top, .prefooter"):
        nav.decompose()
    # remove hidden elements
    for el in main.select('[style*="display:none"], [aria-hidden="true"]'):
        el.decompose()
    return main

def remove_html_comments(soup: BeautifulSoup):
    """Remove all HTML comments from the soup."""
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

def remove_navigation_blocks(root: Tag):
    if not isinstance(root, Tag):
        return
    
    # Remove sections based on module comments
    for comment_text in ["module breadcrumbs", "module sidebar", "module top", "module prefooter"]:
        for comment in root.find_all(string=lambda text: isinstance(text, type(root.string)) and comment_text in text):
            parent = comment.find_parent()
            if parent:
                parent.decompose()
    
    # Remove by common IDs/classes if needed
    for sel in ["#content .s3-w", ".s3-nv", ".mod-custom"]:
        for el in root.select(sel):
            el.decompose()
    
    return root

def strip_noise(root: Tag) -> Tag:
    """Remove non-body elements from a Tag; fully defensive."""
    if not isinstance(root, Tag):
        return root

    # universal removals
    remove_nodes_by_selectors(root, ["script", "style", "noscript",  "template", "svg", "form", "iframe"])
    # structural noise
    remove_nodes_by_selectors(root, ["header", "footer", "aside"])
    # role-based noise
    remove_nodes_by_role(root, ROLE_NOISE)
    # id/class pattern noise
    remove_nodes_by_id_class_pattern(root, NOISE_ID_CLASS_RE)

    return root


def pick_main_container(soup: BeautifulSoup) -> Tag:
    """Choose the most likely 'body' container; always return a Tag."""

    # 1) explicit selectors
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

    # 2) largest content block among article/section/div
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

    # 3) fallback: prefer <body>, else first tag, else empty div
    if isinstance(soup.body, Tag):
        return soup.body
    first_tag = soup.find(True)
    if isinstance(first_tag, Tag):
        return first_tag
    return soup.new_tag("div")


def extract_body_text(node: Tag) -> str:
    """Extract visible text from a page container, keeping paragraphs, headings, and nested divs."""
    if not isinstance(node, Tag):
        return ""

    parts = []
    if not isinstance(node, Tag):
        return ""
    parts = []
    for sel in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tbody", "thead", "td", "th", "pre", "code"]:
        try:
            iter_els = node.select(sel)
        except Exception:
            iter_els = []
        for el in iter_els:
            if not isinstance(el, Tag):
                continue
            t = clean_text(el.get_text(separator=" ", strip=True))
            if t:
                parts.append(t)
    return "\n".join(parts).strip()

# def extract_body_text_with_links(node: Tag) -> str:
#     """
#     Extracts visible text from a page container,
#     but replaces <a> tags pointing to external profiles with their URLs.
#     """
#     if not isinstance(node, Tag):
#         return ""

#     parts = []
#     seen = set()  # track unique lines

#     for sel in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "code"]:
#         for el in node.select(sel):
#             if not isinstance(el, Tag):
#                 continue

#             # Replace <a> text with URL if it's a full link
#             for a in el.find_all("a", href=True):
#                 href = a["href"].strip()
#                 if href.lower().startswith("http"):
#                     a.string = href

#             # Combine stripped strings in this element
#             t = " ".join(el.stripped_strings)
#             t = re.sub(r"\s+", " ", t)

#             # Skip if already seen
#             if t and t not in seen:
#                 seen.add(t)
#                 parts.append(t)

#     return "\n".join(parts).strip()


# unique_links = set()  # global or outer-scope set to track links

# def extract_body_text_recursive(node: Tag) -> str:

#     """Recursively extract all visible text from a node, ignoring script/style."""
#     if not isinstance(node, Tag):
#         return ""
    
#     if node.name in ["script", "style", "noscript", "template", "svg" "iframe", "form", "img"]:
#         return ""
    

#     texts = []
#     for child in node.children:
#         if isinstance(child, Tag):
#             # Recursively extract from child
#             texts.append(extract_body_text_recursive(child))
            
#             # Replace <a> text with URL if it's a full link
#             for a in child.find_all("a", href=True):
#                 href = a["href"].strip()
#                 # Skip unwanted LinkedIn edu link
#                 if href.lower().startswith("http"):
#                     if href not in unique_links:
#                         unique_links.add(href)
#                         texts.append(href)   
        
#         elif hasattr(child, "string") and child.string:
#             texts.append(clean_text(str(child)))

#     return "\n".join([t for t in texts if t]).strip()


unique_links = set()  # global or outer scope

def is_valid_link(href: str) -> bool:
    """
    Filter unwanted links:
    - Must start with http/https
    - Exclude internal/ugly query-heavy links (ldap, php sessions, cmd, etc.)
    """
    href = href.strip().lower()
    if not href.startswith("http"):
        return False

    # blocklist patterns
    blocked_patterns = [
        "ldap", "login", "logout", "cmd=", "session", "token", "phpldapadmin"
    ]
    return not any(b in href for b in blocked_patterns)


def extract_text_with_links(node: Tag) -> str:
    """
    Extract visible text + unique valid links from a BeautifulSoup Tag.
    - Captures block-level text once (avoids duplicates).
    - Replaces <a> text with its href if valid external link.
    - Collects valid unique links into the text column.
    """
    if not isinstance(node, Tag):
        return ""

    # Tags to ignore completely
    if node.name in ["script", "style", "noscript", "template", "svg", "iframe", "form", "img"]:
        return ""

    parts = []
    seen_lines = set()

    # Process only block-level tags (to avoid duplication)
    if node.name in ["h1","h2","h3","h4","h5","h6","p","li","td","th","pre","code"]:
        # Replace <a> text with href if valid
        for a in node.find_all("a", href=True):
            href = a["href"].strip()
            if is_valid_link(href):
                a.string = href

        text = " ".join(node.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen_lines:
            seen_lines.add(text)
            parts.append(text)

    # Recurse into children (but don't double count block-level text)
    for child in node.children:
        if isinstance(child, Tag):
            parts.append(extract_text_with_links(child))
            for a in child.find_all("a", href=True):
                href = a["href"].strip()
                if is_valid_link(href) and href not in unique_links:
                    unique_links.add(href)
                    parts.append(href)  # store link in text column
        elif hasattr(child, "string") and child.string:
            clean = clean_text(str(child))
            if clean:
                parts.append(clean)

    return "\n".join([p for p in parts if p]).strip()


###############################################################################
# JOOMLA-SPECIFIC EMAIL DECODING
###############################################################################
def decode_joomla_email(tag: Tag):
    """Decode emails hidden in Joomla <joomla-hidden-mail> tags."""
    if not tag:
        return None

    # Case 1: base64 first + last
    first = tag.get("d2VibWFzdGVy")
    last = tag.get("c2MuZnN1LmVkdQ==")
    if first and last:
        try:
            local = base64.b64decode(first).decode("utf-8")
            domain = base64.b64decode(last).decode("utf-8")
            return f"{local}@{domain}"
        except Exception:
            pass

    # Case 2: text attribute (hex or encoded string)
    text_val = tag.get("d2VibWFzdGVyQHNjLmZzdS5lZHU=") or tag.string
    if text_val:
        try:
            # if hex-encoded string
            b = bytes.fromhex(text_val)
            key = b[0]
            decoded = bytes([x ^ key for x in b[1:]]).decode("utf-8")
            return decoded
        except Exception:
            try:
                # fallback: maybe base64 in text
                decoded = base64.b64decode(text_val).decode("utf-8")
                return decoded
            except Exception:
                return None

    return None

###############################################################################
# CONTENT EXTRACTION
###############################################################################

def extract_content(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    remove_html_comments(soup)

        # title
    title = clean_text(soup.title.get_text()) if soup.title else ""


    # emails (global)
    cf_emails = []
    for tag in soup.select("[data-cfemail]"):
        try:
            email = decode_cloudflare_email(tag.get("data-cfemail"))
        except Exception:
            email = None
        if email:
            cf_emails.append(email)
    
    # mailto links
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
    
    # Joomla hidden emails
    joomla_emails = []
    for tag in soup.find_all("joomla-hidden-mail"):
        j_mail = decode_joomla_email(tag)
        if j_mail:
            joomla_emails.append(j_mail)

    emails = sorted(set(cf_emails + mailto_emails + joomla_emails))

    # main (guaranteed Tag), cleaned
    main = pick_main_container(soup)
    main = clean_main_container(main)
    main = remove_navigation_blocks(main)

    # TEXT: body only
    page_text = extract_text_with_links(main)

    # anchor texts: body only
    anchor_texts = []
    for a in main.find_all("a"):
        if not isinstance(a, Tag):
            continue
        t = clean_text(a.get_text(" "))
        href = a.get("href", "")
        if t and href and not NOISE_ID_CLASS_RE.search(" ".join(a.get("class", []))):
            anchor_texts.append(t)
    anchor_texts = list(dict.fromkeys(anchor_texts))  # remove duplicates


    # links for BFS
    link_scope = main if CRAWL_LINKS_FROM_BODY_ONLY else soup
    internal_links = set()
    for a in link_scope.find_all("a", href=True):
        try:
            href = normalize_url(base_url, a["href"])
        except Exception:
            href = None
        if href and is_internal(href):
            internal_links.add(href)

    # external profiles (whole doc)
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
            # 🔹 Extract text directly from PDF and collect it
            pdf_text = extract_pdf_text(abs_url)
            if pdf_text:
                pdf_texts.append(pdf_text)

        netloc = urlparse(abs_url).netloc.lower()
        if "scholar.google.com" in netloc or "linkedin.com" in netloc:
            external_profile_links.append(abs_url)
        

        # 🔹 Merge HTML text + all extracted PDF text
    combined_text = " ".join([page_text] + pdf_texts).strip()

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
    WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(0.5)


def allowed_by_robots(url, rp: robotparser.RobotFileParser):
    try:
        return rp.can_fetch("*", url)
    except Exception:
        return True
    
def save_cookies(driver, path="cookies.pkl"):
    with open(path, "wb") as f:
        pickle.dump(driver.get_cookies(), f)

def load_cookies(driver, path="cookies.pkl"):
    import os
    if os.path.exists(path):
        with open(path, "rb") as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            # Selenium needs domain match
            driver.add_cookie(cookie)
        driver.refresh()
    
###############################################################################
# PDF HELPERS
###############################################################################

def extract_pdf_text(url):
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
    
###############################################################################
# PDF (DejaVu only)
###############################################################################
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("DejaVu", size=9)
        self.set_y(8)
        self.cell(0, 6, HEADER_TITLE, ln=1, align="C")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.set_font("DejaVu", size=9)
        self.set_text_color(100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def add_toc_entry(self, title, level=0):
        try:
            self.bookmark((title or "Untitled")[:120], level=level)
        except Exception:
            pass


def ensure_font(pdf: PDF):
    if not os.path.exists(UNICODE_TTF_PATH):
        raise FileNotFoundError(
            f"Unicode font not found at '{UNICODE_TTF_PATH}'. "
            "Place a valid TTF (e.g., DejaVuSans.ttf) next to this script."
        )
    pdf.add_font("DejaVu", "", UNICODE_TTF_PATH, uni=True)
    pdf.add_font("DejaVu", "B", UNICODE_TTF_PATH, uni=True)
    pdf.add_font("DejaVu", "I", UNICODE_TTF_PATH, uni=True)


def pdf_add_kv(pdf: PDF, key: str, value: str, link: str | None = None):
    pdf.set_font("DejaVu", "B", 11)
    pdf.cell(0, 6, key, ln=1)
    pdf.set_font("DejaVu", "", 11)
    if link:
        pdf.set_text_color(30, 30, 180)
        pdf.multi_cell(0, 6, value, link=link)
        pdf.set_text_color(0)
    else:
        pdf.multi_cell(0, 6, value)
    pdf.ln(1)


def pdf_add_section_title(pdf: PDF, title: str):
    pdf.set_font("DejaVu", "B", 14)
    pdf.multi_cell(0, 8, title if title else "Untitled")
    pdf.ln(1)


def pdf_add_body_text(pdf: PDF, text: str):
    text = (text or "").strip()
    if len(text) > MAX_TEXT_CHARS_PER_PAGE:
        text = text[:MAX_TEXT_CHARS_PER_PAGE] + "\n\n[...truncated for brevity in PDF reference...]"
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 6, text)
    pdf.ln(2)


def build_pdf(jsonl_path: str, pdf_path: str):
    pdf = PDF(format="Letter", orientation="P", unit="mm")
    ensure_font(pdf)

    pdf.add_page()
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 12, "FSU Scientific Computing — Website Text Archive", ln=1, align="C")
    pdf.ln(2)

    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 6, "Plain-text reference compiled from sc.fsu.edu (body content only).")
    pdf.ln(2)

    pdf.set_font("DejaVu", "I", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(2)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            url = rec.get("url", "")
            title = rec.get("title") or ""
            emails = rec.get("emails", [])
            body = (rec.get("text", "") or "").strip()
            profile_links = rec.get("external_profile_links", [])

            pdf.add_page()
            pdf.add_toc_entry(title or url, level=0)

            pdf_add_section_title(pdf, title if title else url)
            if url:
                pdf_add_kv(pdf, "Source URL:", url, link=url)
            if emails:
                pdf_add_kv(pdf, "Emails found:", ", ".join(emails))
            if profile_links:
                pdf_add_kv(pdf, "Profile links (Scholar/LinkedIn):", "\n".join(profile_links))

            pdf_add_kv(pdf, "Text:", "")
            pdf_add_body_text(pdf, body if body else "[No extracted text]")

    pdf.output(pdf_path)
    print(f"Saved PDF: {pdf_path}")


###############################################################################
# MAIN CRAWLER
###############################################################################
def crawl(driver, start_url):
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
            if url in seen:
                continue
            seen.add(url)

            if depth > MAX_DEPTH:
                continue
            if not is_internal(url):
                continue

            if not allowed_by_robots(url, rp):
                print(f"[robots.txt] Skipping: {url}")
                continue

            try:
                time.sleep(REQUEST_DELAY)
                driver.get(url)
                load_cookies(driver)
                wait_for_ready(driver)
                html = driver.page_source

                data = extract_content(html, url)
                # Extract PDF contents
                pdf_text = ""
                if url.lower().endswith(".pdf"):
                    pdf_text = extract_pdf_text(url)
                    # Override the page text with PDF text
                    data["text"] = pdf_text  

                record = {
                    "url": url,
                    "depth": depth,
                    "title": data["title"],
                    "text": data["text"],                    # body-only
                    "emails": data["emails"],
                    "anchor_texts": data["anchor_texts"],    # from body
                    "out_links": data["links"],              # from body or full page based on toggle
                    "external_profile_links": data["external_profile_links"],
                    #"pdf_data": pdf_text                    # <--- NEW field with PDFs
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pages_crawled += 1
                print(f"[{pages_crawled}] {url} (links: {len(data['links'])}, emails: {len(data['emails'])})")

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
            input("After you have logged in successfully, press Enter to continue...")
            print("[Login] CAS login successful ✅")

        perform_login(driver, LOGIN_URL2)
        if is_logged_in_heuristic(driver):
            input("After you have logged in successfully, press Enter to continue...")
            print("[Login] sc.fsu.edu login successful ✅")

            
        # Crawl using logged-in session
        crawl(driver, START_URL)

    finally:
        driver.quit()    
        
    build_pdf(OUTPUT_PDF)
