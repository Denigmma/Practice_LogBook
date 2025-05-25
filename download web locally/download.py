import json
import os
from urllib.parse import urlparse, unquote, urljoin

import requests
from bs4 import BeautifulSoup

# 1) dir

BASE_URL = "https://saharasurfer.gitlab.io/aids_study_notes/"
cookies_path = "aids_study_notes.json"

# BASE_URL = "https://education.yandex.ru/handbook/ml"
# cookies_path = "yandex_hendbook_ml.json"

BASE_PATH = urlparse(BASE_URL).path
OUTDIR = f"results/{BASE_PATH.replace('/', '_')}"


# 2) настраиваем requests.Session() с куками из JSON
sess = requests.Session()
with open("cookies/" + cookies_path, encoding="utf-8") as f:
    cookies = json.load(f)
for c in cookies:
    sess.cookies.set(
        name=c["name"],
        value=c["value"],
        domain=c["domain"],
        path=c["path"]
    )

visited_pages = set()
downloaded_assets = set()


def get_rel_path(url: str) -> str:
    """
    Из полного URL возвращает относительный путь после BASE_PATH,
    гарантированно с '/' на конце.
    """
    path = unquote(urlparse(url).path)
    if path.startswith(BASE_PATH):
        rel = path[len(BASE_PATH):]
    else:
        rel = path.lstrip("/")
    if not rel.endswith("/"):
        rel += "/"
    return rel


def save_html(rel_path: str, soup: BeautifulSoup) -> str:
    """
    Сохраняет HTML в results/aids_study_notes/<rel_path>/index.html
    и возвращает полный путь к файлу.
    """
    parts = rel_path.strip("/").split("/")  # e.g. ["semester-2","topology"]
    full_dir = os.path.normpath(os.path.join(OUTDIR, *parts))
    os.makedirs(full_dir, exist_ok=True)
    file_path = os.path.join(full_dir, "index.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return file_path


def download_asset(abs_url: str) -> str | None:
    """
    Скачивает файлы с расширениями из ALLOWED,
    сохраняет в results/aids_study_notes/, возвращает
    абсолютный путь вида "/css/…", "/img/…".
    """
    parsed = urlparse(abs_url)
    ext = os.path.splitext(parsed.path)[1].lower()
    ALLOWED = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
               ".woff", ".woff2", ".ttf", ".eot"}
    if ext not in ALLOWED:
        return None

    decoded = unquote(parsed.path)
    # убираем префикс BASE_PATH
    if decoded.startswith(BASE_PATH):
        rel = decoded[len(BASE_PATH):].lstrip("/")
    else:
        rel = decoded.lstrip("/")

    local_path = os.path.normpath(os.path.join(OUTDIR, rel))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if local_path not in downloaded_assets:
        r = sess.get(abs_url)
        if r.status_code == 200:
            with open(local_path, "wb") as rf:
                rf.write(r.content)
        downloaded_assets.add(local_path)

    # абсолютный путь от корня веб-сервера
    return "/" + rel.replace(os.path.sep, "/")


def process_assets(soup: BeautifulSoup, page_url: str):
    """
    Переписывает в soup все CSS/JS/IMG на локальные
    абсолютные пути, скачивая их через download_asset.
    """
    # CSS
    for link in soup.find_all("link", href=True):
        href = link["href"]
        if href.lower().endswith(".css"):
            abs_url = urljoin(page_url, href)
            new_href = download_asset(abs_url)
            if new_href:
                link["href"] = new_href

    # JS
    for script in soup.find_all("script", src=True):
        src = script["src"]
        if src.lower().endswith(".js"):
            abs_url = urljoin(page_url, src)
            new_src = download_asset(abs_url)
            if new_src:
                script["src"] = new_src

    # IMAGES
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if src.startswith("data:"):
            continue
        abs_url = urljoin(page_url, src)
        new_src = download_asset(abs_url)
        if new_src:
            img["src"] = new_src


def crawl(url: str):
    """
    Рекурсивно обходит все страницы:
    - по <a href>
    - по <link rel="next"> и <link rel="prev">
    Сохраняет HTML + ассеты.
    """
    if url in visited_pages or not url.startswith(BASE_URL):
        return
    visited_pages.add(url)
    print("Fetching:", url)

    r = sess.get(url)
    if r.status_code != 200:
        print("  !!! Ошибка", r.status_code, "при", url)
        return

    soup = BeautifulSoup(r.text, "html.parser")

    # 1) переписываем все внутренние <a href>
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith(BASE_URL):
            rel = h[len(BASE_URL):]
            if not rel.endswith("/"):
                rel += "/"
            a["href"] = rel

    # 2) обрабатываем <link rel="next"> и <link rel="prev">
    page_links = []
    for link in soup.find_all("link", href=True):
        rels = [r.lower() for r in link.get("rel", [])]
        if "next" in rels or "prev" in rels:
            href = link["href"]
            # абсолютный URL
            abs_href = href if href.startswith("http") else urljoin(BASE_URL, href)
            # запомним для краулинга
            page_links.append(abs_href)
            # перепишем href на относительный
            rel = get_rel_path(abs_href)
            link["href"] = rel

    # 3) сохраняем чистый HTML (еще без ассетов)
    rel_path = get_rel_path(url)
    page_file = save_html(rel_path, soup)

    # 4) качаем CSS/JS/img и переписываем ссылки
    process_assets(soup, url)

    # 5) перезаписываем HTML уже с готовыми ассет-ссылками
    with open(page_file, "w", encoding="utf-8") as f:
        f.write(str(soup))

    # 6) рекурсивно идем по всем найденным ссылкам
    #    а) из <a>
    for a in soup.find_all("a", href=True):
        next_url = urljoin(BASE_URL, a["href"])
        crawl(next_url)
    #    б) из <link rel="next|prev">
    for pl in page_links:
        crawl(pl)


if __name__ == "__main__":
    crawl(BASE_URL)
    print("\n✅ Офлайн-версия готова в папке:", OUTDIR)
