#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from urllib.parse import urlparse, urljoin, unquote
import requests
from bs4 import BeautifulSoup

# — Начальные настройки —
BASE_URL = "https://education.yandex.ru/handbook/ml"
COOKIES_PATH = "YandexhHandbookML.json"

BASE_PATH = urlparse(BASE_URL).path
OUTDIR = f"results/{BASE_PATH.replace('/', '_')}"

# — Создаём сессию и настраиваем заголовки, куки —
sess = requests.Session()
sess.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;"
              "q=0.9,*/*;q=0.8"
})

# пытаемся загрузить куки из текущей папки или из ./cookies/
cookie_file = COOKIES_PATH
if not os.path.exists(cookie_file):
    cookie_file = os.path.join("cookies", COOKIES_PATH)
if os.path.exists(cookie_file):
    with open(cookie_file, encoding="utf-8") as f:
        for c in json.load(f):
            sess.cookies.set(
                name=c["name"],
                value=c["value"],
                domain=c.get("domain"),
                path=c.get("path", "/")
            )

visited_pages = set()
downloaded_assets = set()


def get_rel_path(url: str) -> str:
    path = unquote(urlparse(url).path)
    if path.startswith(BASE_PATH):
        rel = path[len(BASE_PATH):]
    else:
        rel = path.lstrip("/")
    if not rel.endswith("/"):
        rel += "/"
    return rel


def save_html(rel_path: str, soup: BeautifulSoup) -> str:
    full_dir = os.path.normpath(
        os.path.join(OUTDIR, *rel_path.strip("/").split("/"))
    )
    os.makedirs(full_dir, exist_ok=True)
    file_path = os.path.join(full_dir, "index.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return file_path


def download_asset(abs_url: str) -> str | None:
    parsed = urlparse(abs_url)
    ext = os.path.splitext(parsed.path)[1].lower()
    ALLOWED = {
        ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".woff", ".woff2", ".ttf", ".eot"
    }
    if ext not in ALLOWED:
        return None

    decoded = unquote(parsed.path)
    if decoded.startswith(BASE_PATH):
        rel_on_disk = decoded[len(BASE_PATH):].lstrip("/")
    else:
        rel_on_disk = decoded.lstrip("/")

    local_path = os.path.normpath(os.path.join(OUTDIR, rel_on_disk))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if local_path not in downloaded_assets:
        r = sess.get(abs_url)
        if r.status_code == 200:
            with open(local_path, "wb") as rf:
                rf.write(r.content)
            downloaded_assets.add(local_path)
    return "/" + rel_on_disk.replace(os.path.sep, "/")


def process_assets(soup: BeautifulSoup, page_url: str):
    for link in soup.find_all("link", href=True):
        href = link["href"]
        path = urlparse(href).path
        if os.path.splitext(path)[1].lower() == ".css":
            new = download_asset(urljoin(page_url, href))
            if new:
                link["href"] = new

    for script in soup.find_all("script", src=True):
        src = script["src"]
        path = urlparse(src).path
        if os.path.splitext(path)[1].lower() == ".js":
            new = download_asset(urljoin(page_url, src))
            if new:
                script["src"] = new

    for img in soup.find_all("img", src=True):
        src = img["src"]
        if not src.startswith("data:"):
            new = download_asset(urljoin(page_url, src))
            if new:
                img["src"] = new


def crawl(url: str):
    if url in visited_pages:
        return
    visited_pages.add(url)
    print("Fetching:", url)

    resp = sess.get(url)
    if resp.status_code != 200:
        print("  !!! Ошибка", resp.status_code, "при", url)
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # вставляем <base href="/">, чтобы все относительные шли от корня
    if soup.head:
        base_tag = soup.new_tag("base", href="/")
        soup.head.insert(0, base_tag)

    next_urls: set[str] = set()

    # <a href>
    for a in soup.find_all("a", href=True):
        h = a["href"].strip()
        if h.startswith(("#", "mailto:", "tel:")) or h.lower().startswith("javascript:"):
            continue
        abs_href = urljoin(url, h)
        if not abs_href.startswith(BASE_URL):
            continue

        p = unquote(urlparse(abs_href).path)
        if not p.startswith(BASE_PATH):
            continue

        # пропускаем «page/N» обёртки
        parts = p[len(BASE_PATH):].strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "lectures" and parts[1] == "page":
            continue

        # добавляем слеш, если директория
        name, ext = os.path.splitext(p)
        if ext == "" and not p.endswith("/"):
            p += "/"

        rel = p[len(BASE_PATH):].lstrip("/")
        a["href"] = "/" + rel
        next_urls.add(BASE_URL + rel)

    # <link rel="next|prev">
    for link in soup.find_all("link", href=True):
        rels = [r.lower() for r in link.get("rel", [])]
        if not any(r in ("next", "prev") for r in rels):
            continue
        h = link["href"].strip()
        abs_href = urljoin(url, h)
        if not abs_href.startswith(BASE_URL):
            continue
        p = unquote(urlparse(abs_href).path)
        if not p.endswith("/"):
            p += "/"
        rel = p[len(BASE_PATH):].lstrip("/")
        link["href"] = "/" + rel
        next_urls.add(BASE_URL + rel)

    # сохраняем HTML до замены ассетов
    rel_path = get_rel_path(url)
    page_file = save_html(rel_path, soup)

    # скачиваем ассеты и правим пути
    process_assets(soup, url)

    # перезаписываем финальный HTML
    with open(page_file, "w", encoding="utf-8") as f:
        f.write(str(soup))

    for u in next_urls:
        crawl(u)


if __name__ == "__main__":
    crawl(BASE_URL)
    print(f"\n Офлайн-версия готова в папке: {OUTDIR}")
