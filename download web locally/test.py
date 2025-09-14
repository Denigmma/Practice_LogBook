#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from urllib.parse import urlparse, urljoin, unquote

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE_URL = "https://education.yandex.ru/handbook/ml/"
COOKIES_PATH = "YandexhHandbookML.json"
DELAY = 3

BASE_PATH = urlparse(BASE_URL).path
OUTDIR = os.path.join("results", BASE_PATH.strip("/").replace("/", "_"))


chrome_opts = Options()
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--no-sandbox")
# Уберите комментарий ниже, если хотите видеть окно браузера и вручную решать капчу
# chrome_opts.headless = False
chrome_opts.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(options=chrome_opts)


sess = requests.Session()
sess.headers.update({
    "User-Agent": chrome_opts.arguments[-1]
})


cookie_file = COOKIES_PATH if os.path.exists(COOKIES_PATH) else os.path.join("cookies", COOKIES_PATH)
if os.path.exists(cookie_file):
    with open(cookie_file, encoding="utf-8") as f:
        cookies = json.load(f)
    for c in cookies:
        sess.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
        try:
            driver.add_cookie({
                "name": c["name"], "value": c["value"],
                "domain": c.get("domain"), "path": c.get("path", "/")
            })
        except Exception:
            pass

visited_pages = set()
downloaded_assets = set()

def get_soup_with_selenium(url: str) -> BeautifulSoup:
    driver.get(url)
    time.sleep(DELAY)
    # если упёрлись в капчу — ждём ручного решения
    if "captcha" in driver.page_source.lower():
        print(f"Обнаружена капча на {url} — решите её в окне браузера, нажмите Enter после решения")
        input()
        time.sleep(DELAY)
    return BeautifulSoup(driver.page_source, "html.parser")

def get_rel_path(url: str) -> str:
    path = unquote(urlparse(url).path)
    if path.startswith(BASE_PATH):
        rel = path[len(BASE_PATH):]
    else:
        rel = path.lstrip("/")
    # делаем чтобы всегда был каталог
    if not rel.endswith("/"):
        rel += "/"
    return rel

def save_html(rel_path: str, soup: BeautifulSoup) -> str:
    full_dir = os.path.normpath(os.path.join(OUTDIR, *rel_path.strip("/").split("/")))
    os.makedirs(full_dir, exist_ok=True)
    file_path = os.path.join(full_dir, "index.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return file_path

def download_asset(abs_url: str) -> str | None:
    parsed = urlparse(abs_url)
    ext = os.path.splitext(parsed.path)[1].lower()
    ALLOWED = {".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ttf", ".eot"}
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
    # правим пути у CSS
    for link in soup.find_all("link", href=True):
        href = link["href"]
        if os.path.splitext(urlparse(href).path)[1].lower() == ".css":
            new = download_asset(urljoin(page_url, href))
            if new:
                link["href"] = new

    # правим пути у JS
    for script in soup.find_all("script", src=True):
        src = script["src"]
        if os.path.splitext(urlparse(src).path)[1].lower() == ".js":
            new = download_asset(urljoin(page_url, src))
            if new:
                script["src"] = new

    # правим пути у картинок
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

    soup = get_soup_with_selenium(url)

    # вставляем <base href="/">, чтобы все относительные пути шли от корня
    if soup.head:
        base_tag = soup.new_tag("base", href="/")
        soup.head.insert(0, base_tag)

    next_urls: set[str] = set()

    # собираем все теги <a>
    for a in soup.find_all("a", href=True):
        h = a["href"].strip()
        if h.startswith(("#", "mailto:", "tel:")) or h.lower().startswith("javascript:"):
            continue
        abs_href = urljoin(url, h)
        if not abs_href.startswith(BASE_URL):
            continue
        # нормализуем слеш в конце
        if not abs_href.endswith("/"):
            abs_href += "/"
        next_urls.add(abs_href)
        # переписываем на относительный корень
        rel = abs_href[len(BASE_URL):]
        a["href"] = "/" + rel

    # сохраняем исходный HTML
    rel_path = get_rel_path(url)
    page_file = save_html(rel_path, soup)

    # скачиваем ассеты и правим пути
    process_assets(soup, url)

    # перезаписываем финальный HTML
    with open(page_file, "w", encoding="utf-8") as f:
        f.write(str(soup))

    # рекурсивно обходим следующие страницы
    for u in next_urls:
        crawl(u)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    crawl(BASE_URL)
    driver.quit()
    print(f"\n Офлайн-версия готова в папке: {OUTDIR}")
