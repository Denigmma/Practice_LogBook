import os
from flask import Flask, render_template_string, send_from_directory, abort

app = Flask(__name__)

# Корневая папка с офлайн-ресурсами
BASE_DIR = os.path.join(os.path.dirname(__file__), "results")

# HTML-шаблон хаба
INDEX_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Offline Hub</title>
  <style>
    body { font-family: sans-serif; background: #f5f5f5; margin:0; padding:2rem; }
    h1 { text-align: center; color: #333; }
    .grid { display: flex; flex-wrap: wrap; gap:1rem; justify-content: center; }
    .card {
      background: white;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 1rem 1.5rem;
      flex: 0 0 160px;
      text-align: center;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      transition: box-shadow .2s;
    }
    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .card a { text-decoration: none; color: #333; font-weight: bold; display: block; }
    .card a:hover { color: #007bff; }
  </style>
</head>
<body>
  <h1>Offline Hub</h1>
  <div class="grid">
    {% for item in items %}
      <div class="card">
        <a href="/{{ item }}/">{{ item }}</a>
      </div>
    {% else %}
      <p>Папок не найдено в results/aids_study_notes/</p>
    {% endfor %}
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    try:
        items = sorted(
            name for name in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, name))
        )
    except FileNotFoundError:
        items = []
    return render_template_string(INDEX_HTML, items=items)

@app.route("/<folder>/", defaults={"req_path": ""})
@app.route("/<folder>/<path:req_path>")
def serve_folder(folder, req_path):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        abort(404)

    if req_path == "" or req_path.endswith("/"):
        req_path = os.path.join(req_path, "index.html")

    full_path = os.path.normpath(os.path.join(folder_path, req_path))
    if not full_path.startswith(os.path.abspath(folder_path)):
        abort(403)

    rel_path = os.path.relpath(full_path, folder_path)
    return send_from_directory(folder_path, rel_path)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
    print("Сервер запущен по адресу: http://127.0.0.1:5000/")
