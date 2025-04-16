from flask import Flask, request, render_template, jsonify, url_for, Response
import numpy as np
import cv2
import os
import uuid
import json
import io, csv
from urllib.parse import quote
from datetime import datetime
from flask import session, redirect
from flask import Flask
from config import Config
from models import Result, db
from telethon.sync import TelegramClient
from telethon.errors import PeerIdInvalidError, UserPrivacyRestrictedError
import subprocess
import sys

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

MASK_PATH = "static/mask.png"
FILL_PATH = "static/fill.png"
BORDER_PATH = "static/border.png"
FRAGMENTS_DIR = 'static/dataset'
SMALL_FRAGMENT_IDS = [1, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 21, 22, 23]
UPLOAD_DIR = "static/uploads"
USER_RESULTS_DIR = "static/user_results"
LINK_LOGS_DIR = "logs/links"
TEMPLATE_PATH = "static/template.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(USER_RESULTS_DIR, exist_ok=True)
os.makedirs(LINK_LOGS_DIR, exist_ok=True)

def load_fragment_masks():
    masks = {}
    for fname in os.listdir(FRAGMENTS_DIR):
        if fname.endswith('.png') and fname[:-4].isdigit():
            idx = int(os.path.splitext(fname)[0])
            path = os.path.join(FRAGMENTS_DIR, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                bin_mask = (img < 200).astype(np.uint8)
                masks[idx] = bin_mask
    return masks

fragment_masks = load_fragment_masks()

def count_hit_fragments(user_bin, fragment_masks, threshold=0.4, small_ids=None):
    if small_ids is None:
        small_ids = []

    hit_count = 0
    hit_ids = []
    small_hit_count = 0

    for frag_id, frag_mask in fragment_masks.items():
        frag_area = np.sum(frag_mask)
        intersection = np.sum((frag_mask == 1) & (user_bin == 1))
        if frag_area > 0 and (intersection / frag_area) >= threshold:
            hit_count += 1
            hit_ids.append(frag_id)
            if frag_id in small_ids:
                small_hit_count += 1

    return hit_count, hit_ids, small_hit_count

def extract_features(filepath):
    user = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    fill = cv2.imread(FILL_PATH, cv2.IMREAD_GRAYSCALE)
    border = cv2.imread(BORDER_PATH, cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in [user, mask, fill, border]):
        raise Exception("Не удалось загрузить одну из масок")
    user_bin = (user < 200).astype(np.uint8)
    mask_bin = (mask < 200).astype(np.uint8)
    border_bin = (border < 200).astype(np.uint8)
    h, w = user.shape
    mid = w // 2
    total_fig_area = np.sum(mask_bin == 1)
    fill_inside = np.sum((user_bin == 1) & (mask_bin == 1))
    fill_ratio = fill_inside / total_fig_area if total_fig_area > 0 else 0
    left_mask = np.zeros_like(mask_bin)
    left_mask[:, :mid] = 1
    right_mask = np.zeros_like(mask_bin)
    right_mask[:, mid:] = 1
    left_area = np.sum(mask_bin & left_mask)
    right_area = np.sum(mask_bin & right_mask)
    left_fill = np.sum((user_bin == 1) & (mask_bin == 1) & (left_mask == 1))
    right_fill = np.sum((user_bin == 1) & (mask_bin == 1) & (right_mask == 1))
    left_ratio = left_fill / left_area if left_area > 0 else 0
    right_ratio = right_fill / right_area if right_area > 0 else 0
    center_mask = np.zeros_like(mask_bin)
    center_mask[:, mid - 50: mid + 50] = 1
    center_fill = np.sum((user_bin == 1) & (mask_bin == 1) & (center_mask == 1))
    center_ratio = center_fill / total_fig_area if total_fig_area > 0 else 0
    border_total = np.sum(border_bin == 1)
    outline_overlap = np.sum((user_bin == 1) & (border_bin == 1))
    outline_ratio = outline_overlap / border_total if border_total > 0 else 0
    outline_detected = outline_ratio >= 0.4
    frag_count, frag_ids, small_fragments = count_hit_fragments(user_bin, fragment_masks, threshold=0.4, small_ids=SMALL_FRAGMENT_IDS)
    outside_fill = np.sum((user_bin == 1) & (mask_bin == 0))
    outside_ratio = outside_fill / (h * w)
    return {
        'total_fill_ratio': fill_ratio,
        'left_ratio': left_ratio,
        'right_ratio': right_ratio,
        'fill_center': center_ratio,
        'outline_detected': bool(outline_detected),
        'outline_ratio': round(outline_ratio * 100, 2),
        'num_fragments': frag_count,
        'matched_fragment_ids': frag_ids,
        'small_fragments': small_fragments,
        'outside_ratio': outside_ratio
    }

def interpret(f):
    keys = []
    levels = {"suicidal": 0.0, "anxiety": 0.0, "depression": 0.0}

    # 🎯 Суицидальность
    levels["suicidal"] += f["left_ratio"] * 100  # Чем больше закрашена левая часть — тем выше риск
    levels["suicidal"] += f["outside_ratio"] * 60  # Рисование вне границ может говорить о нестабильности

    if 0 < f["small_fragments"] <= 3 and f["num_fragments"] == f["small_fragments"]:
        keys.append("🔴 Если закрашено не больше трёх маленьких частей — возможны суицидальные мысли в сложных обстоятельствах.")
        levels["suicidal"] += 30 + (3 - f["small_fragments"]) * 5  # Чем меньше фрагментов, тем тревожнее

    if f["num_fragments"] >= 1:
        keys.append("🔴 Продолженное закрашивание говорит о возможных мыслях о смерти.")
        levels["suicidal"] += min(40, f["num_fragments"] * 5)  # за каждый фрагмент по 5 баллов

    if f["right_ratio"] >= 0.3:
        keys.append("🟠 Закрашивание правой стороны — возможно, использование суицидальности как способа влияния.")
        levels["suicidal"] += (f["right_ratio"] - 0.3) * 100

    # 🎯 Депрессия
    levels["depression"] += f["total_fill_ratio"] * 80  # плотное закрашивание
    levels["depression"] += f["fill_center"] * 40       # фокус на себе

    if f["num_fragments"] >= 7:
        keys.append("🔴 Если на вашем рисунке оказалось больше закрашенных, чем пустых мест — это говорит о мрачном настроении.")
        levels["depression"] += 50

    if f["left_ratio"] >= 0.3:
        keys.append("🔴 Закрашивание левой части говорит о душевной ранимости.")
        levels["depression"] += f["left_ratio"] * 100 * 0.6  # умеренный вклад

    # 🎯 Тревожность
    levels["anxiety"] += f["right_ratio"] * 90
    levels["anxiety"] += f["outside_ratio"] * 50

    if not f["outline_detected"]:
        levels["anxiety"] += 20  # отсутствие обводки

    if f["small_fragments"] == 0 and f["num_fragments"] >= 1:
        keys.append("🟠 Вы выбираете сильные акценты — возможно, у вас стремление к контролю или к подавлению деталей.")
        levels["anxiety"] += 20

    if f["fill_center"] > 0.3:
        keys.append("🔵 Закрашивание центра может говорить о самоанализе и сосредоточенности на себе.")

    if f["outline_detected"] and f["num_fragments"] == 0 and f["small_fragments"] == 0:
        keys.append("🟢 Если вы только обвели фигуру — это говорит о вашей железной воле и самоконтроле.")
        levels["anxiety"] += 5

    if abs(f["left_ratio"] - f["right_ratio"]) < 0.05 and f["left_ratio"] > 0.2:
        keys.append("🟢 Вы стремитесь к внутреннему равновесию. Ваш рисунок сбалансирован.")
        levels["anxiety"] -= 10

    if f["outside_ratio"] > 0.1:
        keys.append("🟠 Закрашена часть вне фигуры — возможно, вы пытаетесь выйти за рамки или теряетесь в себе.")

    if not keys:
        keys.append("🟠 Интерпретация неоднозначна. Возможно, вы закрашивали случайно.")
        levels["anxiety"] += 15

    # 🔐 Ограничения и округление
    for k in levels:
        levels[k] = min(100.0, max(0.0, round(levels[k], 1)))

    # 🧾 Общий вывод
    if levels["suicidal"] >= 70:
        overall = "🔴 Высокий риск. Обратитесь за поддержкой."
    elif levels["depression"] >= 60:
        overall = "🔵 Признаки депрессии. Нужно восстановление."
    elif levels["anxiety"] >= 60:
        overall = "🟠 Повышенная тревожность. Вы напряжены."
    elif levels["suicidal"] < 20 and levels["depression"] < 20:
        overall = "🟢 Эмоциональный баланс в норме."
    else:
        overall = "🟠 Лёгкая тревожность и напряжение."

    return {
        "keys": keys,
        "summary": "\\n".join(keys),
        "levels": levels,
        "overall": overall
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    nickname = request.form.get("user") or f"anon_{uuid.uuid4().hex[:6]}"

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # ✅ Создание изображения с наложением шаблона
    composed_path = os.path.join(USER_RESULTS_DIR, f"{nickname}_composed.png")
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    user_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if user_img.shape != template.shape:
        user_img = cv2.resize(user_img, (template.shape[1], template.shape[0]))

    colored = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    user_mask = (user_img < 200)
    colored[user_mask] = (0, 0, 0)

    cv2.imwrite(composed_path, colored)

    # 👉 Анализ изображения
    features = extract_features(filepath)
    result = interpret(features)

    # ✅ Сохранение результата
    user_data = {
        "nickname": nickname,
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "user_image": f"{nickname}_composed.png",  # важно для отображения
        "features": features,
        "levels": result["levels"],
        "overall": result["overall"],
        "keys": result["keys"]
    }

    with open(os.path.join(USER_RESULTS_DIR, f"{nickname}.json"), "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

    entry = Result(
        nickname=nickname,
        timestamp=user_data["timestamp"],
        image=user_data["user_image"],
        suicidal=user_data["levels"]["suicidal"],
        anxiety=user_data["levels"]["anxiety"],
        depression=user_data["levels"]["depression"],
        keys="\n".join(user_data["keys"])
    )
    
    try:
        db.session.add(entry)
        db.session.commit()
        print("✅ Сохранено в PostgreSQL:", entry.nickname)
    except Exception as e:
        print("❌ Ошибка при сохранении:", e)


    return jsonify({
        "result": result["summary"],
        "levels": result["levels"],
        "overall": result["overall"],
        "keys": result["keys"],
        "features": features,
        "filename": filename,
        "user_image": f"{nickname}_composed.png"
    })


@app.route("/result/<nickname>")
def result_page(nickname):
    path = os.path.join(USER_RESULTS_DIR, f"{nickname}.json")
    if not os.path.exists(path):
        return f"Результат для пользователя {nickname} не найден."

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_url = url_for("static", filename=f"uploads/{data['filename']}")
    return render_template("result.html", image_url=image_url, **data)

# 🔒 Админ логин
from flask import session, redirect
app.secret_key = 'your_secret_key_here'

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == "rusik2002":
            session["admin"] = True
            return redirect("/admin")
        else:
            return "Неверный пароль"
    return '''<form method="post"><input type="password" name="password"><button type="submit">Войти</button></form>'''

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect("/login")

from flask import session, redirect

@app.route("/admin")
def admin_panel():
    if not session.get("admin"):
        return redirect("/login")

    min_suicidal = request.args.get("suicidal", type=int)
    min_anxiety = request.args.get("anxiety", type=int)
    min_depression = request.args.get("depression", type=int)
    user_filter = request.args.get("user", type=str)
    key_filter = request.args.get("key", type=str)
    date_from = request.args.get("date_from", type=str)
    date_to = request.args.get("date_to", type=str)
    sort_by = request.args.get("sort", default="timestamp")
    order = request.args.get("order", default="desc")
    export = request.args.get("export") == "1"

    records = []
    for fname in os.listdir(USER_RESULTS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(USER_RESULTS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                levels = data.get("levels", {})
                nickname = data.get("nickname", "N/A")
                ts = data.get("timestamp", "—")
                keys = data.get("keys", [])
                image = data.get("user_image", "")
                suicidal = levels.get("suicidal", 0)
                anxiety = levels.get("anxiety", 0)
                depression = levels.get("depression", 0)

                if min_suicidal is not None and suicidal < min_suicidal:
                    continue
                if min_anxiety is not None and anxiety < min_anxiety:
                    continue
                if min_depression is not None and depression < min_depression:
                    continue
                if user_filter and user_filter.lower() not in nickname.lower():
                    continue
                if key_filter and not any(key_filter.lower() in k.lower() for k in keys):
                    continue
                if date_from and ts < date_from:
                    continue
                if date_to and ts > date_to:
                    continue

                records.append({
                    "nickname": nickname,
                    "timestamp": ts,
                    "suicidal": suicidal,
                    "anxiety": anxiety,
                    "depression": depression,
                    "image": image
                })

    reverse = (order == "desc")
    if sort_by in ["nickname", "timestamp", "suicidal", "anxiety", "depression"]:
        records.sort(key=lambda x: x[sort_by], reverse=reverse)

    if export:
        output = io.StringIO()
        writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Никнейм", "Дата", "Суицидальность", "Тревожность", "Депрессия"])
        for r in records:
            date_only = r["timestamp"].split("T")[0]
            writer.writerow([r['nickname'], date_only, f"{r['suicidal']}%", f"{r['anxiety']}%", f"{r['depression']}%"])
        csv_content = output.getvalue().encode("utf-8-sig")
        filename = "результаты_анализа.csv"
        safe_filename = quote(filename)
        return Response(csv_content, content_type="text/csv", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"})

    def sort_link(field, title):
        new_order = "asc" if sort_by != field or order == "desc" else "desc"
        base = f"?sort={field}&order={new_order}"
        for p in ["suicidal", "anxiety", "depression", "user", "key", "date_from", "date_to"]:
            val = request.args.get(p)
            if val:
                base += f"&{p}={val}"
        arrow = " 🔽" if sort_by == field and order == "desc" else " 🔼" if sort_by == field else ""
        return f'<a href="{base}">{title}{arrow}</a>'

    rows = ""
    for r in records:
        bg = ""
        if r["suicidal"] >= 70 or r["depression"] >= 70:
            bg = 'bg-red-100'
        elif r["suicidal"] <= 10 and r["anxiety"] <= 10 and r["depression"] <= 10:
            bg = 'bg-green-100'
        date_only = r["timestamp"].split("T")[0]
        image_tag = f'<img src="/static/user_results/{r["image"]}" class="w-12 h-auto inline">' if r["image"] else ""
        rows += f"""
            <tr class="{bg}">
                <td class='p-2'>{r["nickname"]}</td>
                <td class='p-2'>{date_only}</td>
                <td class='p-2'>{r["suicidal"]}%</td>
                <td class='p-2'>{r["anxiety"]}%</td>
                <td class='p-2'>{r["depression"]}%</td>
                <td class='p-2'>{image_tag}</td>
                <td class='p-2'><a href="/result/{r['nickname']}" target="_blank" class="text-blue-600 hover:underline">Открыть</a></td>
            </tr>
        """

    return f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Админ-панель</title>
        <link href='https://cdn.jsdelivr.net/npm/@unocss/reset/tailwind.min.css' rel='stylesheet'>
        <script src='https://cdn.tailwindcss.com'></script>
    </head>

    <body class='bg-gray-100 text-gray-800'>
    <div class='max-w-7xl mx-auto py-10 px-4'>
        <h1 class='text-3xl font-bold mb-6'>📊 Админ-панель: Результаты анализа</h1>

        <!-- 🎯 Загрузка и рассылка ников -->
        <div class="bg-white p-4 rounded shadow mb-6">
            <h2 class="text-lg font-semibold mb-2">📥 Загрузка и рассылка списка ников</h2>
            <form id="uploadForm" enctype="multipart/form-data" class="flex flex-col md:flex-row md:items-center gap-4">
                <input type="file" name="nickfile" required class="border p-2 rounded w-full max-w-xs">
                <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded">📤 Загрузить ники</button>
                <a href="/admin/send_links_from_file" class="bg-purple-600 text-white px-4 py-2 rounded text-center">📨 Разослать</a>
            </form>
            <p id="statusMessage" class="text-green-600 mt-2"></p>
        </div>

        <!-- 🧠 Фильтры -->
        <form method='get' class='grid grid-cols-2 md:grid-cols-4 gap-4 bg-white p-4 rounded shadow mb-6'>
            <input type='number' name='suicidal' placeholder='Суицид ≥' value='{min_suicidal or ''}' class='border p-2 rounded'>
            <input type='number' name='anxiety' placeholder='Тревожн ≥' value='{min_anxiety or ''}' class='border p-2 rounded'>
            <input type='number' name='depression' placeholder='Депресс ≥' value='{min_depression or ''}' class='border p-2 rounded'>
            <input type='text' name='user' placeholder='Никнейм' value='{user_filter or ''}' class='border p-2 rounded'>
            <input type='text' name='key' placeholder='Ключ' value='{key_filter or ''}' class='border p-2 rounded'>
            <input type='date' name='date_from' value='{date_from or ''}' class='border p-2 rounded'>
            <input type='date' name='date_to' value='{date_to or ''}' class='border p-2 rounded'>
            <button class='bg-blue-600 text-white px-4 py-2 rounded'>Фильтровать</button>
        </form>

        <!-- 🔗 Кнопки -->
        <div class='flex gap-4 mb-6'>
            <a href='/admin' class='text-sm text-blue-600 hover:underline'>⛔ Сбросить</a>
            <a href='{request.full_path}&export=1' class='text-sm text-green-600 hover:underline'>⬇️ Экспорт CSV</a>
        </div>

        <!-- 📄 Таблица -->
        <div class='overflow-auto bg-white rounded shadow'>
            <table class='table-auto w-full text-sm'>
                <thead class='bg-gray-200'>
                    <tr>
                        <th class='p-2'>{sort_link("nickname", "Ник")}</th>
                        <th class='p-2'>{sort_link("timestamp", "Дата")}</th>
                        <th class='p-2'>{sort_link("suicidal", "Суицид")}</th>
                        <th class='p-2'>{sort_link("anxiety", "Тревожн")}</th>
                        <th class='p-2'>{sort_link("depression", "Депресс")}</th>
                        <th class='p-2'>Изобр.</th>
                        <th class='p-2'>Ссылка</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>

    <!-- 🧩 Скрипт загрузки ников -->
    <script>
    document.addEventListener("DOMContentLoaded", function () {{
        const form = document.getElementById("uploadForm");
        form.addEventListener("submit", async function (e) {{
            e.preventDefault();
            const formData = new FormData(form);
            const res = await fetch("/admin/upload_nicknames", {{
                method: "POST",
                body: formData,
            }});
            const text = await res.text();
            document.getElementById("statusMessage").textContent = text;
        }});
    }});
    </script>
    </body>
    </html>
    """


@app.route("/admin/send_links")
def send_links():
    uploaded_nicks_path = os.path.join(LINK_LOGS_DIR, "uploaded_nicks.txt")
    use_uploaded = os.path.exists(uploaded_nicks_path)

    # Фильтры
    min_suicidal = request.args.get("suicidal", type=int)
    min_anxiety = request.args.get("anxiety", type=int)
    min_depression = request.args.get("depression", type=int)

    filtered_nicknames = []
    uploaded_nicks = set()

    if use_uploaded:
        with open(uploaded_nicks_path, "r", encoding="utf-8") as f:
            uploaded_nicks = set(line.strip() for line in f)

    for fname in os.listdir(USER_RESULTS_DIR):
        if fname.endswith(".json"):
            path = os.path.join(USER_RESULTS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                levels = data.get("levels", {})
                nickname = data.get("nickname")
                if not nickname:
                    continue

                if use_uploaded and nickname not in uploaded_nicks:
                    continue

                if not use_uploaded:
                    if min_suicidal and levels.get("suicidal", 0) < min_suicidal:
                        continue
                    if min_anxiety and levels.get("anxiety", 0) < min_anxiety:
                        continue
                    if min_depression and levels.get("depression", 0) < min_depression:
                        continue

                filtered_nicknames.append(nickname)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"filtered_links_{now}.txt"
    path = os.path.join(LINK_LOGS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        for nick in filtered_nicknames:
            link = f"https://example.com/?user={nick}"
            f.write(link + "\n")

    return Response(
        open(path, "rb").read(),
        content_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/admin/upload_nicknames", methods=["POST"])
def upload_nicknames():
    file = request.files.get("nickfile")
    if not file:
        return "Файл не выбран", 400

    content = file.read().decode("utf-8")
    nicknames = [line.strip() for line in content.splitlines() if line.strip()]

    # Сохраняем в лог
    os.makedirs(LINK_LOGS_DIR, exist_ok=True)
    path = os.path.join(LINK_LOGS_DIR, "target_nicknames.txt")
    with open(path, "w", encoding="utf-8") as f:
        for nick in nicknames:
            f.write(nick + "\n")

    return "Ники успешно загружены!"

@app.route("/admin/send_links_from_file")
def send_links_from_file():
    path = os.path.join(LINK_LOGS_DIR, "target_nicknames.txt")
    if not os.path.exists(path):
        return "Сначала загрузите файл с никами.", 400

    # ✅ Запуск асинхронного скрипта
    subprocess.Popen([sys.executable, "send_test.py"])

    return "⏳ Рассылка запущена! Проверьте консоль."

@app.route("/admin/test_db")
def test_db():
    results = Result.query.all()
    return f"В базе сейчас {len(results)} записей"

@app.route("/init_db")
def init_db():
    with app.app_context():
        db.create_all()
    return "✅ База данных и таблицы успешно созданы!"

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
