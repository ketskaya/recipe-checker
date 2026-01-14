import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from fuzzywuzzy import fuzz
from datetime import datetime
from tensorflow.keras.models import load_model
import random


# ======================
# Загрузка модели и скейлера
# ======================
try:
    model = load_model("models/recipe_duplicate_mlp.h5")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Ошибка загрузки модели или скейлера: {e}")
    st.stop()

# ======================
# Функции из Colab
# ======================

def normalize_dob(dob_str):
    if pd.isna(dob_str) or dob_str == "":
        return ""
    s = str(dob_str).lower()
    s = re.sub(r'[гг\.]', '', s)
    s = re.sub(r'[^0-9a-zа-яё\s\-\/\.\']', '', s)
    month_map = {
        'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04', 'май': '05', 'июн': '06',
        'июл': '07', 'авг': '08', 'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
    }
    for word, num in month_map.items():
        if word in s:
            s = re.sub(word, num, s)
    formats = ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %m %Y", "%d.%m'%y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except:
            continue
    return ""

def normalize_mnn(mnn):
    if pd.isna(mnn):
        return ""
    s = str(mnn).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    typo_fix = {"парацитамол": "парацетамол", "ибупрафен": "ибупрофен"}
    for w, r in typo_fix.items():
        if w in s:
            s = s.replace(w, r)
    return s

def normalize_snils(snils_str):
    if not isinstance(snils_str, str):
        snils_str = str(snils_str)
    digits = re.sub(r'\D', '', snils_str)
    return digits if len(digits) == 11 else ""

def extract_features(row_a, row_b):
    mnn_a = normalize_mnn(row_a['МНН'])
    mnn_b = normalize_mnn(row_b['МНН'])
    mnn_ratio = fuzz.ratio(mnn_a, mnn_b) / 100.0
    mnn_partial = fuzz.partial_ratio(mnn_a, mnn_b) / 100.0

    issued_a = str(row_a['Выписано ЛС'])
    issued_b = str(row_b['Выписано ЛС'])
    issued_ratio = fuzz.ratio(issued_a, issued_b) / 100.0
    issued_token = fuzz.token_sort_ratio(issued_a, issued_b) / 100.0

    disp_a = str(row_a['ЛС (отпущенное / зарезервированное)'])
    disp_b = str(row_b['ЛС (отпущенное / зарезервированное)'])
    disp_ratio = fuzz.ratio(disp_a, disp_b) / 100.0

    snils_a = normalize_snils(row_a['СНИЛС'])
    snils_b = normalize_snils(row_b['СНИЛС'])
    snils_match = 1.0 if snils_a == snils_b and len(snils_a) == 11 else 0.0

    dob_a = normalize_dob(row_a['Дата рождения пациента'])
    dob_b = normalize_dob(row_b['Дата рождения пациента'])
    dob_match = 1.0 if dob_a == dob_b and dob_a != "" else 0.0

    qty_issued_diff = abs(row_a['Кол-во выписано'] - row_b['Кол-во выписано'])
    qty_disp_diff = abs(row_a['Кол-во отпущенного ЛС'] - row_b['Кол-во отпущенного ЛС'])

    return np.array([
        mnn_ratio,
        mnn_partial,
        issued_ratio,
        issued_token,
        disp_ratio,
        snils_match,
        dob_match,
        qty_issued_diff,
        qty_disp_diff
    ])

# ======================
# База примеров
# ======================
EXAMPLE_RECORDS = [
    {
        "СНИЛС": "12345678900",
        "Дата рождения пациента": "1990-01-01",
        "МНН": "Ибупрофен",
        "Выписано ЛС": "Ибупрофен таблетки 200мг №30",
        "ЛС (отпущенное / зарезервированное)": "Ибупрофен таб. 200мг",
        "Кол-во выписано": 2,
        "Кол-во отпущенного ЛС": 2
    },
    {
        "СНИЛС": "123-456-789 00",
        "Дата рождения пациента": "01.01.1990",
        "МНН": "Парацетамол",
        "Выписано ЛС": "Парацетамол таб. 500мг – №20",
        "ЛС (отпущенное / зарезервированное)": "Парацетамол таб. 500мг",
        "Кол-во выписано": 3,
        "Кол-во отпущенного ЛС": 2
    },
    {
        "СНИЛС": "123 456 789 00",
        "Дата рождения пациента": "1 янв 1990 г.",
        "МНН": "парацетамол",
        "Выписано ЛС": "ПАРАЦЕТАМОЛ ТАБ 500МГ N20",
        "ЛС (отпущенное / зарезервированное)": "Парацетамол таб 500мг",
        "Кол-во выписано": 3,
        "Кол-во отпущенного ЛС": 1
    }
]

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Проверка рецептов", layout="centered")
st.title("💊 Проверка: относятся ли записи к одному рецепту?")

st.markdown("""
Введите данные двух записей из отчёта — даже с опечатками, разным регистром или форматом.
Модель оценит, насколько вероятно, что они описывают **один и тот же рецепт**.
""")

# === Кнопка "Заполнить из базы" ===
if st.button("🎲 Заполнить из базы"):
    # Выбираем две разные записи из EXAMPLE_RECORDS
    if len(EXAMPLE_RECORDS) >= 2:
        rec1, rec2 = random.sample(EXAMPLE_RECORDS, 2)
        st.session_state.update({
            "snils1": rec1["СНИЛС"],
            "dob1": rec1["Дата рождения пациента"],
            "mnn1": rec1["МНН"],
            "issued1": rec1["Выписано ЛС"],
            "disp1": rec1["ЛС (отпущенное / зарезервированное)"],
            "qty1": rec1["Кол-во выписано"],
            "qty_disp1": rec1["Кол-во отпущенного ЛС"],

            "snils2": rec2["СНИЛС"],
            "dob2": rec2["Дата рождения пациента"],
            "mnn2": rec2["МНН"],
            "issued2": rec2["Выписано ЛС"],
            "disp2": rec2["ЛС (отпущенное / зарезервированное)"],
            "qty2": rec2["Кол-во выписано"],
            "qty_disp2": rec2["Кол-во отпущенного ЛС"],
        })
        st.rerun()

# === Запись 1 ===
st.subheader("Запись 1")
col1, col2 = st.columns(2)

with col1:
    snils1 = st.text_input("СНИЛС*", key="snils1", placeholder="123 456 789 00")
    dob1 = st.text_input("Дата рождения*", key="dob1", placeholder="01.01.1990")
    mnn1 = st.text_input("МНН препарата*", key="mnn1", placeholder="Парацетамол")

with col2:
    issued1 = st.text_input("Выписано ЛС*", key="issued1", placeholder="Парацетамол таб. 500мг №20")
    disp1 = st.text_input("Отпущено ЛС*", key="disp1", placeholder="Парацетамол таб. 500мг")
    qty_issued1 = st.number_input("Кол-во выписано*", min_value=1, value=3, key="qty1")
    qty_disp1 = st.number_input("Кол-во отпущенного*", min_value=1, value=1, key="qty_disp1")

# === Запись 2 ===
st.subheader("Запись 2")
col3, col4 = st.columns(2)

with col3:
    snils2 = st.text_input("СНИЛС*", key="snils2", placeholder="123 456 789 00")
    dob2 = st.text_input("Дата рождения*", key="dob2", placeholder="01.01.1990")
    mnn2 = st.text_input("МНН препарата*", key="mnn2", placeholder="Парацетамол")

with col4:
    issued2 = st.text_input("Выписано ЛС*", key="issued2", placeholder="Парацетамол таб. 500мг №20")
    disp2 = st.text_input("Отпущено ЛС*", key="disp2", placeholder="Парацетамол таб. 500мг")
    qty_issued2 = st.number_input("Кол-во выписано*", min_value=1, value=3, key="qty2")
    qty_disp2 = st.number_input("Кол-во отпущенного*", min_value=1, value=1, key="qty_disp2")

# === Кнопка анализа ===
if st.button("🔍 Проверить"):
    # Проверка обязательных полей
    required_fields = [
        ("СНИЛС 1", snils1.strip()),
        ("Дата рождения 1", dob1.strip()),
        ("МНН 1", mnn1.strip()),
        ("Выписано ЛС 1", issued1.strip()),
        ("Отпущено ЛС 1", disp1.strip()),
        ("СНИЛС 2", snils2.strip()),
        ("Дата рождения 2", dob2.strip()),
        ("МНН 2", mnn2.strip()),
        ("Выписано ЛС 2", issued2.strip()),
        ("Отпущено ЛС 2", disp2.strip()),
    ]

    missing = [name for name, val in required_fields if not val]
    if missing:
        st.warning(f"⚠️ Пожалуйста, заполните все обязательные поля: {', '.join(missing)}")
    else:
        row1 = {
            "СНИЛС": snils1,
            "Дата рождения пациента": dob1,
            "МНН": mnn1,
            "Выписано ЛС": issued1,
            "ЛС (отпущенное / зарезервированное)": disp1,
            "Кол-во выписано": qty_issued1,
            "Кол-во отпущенного ЛС": qty_disp1
        }
        row2 = {
            "СНИЛС": snils2,
            "Дата рождения пациента": dob2,
            "МНН": mnn2,
            "Выписано ЛС": issued2,
            "ЛС (отпущенное / зарезервированное)": disp2,
            "Кол-во выписано": qty_issued2,
            "Кол-во отпущенного ЛС": qty_disp2
        }

        features = extract_features(row1, row2)
        features_scaled = scaler.transform([features])
        prob = model.predict(features_scaled)[0][0]

        if prob > 0.5:
            st.success(f"✅ С вероятностью **{prob:.1%}** эти записи относятся к **одному рецепту**.")
        else:
            st.error(f"❌ С вероятностью **{1 - prob:.1%}** это **разные рецепты**.")
            
# streamlit run app.py