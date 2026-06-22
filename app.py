import os, re, sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="HC Analytics Chatbot", page_icon="📊")

SCHEMA_STR = """employees(nip, nama, divisi, jabatan, join_date)
trainings(training_id, nama_diklat, tanggal)
enrollments(nip, training_id, status, nilai)

Relasi:
- enrollments.nip          -> employees.nip
- enrollments.training_id  -> trainings.training_id
Catatan: enrollments.nilai bisa kosong (NULL) jika status = 'berjalan'."""


# ---------- Secrets: st.secrets > environment. JANGAN hardcode key. ----------
def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


LLM_PROVIDER = (get_secret("LLM_PROVIDER", "groq") or "groq").lower()


@st.cache_resource
def get_db():
    con = sqlite3.connect(":memory:", check_same_thread=False)
    for fn in ["employees", "trainings", "enrollments"]:
        pd.read_csv(f"data/{fn}.csv").to_sql(fn, con, index=False, if_exists="replace")
    return con


@st.cache_resource
def get_llm():
    if LLM_PROVIDER == "groq":
        from groq import Groq
        return ("groq", Groq(api_key=get_secret("GROQ_API_KEY")),
                get_secret("GROQ_MODEL", "llama-3.3-70b-versatile"))
    import google.generativeai as genai
    genai.configure(api_key=get_secret("GEMINI_API_KEY"))
    model = get_secret("GEMINI_MODEL", "gemini-1.5-flash")
    return ("gemini", genai.GenerativeModel(model), model)


def build_prompt(question, dialect="sqlite"):
    return f"""Anda adalah generator SQL untuk database {dialect}.
Gunakan HANYA tabel dan kolom pada skema berikut:

{SCHEMA_STR}

ATURAN KETAT:
- Balas dengan TEPAT SATU pernyataan SQL SELECT. Tanpa penjelasan/markdown.
- Hanya SELECT (dilarang INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT).
- Pencarian teks: LOWER(kolom) LIKE LOWER('%kata%') agar tidak case-sensitive.
- Rata-rata nilai: abaikan NULL (WHERE nilai IS NOT NULL).
- "belum mengikuti diklat X" = nip TIDAK ADA di enrollments untuk diklat X.

Contoh:
Pertanyaan: Berapa jumlah pegawai per divisi?
SQL: SELECT divisi, COUNT(*) AS jumlah_pegawai FROM employees GROUP BY divisi ORDER BY jumlah_pegawai DESC;

Pertanyaan: Siapa yang belum mengikuti diklat Leadership?
SQL: SELECT nama, divisi FROM employees WHERE nip NOT IN (SELECT e.nip FROM enrollments e JOIN trainings t ON e.training_id=t.training_id WHERE LOWER(t.nama_diklat) LIKE LOWER('%Leadership%'));

Pertanyaan: {question}
SQL:"""


def clean_sql(raw):
    s = re.sub(r"^```(?:sql)?", "", raw.strip(), flags=re.I).strip()
    s = re.sub(r"```$", "", s).strip()
    m = re.search(r"(?is)\bselect\b.*", s)
    if m:
        s = m.group(0)
    return s.strip().rstrip(";").strip()


def generate_sql(question):
    kind, client, model = get_llm()
    prompt = build_prompt(question)
    if kind == "groq":
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        return clean_sql(r.choices[0].message.content)
    r = client.generate_content(prompt)
    return clean_sql(r.text)


FORBIDDEN = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "grant"]


def validate_sql(sql):
    if not sql or not sql.strip():
        return False
    s = sql.strip().rstrip(";").strip()
    if not s.lower().startswith("select"):
        return False
    if ";" in s:
        return False
    low = s.lower()
    return not any(re.search(r"\b" + w + r"\b", low) for w in FORBIDDEN)


def run_sql(sql):
    return pd.read_sql_query(sql, get_db())


def make_chart(df):
    num = df.select_dtypes(include="number").columns.tolist()
    cat = [c for c in df.columns if c not in num]
    if not num or not cat or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    s = df.set_index(cat[0])[num[0]]
    is_time = bool(re.search(r"(tanggal|date|bulan|tahun|periode)", cat[0], re.I))
    (s.plot.line(ax=ax, marker="o") if is_time else s.plot.bar(ax=ax))
    ax.set_title(f"{num[0]} per {cat[0]}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


st.title("📊 HC Analytics — Conversational Chatbot")
st.caption("Tanya bahasa natural → SQL → database → jawaban + grafik")
with st.expander("Skema database"):
    st.code(SCHEMA_STR)
st.markdown(
    "**Contoh:** jumlah pegawai per divisi · siapa belum ikut diklat Data Engineering · "
    "rata-rata nilai diklat per divisi"
)

if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Tulis pertanyaan Anda..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            sql = generate_sql(q)
            if not validate_sql(sql):
                sql = generate_sql(q)  # retry 1x
            if not validate_sql(sql):
                st.error("Belum bisa membuat query yang aman untuk pertanyaan itu.")
            else:
                st.code(sql, language="sql")
                df = run_sql(sql)
                if df.empty:
                    st.info("Tidak ada data yang cocok.")
                else:
                    st.dataframe(df, use_container_width=True)
                    fig = make_chart(df)
                    if fig:
                        st.pyplot(fig)
                st.session_state.msgs.append(
                    {"role": "assistant", "content": f"```sql\n{sql}\n```"}
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
