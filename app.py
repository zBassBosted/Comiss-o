import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Simulador de Comissão", layout="wide")

# ---------------- Helpers ----------------
def moeda_br(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "R$ 0,00"
    try:
        return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return f"R$ {valor}"

def find_cell(df, text):
    text = text.lower()
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if isinstance(v, str) and text in v.lower():
                return (r, c)
    return None

def find_header_row_with(df, *tokens):
    tokens = [t.lower() for t in tokens]
    for r in range(df.shape[0]):
        row_vals = [(str(x).strip() if x is not None else '') for x in list(df.loc[r, :])]
        lower = [x.lower() for x in row_vals]
        if all(t in lower for t in tokens):
            return r
    return None

def parse_faixas(raw):
    r = find_header_row_with(raw, "de", "até")
    if r is None:
        return None
    block = raw.loc[r:r+100, 0:15].copy()
    headers = [x if (isinstance(x, str) and x.strip() != "") else f"col{j}" for j, x in enumerate(block.iloc[0, :])]
    block.columns = headers
    block = block[1:]

    # cortar em primeira linha totalmente vazia
    def is_empty(sr):
        return all(pd.isna(v) or (isinstance(v, str) and v.strip() == "") for v in sr.values)
    end_idx = None
    for idx, sr in block.iterrows():
        if is_empty(sr):
            end_idx = idx
            break
    if end_idx is not None:
        block = block.loc[:end_idx-1]

    # identificar colunas DE/ATÉ
    de_col = next((h for h in headers if isinstance(h, str) and h.strip().upper() == "DE"), None)
    ate_col = next((h for h in headers if isinstance(h, str) and h.strip().upper() == "ATÉ"), None)
    if not de_col or not ate_col:
        return None

    # escolher próximas 3 colunas mais numéricas como fatores (BL, TKM, Churn — nessa ordem)
    factor_cols = []
    for h in headers:
        if h not in [de_col, ate_col]:
            col = block[h]
            frac_num = col.apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).mean()
            if frac_num > 0.5:
                factor_cols.append(h)
    factor_cols = factor_cols[:3]
    if len(factor_cols) < 3:
        # fallback: tentar colunas pelos rótulos comuns
        candidates = []
        for cand in ["Banda Larga", "BL", "TKM", "Ticket", "Churn"]:
            for h in headers:
                if isinstance(h, str) and cand.lower() in h.lower():
                    candidates.append(h)
        for h in candidates:
            if h in block.columns and h not in factor_cols:
                factor_cols.append(h)
        factor_cols = factor_cols[:3]

    faixas = block[[de_col, ate_col] + factor_cols].copy()
    faixas.columns = ["DE", "ATE", "F_BL", "F_TKM", "F_CHURN"]
    return faixas

def lookup_factor(val, faixas, fator_col):
    if faixas is None or val is None:
        return None
    for _, row in faixas.iterrows():
        try:
            de = float(row["DE"])
        except:
            de = None
        try:
            atef = float(row["ATE"])
        except:
            # pode haver texto como "Acima" ou NaN; trata abaixo
            atef = None

        if de is not None and atef is not None and de <= val <= atef:
            v = row[fator_col]
            return float(v) if isinstance(v, (int, float)) else None

    # se valor acima do último intervalo definido, usar última linha como "acima"
    try:
        last = faixas.iloc[-1]
        thr = float(last["DE"]) if not pd.isna(last["DE"]) else None
        if thr is not None and val > thr:
            v = last[fator_col]
            return float(v) if isinstance(v, (int, float)) else None
    except:
        pass
    return None

def to_csv_bytes(df):
    bio = BytesIO()
    df.to_csv(bio, index=False, sep=";", encoding="utf-8-sig")
    bio.seek(0)
    return bio

# ---------------- Fonte de dados ----------------
st.title("📈 Simulador de Comissão")

# ---------------- Ler faixas ----------------
faixas = parse_faixas(raw)
if faixas is None:
    st.error("Não consegui identificar a tabela de 'Faixas de Atingimento' (DE/ATÉ + 3 colunas de fatores). Verifique a planilha.")
    st.stop()

with st.expander("Ver tabela de Faixas de Atingimento"):
    st.dataframe(faixas, use_container_width=True)

# ---------------- Ler bloco SIMULADOR (valores iniciais) ----------------
# Procurar valores padrão dentro do grid para popular os inputs
def read_after(raw, label):
    # tenta pegar primeiro número à direita ou na linha abaixo
    pos = find_cell(raw, label)
    if not pos:
        return None
    r, c = pos
    candidates = []
    if c + 1 < raw.shape[1]:
        candidates.append((r, c + 1))
    if r + 1 < raw.shape[0]:
        candidates.append((r + 1, c))
    for rr, cc in candidates:
        v = raw.iat[rr, cc]
        if isinstance(v, (int, float)) and not pd.isna(v):
            return float(v)
    # fallback: procurar primeiro número naquela linha
    row_vals = list(raw.loc[r, :])
    for v in row_vals:
        if isinstance(v, (int, float)) and not pd.isna(v):
            return float(v)
    return None

meta_bl_default = read_after(raw, "META:")
real_bl_default = None
meta_tkm_default = None
real_tkm_default = None
meta_churn_default = None
real_churn_default = None
salario_default = read_after(raw, "SALÁRIO:")
fator_final_default = read_after(raw, "FATOR FINAL:")
comissao_default = read_after(raw, "COMISSÃO:")

# Os valores da sua planilha (conhecidos do exemplo)
# BL: meta 12 / realizado 12 | TKM: meta 117.9 / realizado 115.8 | Churn: meta 0.08 / realizado 0.02
# Preenche defaults se não encontrados automaticamente
meta_bl_default = meta_bl_default or 12.0
real_bl_default = 12.0
meta_tkm_default = 117.9
real_tkm_default = 115.8
meta_churn_default = 0.08
real_churn_default = 0.02
salario_default = salario_default or 1697.94

# ---------------- UI de parâmetros ----------------
st.subheader("Parâmetros do Simulador")

col1, col2, col3, col4 = st.columns(4)
with col1:
    salario = st.number_input("Salário", min_value=0.0, value=float(salario_default), step=50.0, format="%.2f")

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Banda Larga**")
        meta_bl = st.number_input("Meta (BL)", min_value=0.0, value=float(meta_bl_default), step=1.0)
        real_bl = st.number_input("Realizado (BL)", min_value=0.0, value=float(real_bl_default), step=1.0)
    with c2:
        st.markdown("**Ticket Médio (TKM)**")
        meta_tkm = st.number_input("Meta (TKM)", min_value=0.0, value=float(meta_tkm_default), step=1.0, format="%.2f")
        real_tkm = st.number_input("Realizado (TKM)", min_value=0.0, value=float(real_tkm_default), step=1.0, format="%.2f")
    with c3:
        st.markdown("**Churn Safra** (menor é melhor)")
        meta_churn = st.number_input("Meta (Churn)", min_value=0.0, value=float(meta_churn_default), step=0.01, format="%.4f")
        real_churn = st.number_input("Realizado (Churn)", min_value=0.0001, value=float(real_churn_default), step=0.01, format="%.4f", help="Não pode ser zero.")

# ---------------- Cálculos ----------------
# Atingimentos
ating_bl = (real_bl / meta_bl) if meta_bl else None
ating_tkm = (real_tkm / meta_tkm) if meta_tkm else None
ating_churn = (meta_churn / real_churn) if real_churn else None  # invertido

# Lookup de fatores
f_bl = lookup_factor(ating_bl, faixas, "F_BL")
f_tkm = lookup_factor(ating_tkm, faixas, "F_TKM")
f_churn = lookup_factor(ating_churn, faixas, "F_CHURN")

fatores = {"Banda Larga": f_bl, "Ticket Médio": f_tkm, "Churn Safra": f_churn}
fator_final = sum(v for v in fatores.values() if isinstance(v, (int, float)))
comissao = salario * fator_final if salario is not None else None

# ---------------- Saída ----------------
st.subheader("Resultados")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Fator – Banda Larga", f"{f_bl:.2f}" if f_bl is not None else "—", f"{ating_bl:.4f}" if ating_bl is not None else None, help="Valor da faixa (variação: Delta mostra o atingimento)")
kpi2.metric("Fator – TKM", f"{f_tkm:.2f}" if f_tkm is not None else "—", f"{ating_tkm:.4f}" if ating_tkm is not None else None)
kpi3.metric("Fator – Churn", f"{f_churn:.2f}" if f_churn is not None else "—", f"{ating_churn:.4f}" if ating_churn is not None else None)
kpi4.metric("FATOR FINAL", f"{fator_final:.2f}")

kpi5, kpi6 = st.columns(2)
kpi5.metric("Salário", moeda_br(salario))
kpi6.metric("COMISSÃO", moeda_br(comissao))

st.divider()
st.markdown("### Tabela detalhada")
tabela = pd.DataFrame({
    "KPI": ["Banda Larga", "Ticket Médio", "Churn Safra"],
    "Meta": [meta_bl, meta_tkm, meta_churn],
    "Realizado": [real_bl, real_tkm, real_churn],
    "Atingimento": [ating_bl, ating_tkm, ating_churn],
    "Fator": [f_bl, f_tkm, f_churn],
})
st.dataframe(tabela, use_container_width=True)

csv_bytes = to_csv_bytes(tabela)
st.download_button("⬇️ Baixar resultados (CSV ;)", data=csv_bytes, file_name="simulador_resultados.csv", mime="text/csv")

with st.expander("Sensibilidade rápida (e se...?)"):
    st.write("- Aumentar TKM para **meta atingida** (>= 1,00) costuma subir o fator de 0,04 para 0,06.")
    st.write("- Piorar o Churn até **2,0** de atingimento ainda mantém o **teto** de 0,40; abaixo disso, muda a faixa.")

    st.write("- Reduzir BL de 1,00 para 0,95 derruba fator de 0,18→0,12 (impacto ≈ 0,06 × salário).")
