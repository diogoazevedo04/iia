import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama
import json


st.set_page_config(
    page_title="Recomendações de Hotéis ", 
    page_icon="🏨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("Filtros Rápidos")
    
    check_pool = st.checkbox("Piscina")
    check_wifi = st.checkbox("Wi-Fi")
    check_breakfast = st.checkbox("Peq. Almoço")
    check_gym = st.checkbox("Ginásio")
    
    st.divider()
    n_results = st.slider("Nº de Recomendações", 1, 5, 3)

st.title("🏨 Recomendações de Hotéis Inteligente")
st.markdown("Exemplo: *Quero um hotel barato em Londres perto do metro*")

# --- CARREGAMENTO DE DADOS ---
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("Hotel_Reviews_processed.csv")
        
        df['review'] = df['review'].fillna("Sem comentário.")
        if 'Negative_Review' not in df.columns: df['Negative_Review'] = ""
        df['Negative_Review'] = df['Negative_Review'].fillna("")
        
        if len(df) > 3000:
            df = df.sample(n=3000, random_state=42).reset_index(drop=True)
            
        return df
    except FileNotFoundError:
        return None

df = carregar_dados()
modelo = carregar_modelo()

if df is None:
    st.error("Erro: Ficheiro 'Hotel_Reviews_processed.csv' não encontrado.")
    st.stop()

if 'embeddings' not in st.session_state:
    with st.spinner('A calcular vetores matemáticos para os hotéis...'):
        st.session_state['embeddings'] = modelo.encode(df['review'].tolist())

# --- EXTRAÇÃO DE ENTIDADES ---
def analisar_pedido_ia(query_utilizador):
    """
    Tenta usar o Ollama. Se falhar, usa regras manuais (Fallback) 
    para não dar erro na app.
    """
    
    # 1. TENTATIVA COM IA
    try:
        prompt = f"""
        Analyze this hotel request: "{query_utilizador}"
        
        Task:
        1. Identify the City (Translate to English, e.g., 'Londres'->'London', 'Milão'->'Milan').
        2. Identify technical features/amenities (Translate to English keywords).
        
        Output ONLY valid JSON format:
        {{
            "city": "CityName or null",
            "features": ["feature1", "feature2"]
        }}
        """
        
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
        conteudo = response['message']['content']
        
        inicio = conteudo.find('{')
        fim = conteudo.rfind('}') + 1
        json_str = conteudo[inicio:fim]
        
        dados = json.loads(json_str)
        
        # Se a IA devolveu algo útil, retorna.
        if dados.get("city") or dados.get("features"):
            return dados

    except Exception:
        # Se der erro no Ollama, passamos para o plano B
        pass

    # 2. PLANO B (FALLBACK MANUAL) - Só corre se a IA falhar
    query_lower = query_utilizador.lower()
    
    features_fallback = []
    if "piscina" in query_lower: features_fallback.append("pool")
    if "pequeno" in query_lower: features_fallback.append("breakfast")
    if "ginásio" in query_lower: features_fallback.append("gym")
    if "wifi" in query_lower: features_fallback.append("wifi")
    
    cidade_fallback = None
    if "londres" in query_lower or "london" in query_lower: cidade_fallback = "London"
    elif "paris" in query_lower: cidade_fallback = "Paris"
    elif "amesterdão" in query_lower or "amsterdam" in query_lower: cidade_fallback = "Amsterdam"
    elif "milão" in query_lower or "milan" in query_lower: cidade_fallback = "Milan"
    elif "viena" in query_lower or "vienna" in query_lower: cidade_fallback = "Vienna"
    elif "barcelona" in query_lower: cidade_fallback = "Barcelona"
    
    return {"city": cidade_fallback, "features": features_fallback}

def destacar_texto(texto, termos):
    texto_lower = str(texto).lower()
    for t in termos:
        idx = texto_lower.find(t)
        if idx != -1:
            return f"...{texto[max(0, idx-50):min(len(texto), idx+300)]}..."
    return texto[:300] + "..."

# --- INTERFACE E LÓGICA ---

pergunta = st.text_input("O que procuras?")
botao = st.button("Pesquisar", type="primary")

if botao and pergunta:
    
    with st.spinner('A IA está a interpretar a tua intenção...'):
        entidades = analisar_pedido_ia(pergunta)
        
        cidade_ia = entidades.get("city")
        features_ia = entidades.get("features", [])
        
        if check_pool: features_ia.append("pool")
        if check_wifi: features_ia.append("wifi")
        if check_gym: features_ia.append("gym")
        if check_breakfast: features_ia.append("breakfast")
        
        
        col1, col2 = st.columns(2)
        with col1:
            if cidade_ia and cidade_ia != "null":
                st.info(f"Cidade Detetada: **{cidade_ia}**")
            else:
                st.caption("Nenhuma cidade específica identificada.")
        with col2:
            st.success(f"Conceitos a validar: {', '.join(features_ia)}")

    with st.spinner('🔍 A cruzar dados...'):
        
        vetor = modelo.encode([pergunta])
        scores = cosine_similarity(vetor, st.session_state['embeddings'])[0]
        
        df_temp = df.copy()
        df_temp['score'] = scores
        
        if cidade_ia and cidade_ia != "null":
            df_temp = df_temp[df_temp['Hotel_Address'].str.contains(cidade_ia, case=False, na=False)]
        
        for feat in features_ia:
            if feat in ["quiet", "calm"]:
                df_temp = df_temp[~df_temp['Negative_Review'].str.contains("noise|loud", case=False)]
            else:
                df_temp = df_temp[~df_temp['Negative_Review'].str.contains(feat, case=False)]
        
        top = df_temp.sort_values(by='score', ascending=False).head(n_results)

        if top.empty:
            st.warning(f"Não encontrámos hotéis compatíveis em **{cidade_ia}**.")
        else:
            st.divider()
            st.subheader("O Conselho da IA")
            
            contexto_hoteis = ""
            for _, row in top.iterrows():
                contexto_hoteis += f"\n- {row['Hotel_Name']} ({row['Hotel_Address']}): {row['review'][:400]}..."
            
            prompt_rag = f"""
            Com base nestes hotéis:
            {contexto_hoteis}
            
            Responde ao pedido do utilizador: "{pergunta}".
            Recomenda o melhor e explica porquê em Português de Portugal.
            """
            
            res_box = st.empty()
            full_res = ""
            
            try:
                stream = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_rag}], stream=True)
                for chunk in stream:
                    full_res += chunk['message']['content']
                    res_box.info(full_res + "▌")
                res_box.success(full_res)
            except Exception:
                res_box.warning("O Consultor IA está ocupado, mas aqui estão os melhores resultados encontrados:")

            st.divider()
            for _, row in top.iterrows():
                with st.expander(f"🏨 {row['Hotel_Name']} ({row['score']:.0%})", expanded=True):
                    st.caption(f"📍 {row['Hotel_Address']}")
                    st.markdown(f"> {destacar_texto(row['review'], features_ia)}")
                    st.metric("Score", f"{row['score']:.2f}")