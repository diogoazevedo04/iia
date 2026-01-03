import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama
import json
import pydeck as pdk

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
st.markdown("Exemplo: *Quero um hotel barato em Londres perto do Big Ben*")

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

# --- EXTRAÇÃO DE ENTIDADES (ATUALIZADO PARA POI) ---
def analisar_pedido_ia(query_utilizador):
    """
    Tenta usar o Ollama. Se falhar, usa regras manuais.
    Agora extrai também 'poi' (Pontos de Interesse).
    """
    
    # 1. TENTATIVA COM IA
    try:
        # --- MUDANÇA AQUI: Adicionei o ponto 3 no Prompt ---
        prompt = f"""
        Analyze this hotel request: "{query_utilizador}"
        
        Task:
        1. Identify the City (Translate to English, e.g., 'Londres'->'London').
        2. Identify technical features/amenities (Translate to English keywords).
        3. Identify specific Point of Interest/Landmark (e.g., 'Eiffel Tower', 'Hyde Park', 'Metro', 'City Center').
        
        Output ONLY valid JSON format:
        {{
            "city": "CityName or null",
            "features": ["feature1", "feature2"],
            "poi": "PointOfInterestName or null"
        }}
        """
        
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
        conteudo = response['message']['content']
        
        inicio = conteudo.find('{')
        fim = conteudo.rfind('}') + 1
        json_str = conteudo[inicio:fim]
        
        dados = json.loads(json_str)
        
        # Retorna se tiver alguma informação útil
        if dados.get("city") or dados.get("features") or dados.get("poi"):
            return dados

    except Exception:
        pass

    # 2. PLANO B (FALLBACK MANUAL)
    query_lower = query_utilizador.lower()
    
    features_fallback = []
    if "piscina" in query_lower: features_fallback.append("pool")
    if "pequeno" in query_lower: features_fallback.append("breakfast")
    if "ginásio" in query_lower: features_fallback.append("gym")
    if "wifi" in query_lower: features_fallback.append("wifi")
    
    # Adicionamos "metro" como feature manual
    if "metro" in query_lower or "transporte" in query_lower: 
        features_fallback.append("metro")
    
    cidade_fallback = None
    if "londres" in query_lower or "london" in query_lower: cidade_fallback = "London"
    elif "paris" in query_lower: cidade_fallback = "Paris"
    elif "amesterdão" in query_lower or "amsterdam" in query_lower: cidade_fallback = "Amsterdam"
    elif "milão" in query_lower or "milan" in query_lower: cidade_fallback = "Milan"
    elif "viena" in query_lower or "vienna" in query_lower: cidade_fallback = "Vienna"
    elif "barcelona" in query_lower: cidade_fallback = "Barcelona"
    
    return {"city": cidade_fallback, "features": features_fallback, "poi": None}

def destacar_texto(texto, termos):
    texto_lower = str(texto).lower()
    for t in termos:
        if t and len(t) > 2: # Evita erros com termos vazios
            idx = texto_lower.find(t.lower()) # Garante que procura em minúsculas
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
        poi_ia = entidades.get("poi") # <--- LER O PONTO DE INTERESSE
        
        if check_pool: features_ia.append("pool")
        if check_wifi: features_ia.append("wifi")
        if check_gym: features_ia.append("gym")
        if check_breakfast: features_ia.append("breakfast")
        
        # Se detetou um POI, adiciona à lista de destaques para aparecer a bold
        if poi_ia and poi_ia != "null":
            features_ia.append(poi_ia)

        col1, col2 = st.columns(2)
        with col1:
            if cidade_ia and cidade_ia != "null":
                st.info(f"📍 Cidade: **{cidade_ia}**")
            else:
                st.caption("🌍 Cidade: Qualquer")
                
            # --- MOSTRAR O POI NA INTERFACE ---
            if poi_ia and poi_ia != "null":
                st.info(f"🎯 Ponto de Interesse: **{poi_ia}**")
            # ----------------------------------

        with col2:
            st.success(f"🏷️ Conceitos: {', '.join([f for f in features_ia if f != poi_ia])}")

    with st.spinner('🔍 A cruzar dados...'):
        
        vetor = modelo.encode([pergunta])
        scores = cosine_similarity(vetor, st.session_state['embeddings'])[0]
        
        df_temp = df.copy()
        df_temp['score'] = scores
        
        if cidade_ia and cidade_ia != "null":
            df_temp = df_temp[df_temp['Hotel_Address'].str.contains(cidade_ia, case=False, na=False)]
        
        for feat in features_ia:
            # Não filtramos negativamente pelo POI para não eliminar hotéis bons que apenas não o mencionam na review negativa
            if feat == poi_ia: continue 
            
            if feat in ["quiet", "calm"]:
                df_temp = df_temp[~df_temp['Negative_Review'].str.contains("noise|loud", case=False)]
            else:
                df_temp = df_temp[~df_temp['Negative_Review'].str.contains(feat, case=False)]
        
        top = df_temp.sort_values(by='score', ascending=False).head(n_results)

        if top.empty:
            st.warning(f"Não encontrámos hotéis compatíveis em **{cidade_ia}**.")
        else:
            
            # --- MAPA GERAL INTERATIVO (PyDeck - Corrigido) ---
            st.subheader("🗺️ Localização dos Hotéis")
            
            df_mapa = top[['lat', 'lng', 'Hotel_Name', 'score']].copy()
            df_mapa = df_mapa.dropna()
            
            if not df_mapa.empty:
                midpoint = (np.average(df_mapa["lat"]), np.average(df_mapa["lng"]))
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_mapa,
                    get_position='[lng, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=200,
                    pickable=True,
                )
                
                view_state = pdk.ViewState(
                    latitude=midpoint[0],
                    longitude=midpoint[1],
                    zoom=11,
                    pitch=0,
                )
                
                st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "{Hotel_Name}\nScore: {score}"}
                ))
            else:
                st.caption("Coordenadas indisponíveis para visualizar o mapa.")
            # ------------------------------------------------

            st.divider()
            st.subheader("O Conselho da IA")
            
            contexto_hoteis = ""
            for _, row in top.iterrows():
                contexto_hoteis += f"\n- {row['Hotel_Name']} ({row['Hotel_Address']}): {row['review'][:400]}..."
            
            # Atualizei o prompt para mencionar o POI se existir
            msg_poi = f"O utilizador quer ficar perto de {poi_ia}." if poi_ia else ""
            
            prompt_rag = f"""
            Com base nestes hotéis:
            {contexto_hoteis}
            
            Responde ao pedido do utilizador: "{pergunta}". {msg_poi}
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
                    # Passamos a lista features_ia (que agora inclui o POI) para destacar no texto
                    st.markdown(f"> {destacar_texto(row['review'], features_ia)}")
                    st.metric("Score", f"{row['score']:.2f}")