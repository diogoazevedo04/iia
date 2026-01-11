import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama
import json
import pydeck as pdk
import random

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
st.markdown("Exemplo: *Quero um hotel barato em Londres perto do Big Ben para uma viagem de negócios*")


@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def carregar_dados():
    try:
        # Tenta carregar o PKL primeiro (Código do teu colega)
        df = pd.read_pickle("Hotel_Reviews_processed.pkl")
    except FileNotFoundError:
        try:
            # Se falhar, carrega o CSV (Teu código antigo)
            df = pd.read_csv("Hotel_Reviews_processed.csv")
        except FileNotFoundError:
            return None
    
    # --- LIMPEZAS DE SEGURANÇA (Do teu código original) ---
    df['review'] = df['review'].fillna("Sem comentário.")
    
    if 'Negative_Review' not in df.columns: 
        df['Negative_Review'] = ""
    df['Negative_Review'] = df['Negative_Review'].fillna("")
    
    # Garante que a coluna City existe
    if 'City' not in df.columns:
        df['City'] = df['Hotel_Address'].apply(lambda x: str(x).split()[-1])
        
    if 'Tags_Clean' not in df.columns: 
        df['Tags_Clean'] = ""
    df['Tags_Clean'] = df['Tags_Clean'].fillna("")
    
    if 'Average_Score' not in df.columns: df['Average_Score'] = 0.0

    # Se vier do CSV e for muito grande, fazemos sample. Se vier do PKL, já está pronto.
    if 'embeddings' not in df.columns and len(df) > 3000:
        df = df.sample(n=3000, random_state=42).reset_index(drop=True)
            
    return df

df = carregar_dados()
modelo = carregar_modelo()

if df is None:
    st.error("Erro: Ficheiro 'Hotel_Reviews_processed.csv' não encontrado.")
    st.stop()

# --- LISTA DINÂMICA ---
CIDADES_DISPONIVEIS = sorted(df['City'].dropna().unique().tolist())
st.caption(f"Cidades detetadas no sistema: {', '.join(CIDADES_DISPONIVEIS)}")

if 'embeddings' not in st.session_state:
    if 'embeddings' in df.columns:
        st.success(f"Carregados {len(df)} hotéis com memória pré-calculada!")
        st.session_state['embeddings'] = np.array(df['embeddings'].tolist())
    else:
        with st.spinner('A calcular vetores matemáticos para os hotéis...'):
            st.session_state['embeddings'] = modelo.encode(df['review'].tolist())

# --- EXTRAÇÃO DE ENTIDADES (COM TRIP TYPE) ---
def analisar_pedido_ia(query_utilizador):
    try:
        prompt = f"""
        Analyze this hotel request: "{query_utilizador}"
        
        Task:
        1. Identify the City (Translate to English, e.g., 'Londres'->'London').
        2. Identify technical features/amenities (Translate to English keywords).
        3. Identify specific Point of Interest/Landmark (e.g., 'Eiffel Tower', 'Hyde Park', 'Metro', 'City Center').
        4. Identify Trip Type (Is it 'Business', 'Family', 'Couple', 'Solo' or 'Leisure'?).
        
        Output ONLY valid JSON format:
        {{
            "city": "CityName or null",
            "features": ["feature1", "feature2"],
            "poi": "PointOfInterestName or null",
            "trip_type": "Business/Family/Couple/Solo or null"
        }}
        """
        
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
        conteudo = response['message']['content']
        
        inicio = conteudo.find('{')
        fim = conteudo.rfind('}') + 1
        json_str = conteudo[inicio:fim]
        
        dados = json.loads(json_str)
        
        if dados.get("city") or dados.get("features") or dados.get("poi"):
            return dados

    except Exception:
        pass

    query_lower = query_utilizador.lower()
    
    features_fallback = []
    if "piscina" in query_lower: features_fallback.append("pool")
    if "pequeno" in query_lower: features_fallback.append("breakfast")
    if "ginásio" in query_lower: features_fallback.append("gym")
    if "wifi" in query_lower: features_fallback.append("wifi")
    if "metro" in query_lower or "transporte" in query_lower: features_fallback.append("metro")
    
    cidade_fallback = None
    for cidade in CIDADES_DISPONIVEIS:
        if cidade.lower() in query_lower:
            cidade_fallback = cidade
            break
            
    return {"city": cidade_fallback, "features": features_fallback, "poi": None, "trip_type": None}

def destacar_texto(texto, termos):
    texto_lower = str(texto).lower()
    for t in termos:
        if t and len(t) > 2:
            idx = texto_lower.find(t.lower())
            if idx != -1:
                return f"...{texto[max(0, idx-50):min(len(texto), idx+300)]}..."
    return texto[:300] + "..."


#Interface
pergunta = st.text_input("O que procuras?")
botao = st.button("Pesquisar", type="primary")

if botao and pergunta:
    
    with st.spinner('A IA está a interpretar a tua intenção...'):
        entidades = analisar_pedido_ia(pergunta)
        
        cidade_ia = entidades.get("city")
        features_ia = entidades.get("features", [])
        poi_ia = entidades.get("poi")
        tipo_viagem = entidades.get("trip_type") 
        
        if cidade_ia and cidade_ia != "null":
            cidade_encontrada = False
            for c in CIDADES_DISPONIVEIS:
                if cidade_ia.lower() in c.lower():
                    cidade_ia = c 
                    cidade_encontrada = True
                    break
            
            if not cidade_encontrada:
                st.warning(f"Não encontrámos hotéis em **{cidade_ia}**.")
                st.markdown("### ✈️ No entanto, temos estes destinos fantásticos:")
                
                IMAGENS_CIDADES = {
                    "Amsterdam": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=600&q=80",
                    "Barcelona": "https://images.unsplash.com/photo-1583422409516-2895a77efded?w=600&q=80",
                    "London": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=600&q=80",
                    "Milan": "https://images.unsplash.com/photo-1572602648934-1d98de6dab48?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                    "Paris": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=600&q=80",
                    "Vienna": "https://media.istockphoto.com/id/519836276/photo/vienna-sunrise.jpg?s=612x612&w=0&k=20&c=HRnz-vIh2IVG9XkDF4v3FRk_5eiO0oKEFLlK4O4G2tU="
                }

                sugestoes = random.sample(CIDADES_DISPONIVEIS, min(4, len(CIDADES_DISPONIVEIS)))
                cols = st.columns(4)
                for idx, cidade_sug in enumerate(sugestoes):
                    with cols[idx]:
                        img_url = IMAGENS_CIDADES.get(cidade_sug, "https://placehold.co/600x400")
                        st.image(img_url, use_container_width=True)
                        st.subheader(cidade_sug)
                st.stop()
        
        if check_pool: features_ia.append("pool")
        if check_wifi: features_ia.append("wifi")
        if check_gym: features_ia.append("gym")
        if check_breakfast: features_ia.append("breakfast")
        if poi_ia and poi_ia != "null": features_ia.append(poi_ia)

        col1, col2 = st.columns(2)
        with col1:
            if cidade_ia and cidade_ia != "null": st.info(f"📍 Cidade: **{cidade_ia}**")
            else: st.caption("🌍 Cidade: Qualquer")
            
            if poi_ia and poi_ia != "null": st.info(f"🎯 Ponto de Interesse: **{poi_ia}**")

        with col2:
            if tipo_viagem and tipo_viagem != "null": st.success(f"✈️ Viagem: **{tipo_viagem}**")
            st.success(f"🏷️ Conceitos: {', '.join([f for f in features_ia if f != poi_ia])}")

    with st.spinner('🔍 A cruzar dados...'):
        

        texto_pesquisa = pergunta
        

        termos_extra = []
        if check_pool: termos_extra.append("swimming pool")
        if check_wifi: termos_extra.append("wifi internet")
        if check_breakfast: termos_extra.append("breakfast included morning food")
        if check_gym: termos_extra.append("fitness gym")
        
        if termos_extra:
            texto_pesquisa += " " + " ".join(termos_extra)

        if poi_ia and poi_ia != "null":
            texto_pesquisa += f" near {poi_ia} close to {poi_ia}"
            
        vetor = modelo.encode([texto_pesquisa])
        scores = cosine_similarity(vetor, st.session_state['embeddings'])[0]
        
        df_temp = df.copy()
        df_temp['score'] = scores

        if cidade_ia and cidade_ia != "null":
            match_cidade = df_temp[df_temp['City'].str.lower() == cidade_ia.lower()]
            if not match_cidade.empty:
                df_temp = match_cidade
            else:
                df_temp = df_temp[df_temp['Hotel_Address'].str.contains(cidade_ia, case=False, na=False)]

        filtros_ativos_nomes = []
        
        if check_pool:
            mask_pool = (df_temp['Positive_Review'].str.contains('pool|swimming', case=False, na=False)) | \
                        (df_temp['Tags_Clean'].str.contains('pool|swimming', case=False, na=False))
            df_temp = df_temp[mask_pool]
            filtros_ativos_nomes.append("Piscina")
            
        if check_wifi:
            mask_wifi = (df_temp['Positive_Review'].str.contains('wifi|wi-fi|internet', case=False, na=False)) | \
                        (df_temp['Tags_Clean'].str.contains('wifi|wi-fi|internet', case=False, na=False))
            df_temp = df_temp[mask_wifi]
            filtros_ativos_nomes.append("Wi-Fi")
            
        if check_breakfast:
            mask_bf = (df_temp['Positive_Review'].str.contains('breakfast|buffet|morning|eggs', case=False, na=False)) | \
                      (df_temp['Tags_Clean'].str.contains('breakfast', case=False, na=False))
            df_temp = df_temp[mask_bf]
            filtros_ativos_nomes.append("Pequeno Almoço")
            
        if check_gym:
            mask_gym = (df_temp['Positive_Review'].str.contains('gym|fitness|workout', case=False, na=False)) | \
                       (df_temp['Tags_Clean'].str.contains('gym|fitness', case=False, na=False))
            df_temp = df_temp[mask_gym]
            filtros_ativos_nomes.append("Ginásio")

        if df_temp.empty:
            st.warning(f"⚠️ Não encontrámos resultados em **{cidade_ia}** com esses filtros específicos.")
            st.info("A mostrar resultados baseados apenas na pesquisa de texto e cidade.")
     
            df_temp = df.copy()
            df_temp['score'] = scores
            if cidade_ia:
                 df_temp = df_temp[df_temp['City'].str.lower() == cidade_ia.lower()]
        
        top = df_temp.sort_values(by='score', ascending=False).head(n_results)

       
        st.subheader("🗺️ Localização")
        df_mapa = top[['lat', 'lng', 'Hotel_Name', 'score']].dropna()
        
        if not df_mapa.empty:
            midpoint = (np.average(df_mapa["lat"]), np.average(df_mapa["lng"]))
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11),
                layers=[pdk.Layer(
                    "ScatterplotLayer", data=df_mapa, get_position='[lng, lat]',
                    get_color='[200, 30, 0, 160]', get_radius=200, pickable=True
                )],
                tooltip={"text": "{Hotel_Name}\nScore: {score}"}
            ))

        st.divider()
        st.subheader("O Conselho da IA")
        
       
        contexto_hoteis = ""
        for _, row in top.iterrows():
            review_text = str(row['review'])[:600].replace("\n", " ")
            tags = str(row['Tags_Clean'])[:100]
            contexto_hoteis += f"\n🏨 HOTEL: {row['Hotel_Name']}\n   - Score: {row['score']:.2f}\n   - Tags: {tags}\n   - O que dizem: {review_text}\n"

        msg_filtros = f"O utilizador EXIGE estas comodidades: {', '.join(filtros_ativos_nomes)}." if filtros_ativos_nomes else "O utilizador não escolheu filtros de comodidades."
        msg_poi = f"O hotel deve ser perto de: {poi_ia}." if poi_ia else ""
        msg_trip = f"Tipo de viagem: {tipo_viagem}." if tipo_viagem else ""
        
        prompt_rag = f"""
        Tu és um Consultor de Viagens Especialista e muito persuasivo.
        
        O TEU OBJETIVO: Recomendar o melhor hotel da lista abaixo para o utilizador.
        
        DADOS DO PEDIDO:
        - Pergunta original: "{pergunta}"
        - {msg_filtros}
        - {msg_poi}
        - {msg_trip}

        LISTA DE CANDIDATOS (Já filtrados pelo sistema):
        {contexto_hoteis}

        INSTRUÇÕES PARA A RESPOSTA:
        1. Escolhe O MELHOR hotel desta lista.
        2. Não dês apenas o nome. "Vende" o hotel! Escreve um texto fluido e cativante.
        3. Menciona explicitamente as comodidades pedidas (ex: "O pequeno-almoço é fantástico porque...").
        4. Usa detalhes das reviews fornecidas para justificar a escolha.
        5. Responde sempre em Português de Portugal (PT-PT).
        
        Resposta:
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
            res_box.warning("O Consultor IA está indisponível momentaneamente.")

        st.divider()
        for _, row in top.iterrows():
            with st.expander(f"🏨 {row['Hotel_Name']} ({row['score']:.0%})", expanded=True):
                st.caption(f"📍 {row['Hotel_Address']}")
                
                
                encontrado_em = []
                if check_breakfast:
                    if "breakfast" in str(row['Tags_Clean']).lower(): encontrado_em.append("✅ Tag: Breakfast Included")
                    elif "breakfast" in str(row['Positive_Review']).lower(): encontrado_em.append("✅ Review: Bom pequeno almoço")
                
                if check_pool:
                     if "pool" in str(row['Tags_Clean']).lower() or "pool" in str(row['Positive_Review']).lower(): encontrado_em.append("✅ Piscina Confirmada")

                if encontrado_em:
                    st.success(" | ".join(encontrado_em))
                
               
                termos_para_destaque = filtros_ativos_nomes + features_ia
                st.markdown(f"> {destacar_texto(row['review'], termos_para_destaque)}")
                st.metric("Score", f"{row['score']:.2f}")
