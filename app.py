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

modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_pickle("Hotel_Reviews_processed.pkl")
    except FileNotFoundError:
        try:
            df = pd.read_csv("Hotel_Reviews_processed.csv")
        except FileNotFoundError:
            return None

    return df

df = carregar_dados()

if df is None:
    st.error("Erro: Ficheiros 'Hotel_Reviews_processed.csv' ou 'Hotel_Reviews_processed.pkl' não encontrados.")
    st.stop()

CIDADES_DISPONIVEIS = sorted(df['City'].dropna().unique().tolist())
st.caption(f"Cidades disponíveis no sistema: {', '.join(CIDADES_DISPONIVEIS)}")

if 'embeddings' not in st.session_state:
    if 'embeddings' in df.columns:
        st.session_state['embeddings'] = df['embeddings'].tolist()
    else:
        with st.spinner('A calcular embeddings para os hotéis...'):
            st.session_state['embeddings'] = modelo.encode(df['review'].tolist())

# --- EXTRAÇÃO DE ENTIDADES ---
def analisar_pedido_ia(query_utilizador):
    try:
        prompt = f"""
        Analyze this hotel request: "{query_utilizador}"
        
        Task:
        1. Identify the City (Translate to English, e.g., 'Londres'->'London'). Use null if no city is mentioned.
        2. Identify ONLY EXPLICITLY MENTIONED features/amenities (Translate to English keywords), e.g., 'piscina'->'pool', 'pequeno almoço'->'breakfast', 'ginásio'->'gym', 'wifi'->'wifi'. DO NOT infer features based on trip type. Also include price concepts as features if present: use 'cheap'/'budget' for low price intent, or 'luxury' for high price intent.
        3. Identify specific Point of Interest/Landmark (e.g., 'Eiffel Tower', 'Hyde Park', 'Metro', 'City Center'). Use null if none mentioned.
        4. Identify Trip Type (Is it 'Business', 'Family', 'Couple', 'Solo' or 'Leisure'?). Use null if not specified.
        
        EXAMPLES:
        
        User Input: "Quero um hotel barato em Paris com piscina e wifi."
        JSON Output: {{"city": "Paris", "features": ["cheap", "pool", "wifi"], "poi": null, "trip_type": null}}
        
        User Input: "Vou a Londres em trabalho. Preciso de ficar perto do Big Ben."
        JSON Output: {{"city": "London", "features": [], "poi": "Big Ben", "trip_type": "Business"}}
        
        User Input: "Hotel de luxo para casal com ginásio e pequeno-almoço."
        JSON Output: {{"city": null, "features": ["luxury", "gym", "breakfast"], "poi": null, "trip_type": "Couple"}}
        
        User Input: "Quero um hotel para uma viagem de negócios."
        JSON Output: {{"city": null, "features": [], "poi": null, "trip_type": "Business"}}
        
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
        
        # Remove valores null ou "null"
        dados_limpo = {k: v for k, v in dados.items() if v is not None and v != "null"}
        
        if dados_limpo.get("city") or dados_limpo.get("features") or dados_limpo.get("poi"):
            return dados_limpo

    except Exception:
        pass

    query_lower = query_utilizador.lower()
    
    # Mapa de traduções de cidades (português -> inglês)
    cidade_traduzidas = {
        "milão": "Milan",
        "milao": "Milan",
        "milano": "Milan",
        "londres": "London",
        "paris": "Paris",
        "viena": "Vienna",
        "vienna": "Vienna",
        "amesterdão": "Amsterdam",
        "amesterdao": "Amsterdam",
        "barcelona": "Barcelona",
    }
    
    features_fallback = []
    if "piscina" in query_lower: features_fallback.append("pool")
    if "pequeno" in query_lower: features_fallback.append("breakfast")
    if "ginásio" in query_lower: features_fallback.append("gym")
    if "wifi" in query_lower: features_fallback.append("wifi")
    if "metro" in query_lower or "transporte" in query_lower: features_fallback.append("metro")
    
    cidade_fallback = None
    # Primeiro tenta com traduções
    for pt_city, en_city in cidade_traduzidas.items():
        if pt_city in query_lower:
            cidade_fallback = en_city
            break
    
    # Se não encontrou, tenta busca direta
    if not cidade_fallback:
        for cidade in CIDADES_DISPONIVEIS:
            if cidade.lower() in query_lower:
                cidade_fallback = cidade
                break
    
    tipo_viagem_fallback = None
    if any(p in query_lower for p in ["negócio", "business", "trabalho"]): tipo_viagem_fallback = "Business"
    elif any(p in query_lower for p in ["família", "family", "filhos", "crianças"]): tipo_viagem_fallback = "Family"
    elif any(p in query_lower for p in ["casal", "couple", "romântico", "romantic"]): tipo_viagem_fallback = "Couple"
    elif any(p in query_lower for p in ["solo", "sozinho", "single"]): tipo_viagem_fallback = "Solo"

    # Detetar intenção de preço e incluir como conceito nas features
    cheap_words = ["barato", "barat", "económico", "economico", "budget", "acessível", "acessivel", "low cost", "low-cost", "cheap"]
    expensive_words = ["caro", "luxo", "luxury", "caríssimo", "carissimo", "premium", "5 estrelas", "5-estrelas", "5 star", "5-star", "expensive"]
    if any(w in query_lower for w in cheap_words):
        if "cheap" not in features_fallback and "budget" not in features_fallback:
            features_fallback.append("cheap")
    elif any(w in query_lower for w in expensive_words):
        if "luxury" not in features_fallback:
            features_fallback.append("luxury")
    
    poi_fallback = None
    poi_map = {
        "metro": "Metro",
        "estação": "Station",
        "aeroporto": "Airport",
        "centro": "City Center",
        "big ben": "Big Ben",
        "eiffel": "Eiffel Tower",
        "hyde park": "Hyde Park",
        "torre": "Tower",
        "museu": "Museum",
        "praia": "Beach",
        "parque": "Park",
        "shopping": "Shopping Mall",
    }
    for palavra, poi_nome in poi_map.items():
        if palavra in query_lower:
            poi_fallback = poi_nome
            break
                
    return {"city": cidade_fallback, "features": features_fallback, "poi": poi_fallback, "trip_type": tipo_viagem_fallback}

# --- INTERFACE E LÓGICA ---

pergunta = st.text_input("O que procuras?")
botao = st.button("Pesquisar", type="primary")

if botao and pergunta:
    
    with st.spinner('A IA está a interpretar a tua intenção...'):
        entidades = analisar_pedido_ia(pergunta)
        
        cidade_ia = entidades.get("city")
        features_ia = entidades.get("features", [])
        poi_ia = entidades.get("poi")
        tipo_viagem = entidades.get("trip_type")
        
        # Normalizar cidades (tratar variações)
        cidade_normalizacoes = {
            "milano": "Milan",
            "milao": "Milan",
            "milão": "Milan",
            "londres": "London",
            "paris": "Paris",
            "viena": "Vienna",
            "vienna": "Vienna",
            "amesterdão": "Amsterdam",
            "amesterdao": "Amsterdam",
            "barcelona": "Barcelona",
        }
        if cidade_ia and cidade_ia.lower() in cidade_normalizacoes:
            cidade_ia = cidade_normalizacoes[cidade_ia.lower()]
        
        if cidade_ia and cidade_ia != "null":
            cidade_encontrada = False
            for c in CIDADES_DISPONIVEIS:
                if cidade_ia.lower() in c.lower():
                    cidade_ia = c 
                    cidade_encontrada = True
                    break
            
            if not cidade_encontrada:
                st.warning(f"Não encontrámos hotéis em **{cidade_ia}**.")
                st.markdown("### No entanto, temos estes destinos fantásticos:")
                
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

        col1, col2 = st.columns(2)
        with col1:
            if cidade_ia and cidade_ia != "null": st.info(f"📍 Cidade: **{cidade_ia}**")
            else: st.caption("🌍 Cidade: Qualquer")
            
            if poi_ia and poi_ia != "null": st.info(f"🎯 Ponto de Interesse: **{poi_ia}**")

        with col2:
            if tipo_viagem and tipo_viagem != "null": st.success(f"✈️ Viagem: **{tipo_viagem}**")
            st.success(f"🏷️ Conceitos: {', '.join([f for f in features_ia if f != poi_ia])}")

    with st.spinner('🔍 A cruzar dados...'):
        
        # --- 1. Preparar a Pesquisa Semântica (IA) ---
        texto_pesquisa = pergunta
        
        # Juntar termos em Inglês para ajudar a IA
        termos_extra = []
        if check_pool: termos_extra.append("swimming pool")
        if check_wifi: termos_extra.append("wifi internet")
        if check_breakfast: termos_extra.append("breakfast included morning food")
        if check_gym: termos_extra.append("fitness gym")
        
        # Incluir termos de preço conforme a intenção
        pergunta_lower = pergunta.lower()
        cheap_words = ["barato", "barat", "económico", "economico", "budget", "acessível", "acessivel", "low cost", "low-cost", "cheap"]
        expensive_words = ["luxuoso","caro", "luxo", "luxury", "caríssimo", "carissimo", "premium", "5 estrelas", "5-estrelas", "5 star", "5-star", "expensive"]
        
        # Preferência de preço derivada de conceitos e texto
        feature_text = " ".join(features_ia).lower()
        pref_cheap = any(w in pergunta_lower for w in cheap_words) or ("cheap" in feature_text or "budget" in feature_text)
        pref_expensive = any(w in pergunta_lower for w in expensive_words) or ("luxury" in feature_text or "5-star" in feature_text or "5 star" in feature_text)
        if pref_cheap:
            termos_extra.append("cheap affordable budget good value")
        elif pref_expensive:
            termos_extra.append("luxury premium upscale 5-star five star")
        
        if termos_extra:
            texto_pesquisa += " " + " ".join(termos_extra)

        if poi_ia and poi_ia != "null":
            texto_pesquisa += f" near {poi_ia} close to {poi_ia}"
            
        vetor = modelo.encode([texto_pesquisa])
        scores = cosine_similarity(vetor, st.session_state['embeddings'])[0]
        
        df_temp = df.copy()
        df_temp['score'] = scores
        
        # --- 2. Filtro de Cidade ---
        if cidade_ia and cidade_ia != "null":
            match_cidade = df_temp[df_temp['City'].str.lower() == cidade_ia.lower()]
            if not match_cidade.empty:
                df_temp = match_cidade
            else:
                df_temp = df_temp[df_temp['Hotel_Address'].str.contains(cidade_ia, case=False, na=False)]

        # --- 3. FILTROS ---
        filtros_ativos_nomes = []
        
        if check_pool:
            # Procura em Tags e Reviews Positivas
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

        # --- 4. TIPO DE VIAGEM (estrito se houver correspondências, senão reforço) ---
        if tipo_viagem and tipo_viagem != "null":
            trip_patterns = {
                "Business": r"business|work|conference|trade fair|expo",
                "Family": r"family",
                "Couple": r"couple|romantic",
                "Solo": r"solo|single",
                "Leisure": r"leisure"
            }
            pattern = trip_patterns.get(tipo_viagem, tipo_viagem.lower())
            mask_trip = df_temp['Tags_Clean'].str.contains(pattern, case=False, na=False)

            if mask_trip.any():
                df_temp = df_temp[mask_trip]
                filtros_ativos_nomes.append(f"Tipo de viagem: {tipo_viagem}")
            else:
                # Sem matches estritos: aplicar reforço/penalização suave
                if tipo_viagem == "Business":
                    df_temp.loc[df_temp['Tags_Clean'].str.contains(r"business", case=False, na=False), 'score'] = df_temp.loc[df_temp['Tags_Clean'].str.contains(r"business", case=False, na=False), 'score'] + 0.12
                    df_temp.loc[df_temp['Tags_Clean'].str.contains(r"leisure", case=False, na=False), 'score'] = df_temp.loc[df_temp['Tags_Clean'].str.contains(r"leisure", case=False, na=False), 'score'] - 0.08
                elif tipo_viagem == "Family":
                    df_temp.loc[df_temp['Tags_Clean'].str.contains(r"family", case=False, na=False), 'score'] = df_temp.loc[df_temp['Tags_Clean'].str.contains(r"family", case=False, na=False), 'score'] + 0.12
                elif tipo_viagem == "Couple":
                    df_temp.loc[df_temp['Tags_Clean'].str.contains(r"couple|romantic", case=False, na=False), 'score'] = df_temp.loc[df_temp['Tags_Clean'].str.contains(r"couple|romantic", case=False, na=False), 'score'] + 0.12
                elif tipo_viagem == "Solo":
                    df_temp.loc[df_temp['Tags_Clean'].str.contains(r"solo|single", case=False, na=False), 'score'] = df_temp.loc[df_temp['Tags_Clean'].str.contains(r"solo|single", case=False, na=False), 'score'] + 0.12

        # --- 5. FILTRO DE PREÇO (boost/penalização suave) ---
        if pref_cheap:
            mask_barato = (
                df_temp['Positive_Review'].str.contains('value|cheap|affordable|budget|price|good deal', case=False, na=False)
            ) | (
                df_temp['Tags_Clean'].str.contains('budget|value|cheap|affordable', case=False, na=False)
            )
            df_temp.loc[mask_barato, 'score'] = df_temp.loc[mask_barato, 'score'] + 0.10
            mask_caro = df_temp['Negative_Review'].str.contains('expensive|overpriced|pricey|costly', case=False, na=False)
            df_temp.loc[mask_caro, 'score'] = df_temp.loc[mask_caro, 'score'] - 0.05
        elif pref_expensive:
            mask_lux = (
                df_temp['Positive_Review'].str.contains('luxury|upscale|premium|5[- ]?star|boutique', case=False, na=False)
            ) | (
                df_temp['Tags_Clean'].str.contains('luxury|premium|5[- ]?star|boutique', case=False, na=False)
            )
            df_temp.loc[mask_lux, 'score'] = df_temp.loc[mask_lux, 'score'] + 0.1
            mask_budget = df_temp['Positive_Review'].str.contains('cheap|budget|basic', case=False, na=False)
            df_temp.loc[mask_budget, 'score'] = df_temp.loc[mask_budget, 'score'] - 0.05

        # --- 7. Verificação de Falha ---
        if df_temp.empty:
            st.error(f"Não encontrámos resultados em **{cidade_ia}** com esses filtros específicos.")
            st.info("Sugestão: O hotel pode ter essa comodidade mas não estar explicito nas reviews/tags deste dataset.")
            # Fallback: mostra resultados sem filtros para não ficar vazio
            df_temp = df.copy()
            df_temp['score'] = scores
            if cidade_ia:
                 df_temp = df_temp[df_temp['City'].str.lower() == cidade_ia.lower()]
        
        # 6. Normaliza nomes e remove duplicatas (mantém o score mais alto)
        df_temp['Hotel_Name_Clean'] = df_temp['Hotel_Name'].str.strip().str.lower()
        df_temp = df_temp.sort_values(by='score', ascending=False).drop_duplicates(subset=['Hotel_Name_Clean'], keep='first')
        
        top = df_temp.head(n_results)

        # --- EXIBIÇÃO ---
        st.subheader("Localização")
        df_mapa = top[['lat', 'lng', 'Hotel_Name', 'score']].dropna()
        
        if not df_mapa.empty:
            midpoint = (np.average(df_mapa["lat"]), np.average(df_mapa["lng"]))
            
            # Calcula spread geográfico para ajustar zoom
            lat_range = df_mapa["lat"].max() - df_mapa["lat"].min()
            lng_range = df_mapa["lng"].max() - df_mapa["lng"].min()
            max_range = max(lat_range, lng_range)
            
            # Zoom do mapa dinâmico baseado na distância
            if max_range > 5:      # Países diferentes (> 5 graus)
                zoom_level = 3
            elif max_range > 1:    # Estados/regiões diferentes
                zoom_level = 6
            elif max_range > 0.1:  # Cidades
                zoom_level = 9
            else:                  # Mesmo bairro
                zoom_level = 12
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_mapa,
                get_position='[lng, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
                radius_min_pixels=3,
                radius_max_pixels=12,
                pickable=True,
            )
            
            view_state = pdk.ViewState(
                latitude=midpoint[0],
                longitude=midpoint[1],
                zoom=zoom_level,
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


        st.divider()
        st.subheader("O Conselho da IA")
        
        contexto_hoteis = ""
        for _, row in top.iterrows():
            review_text = str(row['review'])[:600].replace("\n", " ")
            tags = str(row['Tags_Clean'])[:100]
            contexto_hoteis += f"\nHOTEL: {row['Hotel_Name']}\n   - Match_score: {row['score']:.2f}\n   - Evaluation: {row.get('Average_Score', 'N/A')}/10\n   - Tags: {tags}\n   - What people say: {review_text}\n"

        msg_filtros = f"Mandatory Requirements: {', '.join(filtros_ativos_nomes)}." if filtros_ativos_nomes else ""
        msg_poi = f"Important Requirement: Proximity to {poi_ia}" if poi_ia and poi_ia != "null" else ""
        msg_trip = f"Trip Type: {tipo_viagem}." if tipo_viagem and tipo_viagem != "null" else ""
        
        prompt_rag = f"""
        You are an expert travel consultant specializing in hotel recommendations.
        
        ### USER INPUTS:
        - Customer Query: "{pergunta}"
        - {msg_filtros}
        - {msg_poi}
        - {msg_trip}
        
        ### CANDIDATE HOTELS (Pre-filtered list):
        {contexto_hoteis}
        
        ### YOUR GOAL:
        Recommend the SINGLE BEST hotel from the candidate list based on the customer's requirements.
        
        ### CRITICAL INSTRUCTIONS:
        1. **LOCATION IS PRIORITY:** If the user asked for proximity to a specific place (POI, metro, station), the recommended hotel MUST be provably close to it based on the data. Discard any hotel that fails this.
        2. Secondary factors: Consider trip type, amenities, price, and overall quality. If the client asked for "luxury", don't recommend budget hotels, and vice versa.
        3. If no hotel fits the essential requirements, state that honestly.

        ### RESPONSE STRUCTURE:
        - Start with the **Hotel Name** in bold \n.
        - Explain why it meets the main requirements (especially location).
        - "Sell" the hotel! Write a fluid, engaging, and persuasive text on why it is the best choice.
        - Use specific details from the provided reviews/tags to justify the choice.
        - If the user chooses a POI, calculate the distance from each hotel to the POI and confirm the closest one is selected.
        - Briefly explain why the other candidates are less suitable.
        
        ### LANGUAGE QUALITY RULE:
        - FORBIDDEN: English words (money, value, etc), Spanish words (desayuno) or language mixing
        - MANDATORY: Fluid and natural European Portuguese. Write as if you were a human consultant.
        - If you find translation errors in your response, correct them before sending.
        
        ### CONSTRAINTS:
        - **Language:** The output must be strictly in **Portuguese (Portugal) fluid and natural.**.
        - **Length:** Maximum 7-8 sentences.
        - **Formatting:** Do not use introductory filler (e.g., "Here is my recommendation"). Output only the final text.
        """
        
        res_box = st.empty()
        full_res = ""
        try:
            stream = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_rag}], stream=True)
            for chunk in stream:
                full_res += chunk['message']['content']
                res_box.info(full_res + "▌")
            
            full_res = full_res.replace("se encontra", "encontra-se")
            full_res = full_res.replace("se localiza", "localiza-se")
            full_res = full_res.replace("valor por dinheiro", "relação preço/qualidade")
            full_res = full_res.replace("estada", "estadia")
            full_res = full_res.replace("desjejum", "pequeno-almoço")
            full_res = full_res.replace("desjeito", "pequeno-almoço")
            full_res = full_res.replace("você", "si")
            
            res_box.success(full_res)
        except Exception:
            res_box.warning("O Consultor IA está indisponível momentaneamente.")

        st.divider()
        for _, row in top.iterrows():
            avg_score = row.get('Average_Score', 'N/A')
            score_display = f"{avg_score}/10" if avg_score != 'N/A' else 'N/A'
                
            with st.expander(f"🏨 {row['Hotel_Name']} - Match: {row['score']:.0%} · Avaliação: {score_display}", expanded=True):
                st.caption(f"📍 {row['Hotel_Address']}")
                
                # Descrição rápida gerada pelo Ollama
                try:
                    prompt_desc = f"""
                    Based on these hotel reviews, write a very short general description (1-2 sentences, max 25 words) in European Portuguese about the hotel overall.
                    
                    Positive: "{row.get('Positive_Review', '')[:300]}"
                    Tags: "{row.get('Tags_Clean', '')}"
                    
                    Rules:
                    - One or two short sentences only
                    - Focus on the hotel's location, overall atmosphere, facilities, or main strengths
                    - Do NOT focus only on rooms; describe the hotel as a whole
                    - Be positive and highlight general appeal
                    - No introductions, just the general description
                    """
                    
                    resp_desc = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_desc}])
                    descricao = resp_desc['message']['content'].strip()
                    descricao = descricao.replace("desjejum", "pequeno-almoço").replace("desjeito", "pequeno-almoço")
                    st.markdown(f"*{descricao}*")
                except:
                    pass
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Match com Pesquisa", f"{row['score']:.0%}")
                with col_s2:
                    st.metric("Avaliação Geral", score_display)
                
                if tipo_viagem: st.caption(f"🏷️ Tags: {str(row['Tags_Clean'])[:100]}...")