import os
import streamlit as st
from swarm import Swarm, Agent
from duckduckgo_search import DDGS
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import html

# Configuração inicial
load_dotenv()
MODEL = "llama3.2"  # Verifique o modelo correto
CACHE_TIME = timedelta(hours=1)
MAX_QUERY_LENGTH = 500
DEBUG_MODE = False  # Altere para True para ver informações de debug

# Configurar logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

client = Swarm()
ddgs = DDGS()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def search_web(query):
    """Realiza pesquisa web com tratamento de erros"""
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        results = ddgs.text(
            f"{query} after:{current_date}",
            max_results=10,
            region="br-pt",
            safesearch="Moderate"
        )
        
        if not results:
            return "Nenhum resultado encontrado."
            
        news_results = []
        seen_urls = set()
        for result in results:
            if result['href'] not in seen_urls:
                news_entry = (
                    f"Título: {html.escape(result['title'])}\n"
                    f"URL: {result['href']}\n"
                    f"Descrição: {html.escape(result['body'])}\n"
                )
                news_results.append(news_entry)
                seen_urls.add(result['href'])
                
        return "\n\n".join(news_results) if news_results else "Nenhum resultado válido."
        
    except Exception as e:
        logging.error(f"Erro na pesquisa: {str(e)}")
        return f"Erro na pesquisa: {str(e)}"

# Configuração dos Agentes
web_search_agent = Agent(
    name="Pesquisador Web",
    instructions="Colete artigos e notícias recentes usando DuckDuckGo",
    functions=[search_web],
    model=MODEL
)

researcher_agent = Agent(
    name="Analista de Conteúdo",
    instructions="""Organize e analise o conteúdo coletado:
    1. Estruture o texto em seções temáticas claras com base nos tópicos coletados.
    2. Verifique a validade e relevância das informações, cruzando dados com pelo menos duas fontes confiáveis.
    3. Adicione contexto histórico ou explicativo quando necessário para tornar o conteúdo mais acessível.
    4. Priorize informações relevantes ao público do LinkedIn, como insights acionáveis e tendências de mercado.""",
    model=MODEL
)

writer_agent = Agent(
    name="Redator Profissional",
    instructions="""Escreva um artigo de notícias em formato markdown baseado no conteúdo analisado:
    0. Formato markdown sem emojis
    1. Estruture em seções: Introdução, Tópicos Principais, Conclusão.
    2. Adicione subtítulos claros e utilize listas quando necessário para facilitar a leitura.
    3. Mantenha um tom profissional, informativo e engajador, apropriado para redes profissionais como LinkedIn.
    4. Garanta fluidez e transição entre tópicos, evitando redundâncias.
    5. Use dados coletados pelo analista, citando fontes no final do artigo (quando aplicável)
    6. Caso o texto contenha termos técnicos em inglês, mantenha-os em inglês.""",
    model=MODEL
)

proofreader_agent = Agent(
    name="Revisor Final",
    instructions="""Realize uma revisão detalhada do artigo, corrigindo:
    1. Erros gramaticais, ortográficos e de concordância em português.
    2. O estilo, mantendo o tom profissional e objetivo.
    3. A formatação em markdown, garantindo que títulos, subtítulos e listas estejam claros.
    4. Remover qualquer menção a limitação dos agentes Swarm ou do LLaMA
    5, Garanta que não tenha emojis no texto final.""",
    model=MODEL
)

def validate_input(query):
    """Valida e sanitiza a entrada do usuário"""
    if not query or len(query.strip()) < 5:
        raise ValueError("Por favor insira um tópico válido (mínimo 5 caracteres)")
    
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Tópico muito longo (máximo {MAX_QUERY_LENGTH} caracteres)")
        
    return html.escape(query.strip())

def debug_step(content, step_name):
    """Exibe informações de debug quando ativado"""
    if DEBUG_MODE:
        st.write(f"""
        ### Debug: {step_name}
        **Tipo:** `{type(content)}`  
        **Tamanho:** `{len(content)} caracteres`  
        **Prévia:**  
        ```text
        {content[:200]}{'...' if len(content) > 200 else ''}
        ```
        """)

@st.cache_data(ttl=CACHE_TIME, show_spinner=False)
def process_research(query):
    """Processa a pesquisa e análise"""
    try:
        # Pesquisa Web
        search_result = client.run(
            agent=web_search_agent,
            messages=[{"role": "user", "content": f"Pesquise sobre: {query}"}]
        )
        raw_data = search_result.messages[-1]["content"]
        debug_step(raw_data, "Dados Brutos da Pesquisa")

        # Análise de Conteúdo
        analysis_result = client.run(
            agent=researcher_agent,
            messages=[{"role": "user", "content": f"Analise estes dados:\n{raw_data}"}]
        )
        clean_data = analysis_result.messages[-1]["content"]
        debug_step(clean_data, "Dados Processados")

        # Redação Inicial
        draft_result = client.run(
            agent=writer_agent,
            messages=[{"role": "user", "content": f"Crie um artigo usando:\n{clean_data}"}]
        )
        return draft_result.messages[-1]["content"]

    except Exception as e:
        logging.error(f"Erro no processamento: {str(e)}")
        raise

def main():
    """Interface principal"""
    st.set_page_config(
        page_title="Gerador de Artigos",
        page_icon="📄",
        layout="centered"
    )
    
    st.title("📄 Gerador de Artigos")
    st.markdown("---")
    
    # Gerenciamento de estado
    if 'article' not in st.session_state:
        st.session_state.article = ""
    
    # Formulário de entrada
    with st.form(key="main_form"):
        query_input = st.text_input(
            "Sobre qual tema deseja escrever?",
            placeholder="Ex: Impacto da IA no mercado de trabalho 2024",
            max_chars=MAX_QUERY_LENGTH
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_btn = st.form_submit_button("Gerar Artigo")
        with col2:
            clear_btn = st.form_submit_button("Limpar Tudo")

    if clear_btn:
        st.session_state.clear()
        st.rerun()

    if submit_btn and query_input:
        try:
            clean_query = validate_input(query_input)
            
            with st.status("Processando...", expanded=True) as status:
                # Pesquisa e análise
                st.write("🔍 Realizando pesquisa web...")
                article_draft = process_research(clean_query)
                
                # Geração do artigo com streaming
                st.write("✍️ Gerando artigo em tempo real...")
                proofread_stream = client.run(
                    agent=proofreader_agent,
                    messages=[{"role": "user", "content": f"Revise:\n{article_draft}"}],
                    stream=True
                )

                # Configurar área de streaming
                article_placeholder = st.empty()
                full_response = ""
                
                # Processar cada chunk do stream
                for chunk in proofread_stream:
                    if chunk.get('content'):
                        full_response += chunk['content']
                        # Atualizar o texto em tempo real com efeito de digitação
                        article_placeholder.markdown(full_response + "▌")
                
                # Atualizar estado e interface
                article_placeholder.markdown(full_response)
                st.session_state.article = full_response
                status.update(label="Artigo completo! ✅", state="complete")

        except Exception as e:
            st.error(f"Erro: {str(e)}")

    # Exibição do resultado final
    if st.session_state.article:
        st.markdown("---")
        st.subheader("Seu Artigo Pronto")
        st.markdown(st.session_state.article)
        
        st.download_button(
            label="Baixar Artigo",
            data=st.session_state.article,
            file_name="artigo_llm.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()