import streamlit as st
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langfuse.langchain import CallbackHandler
import os
import re
import time
from dotenv import load_dotenv
import hashlib

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file. Please add it.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found in .env file. Please add it.")
    st.stop()

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Initialize Langfuse callback handler for observability
# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from .env automatically
langfuse_handler = CallbackHandler()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = 0

MAX_QUERIES_PER_SESSION = 50
MIN_SECONDS_BETWEEN_QUERIES = 1

# ============================================================================
# MODELS INITIALIZATION
# ============================================================================

@st.cache_resource
def load_models():
    """Load embedding model, vector store, and LLM"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vector_store = PineconeVectorStore.from_existing_index("hello", embedding_model)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=512,
            openai_api_key=OPENAI_API_KEY,
            request_timeout=30,
            max_retries=2
        )

        topic_model = SentenceTransformer('all-MiniLM-L6-v2')

        return embedding_model, vector_store, llm, topic_model

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

embedding_model, vector_store, llm, topic_model = load_models()

# ============================================================================
# ANTI-HALLUCINATION: STRICT PROMPT TEMPLATE
# ============================================================================

prompt_template = """You are an AI policy expert. You MUST follow these rules strictly:

1. ONLY use information from the provided context below
2. If the context doesn't contain enough information to answer, say "I don't have sufficient information in my knowledge base to answer that question."
3. Do NOT make up facts, speculate, or use external knowledge
4. Be specific and cite information from the context when possible
5. If you're uncertain about any part of the answer, express that uncertainty clearly

Context:
{context}

Question: {question}

Provide a clear, factual answer based ONLY on the context above. If the context is insufficient, say so explicitly.

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ============================================================================
# ANTI-HALLUCINATION: RETRIEVER WITH SIMILARITY THRESHOLD
# ============================================================================

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7
    }
)

# ============================================================================
# QA CHAIN WITH SOURCE DOCUMENTS
# ============================================================================

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ============================================================================
# INPUT GUARDRAILS
# ============================================================================

def input_guardrail(query):
    """
    Validate user input before processing.
    Returns: (is_valid: bool, message: str)
    """

    # Length validation
    if len(query.strip()) < 5:
        return False, "Question too short. Please provide more details (minimum 5 characters)."

    if len(query) > 500:
        return False, "Question too long. Please keep it under 500 characters."

    # Malicious pattern detection
    malicious_patterns = [
        r'ignore\s+(previous|prior|above|all).*instructions',
        r'disregard.*(?:above|previous|prior)',
        r'system\s*prompt',
        r'<script[^>]*>',
        r'DROP\s+TABLE',
        r'DELETE\s+FROM',
        r'<iframe',
        r'javascript:',
        r'eval\s*\(',
        r'you\s+are\s+now',
        r'new\s+instructions',
    ]

    for pattern in malicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Invalid input detected. Please rephrase your question."

    # Topic relevance check using semantic similarity
    policy_topics = [
        "artificial intelligence regulation and policy",
        "AI ethics and governance frameworks",
        "machine learning safety and compliance",
        "algorithmic accountability and transparency",
        "data privacy and AI systems"
    ]

    query_embedding = topic_model.encode([query])
    topic_embeddings = topic_model.encode(policy_topics)

    similarities = cosine_similarity(query_embedding, topic_embeddings)[0]
    max_similarity = max(similarities)

    if max_similarity < 0.3:
        return False, "This chatbot only answers questions about AI regulations, policies, and governance. Please ask about AI-related topics."

    return True, query

# ============================================================================
# OUTPUT GUARDRAILS
# ============================================================================

def output_guardrail(response, source_docs):
    """
    Validate LLM response before showing to user.
    Returns: (is_valid: bool, message: str, warning: str or None)
    """

    response_lower = response.lower()

    # Check for explicit refusal/uncertainty
    refusal_phrases = [
        "don't have sufficient information",
        "cannot find",
        "not enough information",
        "insufficient information",
        "don't have information",
        "no information available"
    ]

    if any(phrase in response_lower for phrase in refusal_phrases):
        return False, "I don't have enough information in my knowledge base to answer that question accurately. Please try rephrasing or ask about a different aspect of AI policy.", None

    # Length validation
    word_count = len(response.split())
    if word_count < 15:
        return False, "Unable to generate a complete answer. Please try rephrasing your question.", None

    # Check uncertainty markers
    uncertainty_markers = [
        "i'm not sure",
        "i don't know",
        "unclear",
        "might be",
        "possibly",
        "perhaps"
    ]

    warning = None
    if any(marker in response_lower for marker in uncertainty_markers):
        warning = "Note: The answer contains some uncertainty. Please verify with official sources."

    # Check retrieval quality
    if len(source_docs) < 2:
        warning = "Confidence: Medium - Limited relevant sources found. Answer may be incomplete."

    return True, response, warning

# ============================================================================
# COST CONTROLS: RATE LIMITING & CACHING
# ============================================================================

def check_rate_limit():
    """Check if user has exceeded rate limits"""

    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        return False, f"You've reached the maximum of {MAX_QUERIES_PER_SESSION} queries per session. Please refresh the page to continue."

    current_time = time.time()
    time_since_last = current_time - st.session_state.last_query_time

    if time_since_last < MIN_SECONDS_BETWEEN_QUERIES:
        return False, f"Please wait {MIN_SECONDS_BETWEEN_QUERIES} second(s) between queries."

    return True, None

def get_query_hash(query):
    """Generate hash for query caching"""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()

def get_cached_response(query):
    """Get cached response if available"""
    query_hash = get_query_hash(query)
    return st.session_state.query_cache.get(query_hash)

def cache_response(query, result):
    """Cache response for future use"""
    query_hash = get_query_hash(query)
    st.session_state.query_cache[query_hash] = result

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("AIRegulate Chatbot")
st.write("Ask questions about AI regulations, policies, and governance across China, EU, and the US!")

st.sidebar.header("Usage Stats")
st.sidebar.write(f"Queries this session: {st.session_state.query_count}/{MAX_QUERIES_PER_SESSION}")
st.sidebar.write(f"Cached responses: {len(st.session_state.query_cache)}")

st.sidebar.header("About")
st.sidebar.write("""
This chatbot uses:
- **GPT-4o-mini** for answers
- **Retrieval-Augmented Generation (RAG)** to ground responses in verified documents
- **Input/Output Guardrails** for safety
- **Source citations** to prevent hallucinations
""")

query = st.text_area(
    "Ask a question:",
    placeholder="Example: What are the main principles of the EU AI Act?",
    height=100
)

if st.button("Submit", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    else:
        # Input validation
        is_valid, validation_result = input_guardrail(query)

        if not is_valid:
            st.error(validation_result)
        else:
            # Check rate limits
            rate_ok, rate_message = check_rate_limit()
            if not rate_ok:
                st.error(rate_message)
            else:
                # Check cache first
                cached_result = get_cached_response(query)

                if cached_result:
                    st.info("Retrieved from cache (no API call made)")
                    result = cached_result
                else:
                    # API call with error handling
                    try:
                        with st.spinner("Searching knowledge base and generating answer..."):
                            result = qa.invoke(
                                {"query": query},
                                config={"callbacks": [langfuse_handler]}
                            )

                            cache_response(query, result)

                            st.session_state.query_count += 1
                            st.session_state.last_query_time = time.time()

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.error("This could be due to API issues, network problems, or service unavailability. Please try again in a moment.")
                        st.stop()

                # Output validation
                is_valid_output, filtered_response, warning = output_guardrail(
                    result["result"],
                    result.get("source_documents", [])
                )

                if not is_valid_output:
                    st.warning(filtered_response)
                else:
                    if warning:
                        st.warning(warning)

                    st.write("### Answer:")
                    st.write(filtered_response)

                    # Display source citations
                    if result.get("source_documents"):
                        st.write("---")
                        st.write("### Sources:")
                        st.caption("Answer is grounded in the following source documents:")

                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            with st.expander(f"Source {i}"):
                                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    else:
                        st.warning("No highly relevant sources found. Answer quality may be limited.")

st.write("---")
st.caption("Powered by GPT-4o-mini + RAG | Protected by Input/Output Guardrails")
