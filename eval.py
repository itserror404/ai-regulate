"""
Evaluation script for AIRegulate Chatbot
Tests the chatbot's accuracy and quality on AI policy questions
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langfuse.langchain import CallbackHandler
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    print("Error: OPENAI_API_KEY or PINECONE_API_KEY not found in .env file")
    exit(1)

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Initialize Langfuse callback handler for observability
# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from .env automatically
langfuse_handler = CallbackHandler()

print("Loading models...")

# ============================================================================
# MODELS INITIALIZATION
# ============================================================================

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

eval_model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================================================
# PROMPT TEMPLATE
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
# RETRIEVER
# ============================================================================

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7
    }
)

# ============================================================================
# QA CHAIN
# ============================================================================

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

print("Models loaded successfully\n")

# ============================================================================
# TEST DATASET
# ============================================================================

test_cases = [
    {
        "question": "What are the main principles of the EU AI Act?",
        "expected_topics": [
            "risk-based approach",
            "high-risk AI systems",
            "prohibited practices",
            "transparency requirements",
            "fundamental rights"
        ],
        "expected_answer": "The EU AI Act follows a risk-based approach, categorizing AI systems by risk level. It prohibits certain high-risk practices, requires transparency for AI systems, and emphasizes protecting fundamental rights and safety."
    },
    {
        "question": "How does China regulate artificial intelligence?",
        "expected_topics": [
            "state control",
            "algorithm registry",
            "content regulation",
            "social credit",
            "data security"
        ],
        "expected_answer": "China uses a state-controlled approach with algorithm registries, content moderation requirements, integration with social credit systems, and strict data security laws."
    },
    {
        "question": "What is the US approach to AI regulation?",
        "expected_topics": [
            "flexible",
            "sector-specific",
            "voluntary guidelines",
            "federal agencies",
            "innovation-focused"
        ],
        "expected_answer": "The US takes a flexible, sector-specific approach with voluntary guidelines, relying on existing federal agencies and prioritizing innovation while addressing specific risks in different sectors."
    },
    {
        "question": "What are the key differences between EU, US, and China AI regulations?",
        "expected_topics": [
            "risk-based vs flexible",
            "comprehensive vs sector-specific",
            "state control vs market-driven",
            "regulatory approaches",
            "enforcement mechanisms"
        ],
        "expected_answer": "The EU uses comprehensive risk-based regulations, the US applies flexible sector-specific guidelines, and China employs state-controlled top-down regulation. They differ in enforcement, scope, and balance between innovation and control."
    },
    {
        "question": "What is algorithmic accountability and why is it important?",
        "expected_topics": [
            "transparency",
            "explainability",
            "responsibility",
            "decision-making",
            "bias prevention"
        ],
        "expected_answer": "Algorithmic accountability means making AI systems transparent, explainable, and ensuring responsibility for their decisions. It's important for preventing bias, ensuring fairness, and maintaining trust in AI systems."
    },
    {
        "question": "What are the ethical principles for AI policy research?",
        "expected_topics": [
            "human welfare",
            "transparency",
            "inclusivity",
            "accountability",
            "planetary welfare"
        ],
        "expected_answer": "AI policy research should prioritize human and planetary welfare, ensure transparency and accountability, promote inclusivity and diversity, and follow ethical research practices."
    },
    {
        "question": "What is the AI dilemma in regulation?",
        "expected_topics": [
            "innovation vs safety",
            "economic growth vs risk",
            "regulation balance",
            "competitiveness",
            "public interest"
        ],
        "expected_answer": "The AI dilemma involves balancing innovation and economic competitiveness with safety, ethical concerns, and public interest. Regulators must promote AI development while preventing harm and protecting rights."
    },
    {
        "question": "What role does transparency play in AI governance?",
        "expected_topics": [
            "explainability",
            "accountability",
            "trust",
            "auditing",
            "public understanding"
        ],
        "expected_answer": "Transparency in AI governance enables explainability, accountability, and trust. It allows for auditing, public understanding, and ensures AI systems can be scrutinized and held accountable for their decisions."
    },
    {
        "question": "How should AI policy address bias and discrimination?",
        "expected_topics": [
            "fairness",
            "testing",
            "diverse data",
            "accountability",
            "protected characteristics"
        ],
        "expected_answer": "AI policy should require fairness testing, diverse training data, accountability for discriminatory outcomes, and protection of groups based on protected characteristics. Regular audits and bias mitigation are essential."
    },
    {
        "question": "What are high-risk AI systems according to EU regulations?",
        "expected_topics": [
            "critical infrastructure",
            "education",
            "employment",
            "law enforcement",
            "biometric identification"
        ],
        "expected_answer": "High-risk AI systems include those used in critical infrastructure, education and vocational training, employment decisions, law enforcement, and biometric identification. These require strict compliance and oversight."
    }
]

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_keyword_coverage(answer, expected_topics):
    """Calculate what percentage of expected topics are mentioned"""
    answer_lower = answer.lower()
    topics_found = [topic for topic in expected_topics if topic.lower() in answer_lower]
    coverage = len(topics_found) / len(expected_topics) if expected_topics else 0
    return coverage, topics_found

def calculate_semantic_similarity(answer, expected_answer):
    """Calculate semantic similarity between actual and expected answers"""
    answer_embedding = eval_model.encode([answer])
    expected_embedding = eval_model.encode([expected_answer])
    similarity = cosine_similarity(answer_embedding, expected_embedding)[0][0]
    return similarity

def evaluate_answer_quality(result):
    """Evaluate various quality metrics of the answer"""
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    metrics = {
        "word_count": len(answer.split()),
        "has_sources": len(source_docs) > 0,
        "source_count": len(source_docs),
        "is_refusal": any(phrase in answer.lower() for phrase in [
            "don't have sufficient information",
            "cannot find",
            "insufficient information"
        ]),
        "has_uncertainty": any(marker in answer.lower() for marker in [
            "i'm not sure",
            "i don't know",
            "unclear",
            "might be"
        ])
    }

    return metrics

# ============================================================================
# RUN EVALUATION
# ============================================================================

print("=" * 70)
print("STARTING EVALUATION")
print("=" * 70)
print(f"Test cases: {len(test_cases)}")
print(f"Model: GPT-4o-mini")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print()

results = []
total_keyword_coverage = 0
total_semantic_similarity = 0
successful_answers = 0

for i, test in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] Testing: {test['question'][:60]}...")

    try:
        result = qa.invoke(
            {"query": test['question']},
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "test_case": i,
                    "evaluation": True
                }
            }
        )
        answer = result["result"]

        keyword_coverage, topics_found = calculate_keyword_coverage(
            answer,
            test["expected_topics"]
        )
        semantic_sim = calculate_semantic_similarity(
            answer,
            test["expected_answer"]
        )
        quality_metrics = evaluate_answer_quality(result)

        is_successful = (
            not quality_metrics["is_refusal"] and
            quality_metrics["word_count"] >= 15 and
            keyword_coverage > 0.3
        )

        if is_successful:
            successful_answers += 1

        total_keyword_coverage += keyword_coverage
        total_semantic_similarity += semantic_sim

        result_entry = {
            "question": test["question"],
            "answer": answer,
            "expected_topics": test["expected_topics"],
            "topics_found": topics_found,
            "keyword_coverage": round(float(keyword_coverage), 3),
            "semantic_similarity": round(float(semantic_sim), 3),
            "quality_metrics": quality_metrics,
            "is_successful": is_successful
        }
        results.append(result_entry)

        print(f"   Keyword Coverage: {keyword_coverage:.1%}")
        print(f"   Semantic Similarity: {semantic_sim:.1%}")
        print(f"   Sources: {quality_metrics['source_count']}")
        print(f"   Status: {'Success' if is_successful else 'Needs Improvement'}")
        print()

    except Exception as e:
        print(f"   Error: {str(e)}")
        results.append({
            "question": test["question"],
            "answer": None,
            "error": str(e),
            "is_successful": False
        })
        print()

# ============================================================================
# CALCULATE OVERALL METRICS
# ============================================================================

avg_keyword_coverage = total_keyword_coverage / len(test_cases)
avg_semantic_similarity = total_semantic_similarity / len(test_cases)
success_rate = successful_answers / len(test_cases)

overall_metrics = {
    "timestamp": datetime.now().isoformat(),
    "model": "gpt-4o-mini",
    "total_test_cases": len(test_cases),
    "successful_answers": successful_answers,
    "success_rate": round(float(success_rate), 3),
    "avg_keyword_coverage": round(float(avg_keyword_coverage), 3),
    "avg_semantic_similarity": round(float(avg_semantic_similarity), 3),
    "retrieval_threshold": 0.7
}

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print(f"Total Test Cases:        {len(test_cases)}")
print(f"Successful Answers:      {successful_answers}/{len(test_cases)} ({success_rate:.1%})")
print(f"Avg Keyword Coverage:    {avg_keyword_coverage:.1%}")
print(f"Avg Semantic Similarity: {avg_semantic_similarity:.1%}")
print("=" * 70)
print()

refusals = sum(1 for r in results if r.get("quality_metrics", {}).get("is_refusal", False))
with_uncertainty = sum(1 for r in results if r.get("quality_metrics", {}).get("has_uncertainty", False))

print("QUALITY METRICS")
print("-" * 70)
print(f"Answers with sources:    {sum(1 for r in results if r.get('quality_metrics', {}).get('has_sources', False))}/{len(test_cases)}")
print(f"Refusals:                {refusals}/{len(test_cases)}")
print(f"With uncertainty:        {with_uncertainty}/{len(test_cases)}")
print()

if results:
    sorted_by_coverage = sorted(
        [r for r in results if r.get("keyword_coverage") is not None],
        key=lambda x: x["keyword_coverage"],
        reverse=True
    )

    if sorted_by_coverage:
        print("BEST PERFORMING QUESTION")
        print("-" * 70)
        best = sorted_by_coverage[0]
        print(f"Q: {best['question']}")
        print(f"Coverage: {best['keyword_coverage']:.1%} | Similarity: {best['semantic_similarity']:.1%}")
        print()

        print("WORST PERFORMING QUESTION")
        print("-" * 70)
        worst = sorted_by_coverage[-1]
        print(f"Q: {worst['question']}")
        print(f"Coverage: {worst['keyword_coverage']:.1%} | Similarity: {worst['semantic_similarity']:.1%}")
        print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

output = {
    "overall_metrics": overall_metrics,
    "test_results": results
}

output_file = "eval_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {output_file}")
print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
