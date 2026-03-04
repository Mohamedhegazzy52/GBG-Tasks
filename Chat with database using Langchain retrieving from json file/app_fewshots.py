import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import re
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# ------------------- CONFIG -------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Database :")

# ------------------- DATABASE -------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def get_schema():
    engine = get_db_engine()
    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    schema_string = ""
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = None
            for row in result:
                table_name, column_name = row
                if table_name != current_table:
                    if current_table is not None:
                        schema_string += "\n"
                    schema_string += f"Table: {table_name}\n"
                    current_table = table_name
                schema_string += f"  - {column_name}\n"
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return ""
    return schema_string

# ------------------- LLM & EMBEDDINGS -------------------
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

# ------------------- LOAD FEW SHOTS -------------------
@st.cache_resource
def load_example_selector():
    with open("fewshots.json", "r") as f:
        raw = json.load(f)

    # Reformat for LangChain: keys must match the example_prompt variables
    examples = [
        {
            "naturalQuestion": item["naturalQuestion"],
            "sqlQuery": item["sqlQuery"]
        }
        for item in raw
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=get_embeddings(),
        vectorstore_cls=FAISS,
        k=3,  # number of similar examples to retrieve
        input_keys=["naturalQuestion"]
    )
    return example_selector

# ------------------- HELPERS -------------------
def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()

# ------------------- SQL GENERATION WITH FEW SHOTS -------------------
def generate_sql_query(user_question, schema, conversation_history=None):
    example_selector = load_example_selector()

    # Each example will be formatted like this
    example_prompt = PromptTemplate(
        input_variables=["naturalQuestion", "sqlQuery"],
        template="Question: {naturalQuestion}\nSQL: {sqlQuery}"
    )

    # Prefix: context + schema
    conversation_context = (
        f"\nCONVERSATION CONTEXT: {conversation_history}" if conversation_history else ""
    )

    prefix = f"""You are an expert PostgreSQL data analyst. Given the database schema and similar example questions with their SQL queries, write an accurate PostgreSQL query for the user's question.

DATABASE SCHEMA:
{schema}
{conversation_context}

INSTRUCTIONS:
- Use double quotes around ALL table and column names exactly as shown in the schema
- Return ONLY the SQL query, no explanations or markdown
- Handle aggregations, filters, joins, and ordering as needed

Here are some similar example questions and their SQL queries:"""

    suffix = """Question: {naturalQuestion}
SQL:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["naturalQuestion"]
    )

    chain = few_shot_prompt | get_llm() | StrOutputParser()

    try:
        response = chain.invoke({"naturalQuestion": user_question})
        return clean_sql(response)
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

# ------------------- NATURAL LANGUAGE RESPONSE -------------------
NL_TEMPLATE = """You are a helpful data analyst assistant explaining query results in natural, conversational language.

USER'S QUESTION:
{question}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{data}

INSTRUCTIONS:
1. Answer the user's question directly and conversationally
2. Format numbers clearly, present dates in readable format
3. Be specific - include actual numbers and values from the results
4. If data is empty, say: "No relevant items were found matching your criteria."
5. Add context when helpful (trends, comparisons, percentages with counts)
6. Keep it concise but complete

RESPONSE:"""

def get_natural_language_response(question, data, sql_query=""):
    prompt = PromptTemplate.from_template(NL_TEMPLATE)
    chain = prompt | get_llm() | StrOutputParser()

    try:
        response = chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "data": data,
        })
        return response.strip()
    except Exception as e:
        st.error(f"Error generating natural language response: {e}")
        return "Error generating response."

# ------------------- STREAMLIT APP -------------------
schema = get_schema()
if not schema:
    st.stop()

user_question = st.text_input("Write a question about DB :")

if st.button("Get") and user_question:
    with st.spinner("Generating SQL query..."):
        sql_query = generate_sql_query(user_question, schema)

    st.code(sql_query, language="sql")

    if not sql_query.lower().startswith("select"):
        st.warning("LLM did not generate a SELECT query. Cannot execute.")
        result_df = pd.DataFrame()
    else:
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                result_df = pd.read_sql(sql_query, conn)
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Error executing SQL: {e}")
            result_df = pd.DataFrame()

    if not result_df.empty:
        with st.spinner("Generating answer..."):
            answer = get_natural_language_response(
                question=user_question,
                data=result_df.to_string(),
                sql_query=sql_query,
            )
        st.markdown(f"**Answer:** \n{answer}")