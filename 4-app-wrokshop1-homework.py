import os
import gradio as gr
import fitz  # PyMuPDF
import sqlite3
from datetime import datetime
import uuid
import anthropic


client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

DB_FILE = "pdf_qa_logs.db"

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id TEXT PRIMARY KEY,
                 timestamp TEXT,
                 model_instruction TEXT,
                 prompt TEXT,
                 response TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

model_instruction = "You are an AI assistant. " \
"Given a user query and the content of a PDF document, extract only the relevant information that directly answers the query. Be precise and concise. If the query cannot be answered from the content, " \
"respond with: Information not found in the document."

# Log to SQLite
def log_interaction(system_instruction, prompt, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?)", 
              (interaction_id, timestamp, system_instruction, prompt, response))
    conn.commit()
    conn.close()
# Query the model
def query_model(pdf_upload, query):
    if pdf_upload is None:
        return "Please upload a PDF."
    if not query.strip():
        return "Please enter a valid query."

    try:
        document_content = extract_text_from_pdf(pdf_upload)
        prompt = "Here's the document content" + document_content + "\n\nProvide answer to: " + query
        response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.0,
        system=model_instruction,
        messages=[
        {"role": "user", "content": prompt }
    ]
    ) 
        log_interaction( model_instruction, prompt, response.content[0].text)

        return response.content[0].text
    except Exception as e:
         return f"An error occurred: {str(e)}"

# Gradio app
with gr.Blocks() as app:
    pdf_upload = gr.File(label="Upload PDF", type="binary")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")
    query_button = gr.Button("Submit")
    query_button.click(query_model, inputs=[pdf_upload, query_input], outputs=output)

app.launch()