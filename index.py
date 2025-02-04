from robyn.robyn import Request
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
import supabase
from openai import OpenAI
import requests
import json
import os
import time
import numpy as np
from hashlib import sha256
from typing import List, Optional
from groq import Groq
import uuid
import fitz
from dotenv import load_dotenv
from mixedbread_ai.client import MixedbreadAI
from io import BytesIO
from PyPDF2 import PdfReader
from robyn import Robyn, ALLOW_CORS, WebSocket, Response, Request
from robyn.types import Body
import logging
from database.db_manager import DatabaseManager
from functools import lru_cache
import asyncio

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize database manager
db = DatabaseManager()

class AIClientAdapter:
    def __init__(self, client_mode, ollama_url):
        self.client_mode = client_mode
        self.ollama_url = f"{ollama_url}/api/chat"
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def chat_completions_create(self, model, messages, temperature=0.2, response_format=None):
        # expect llama3.2 as the model name
        local = {
            "llama3.2": "llama3.2",
            "gpt-4o": "llama3.2"
        }
        groq = {
            "llama-3.3": "llama-3.3-70b-versatile",
            "llama-3.2": "llama3-70b-8192"
        }
        if self.client_mode == "LOCAL":
            # Use Ollama client
            data = {
                "messages": messages,
                "model": local[model],
                "stream": False,
            }
            response = requests.post(self.ollama_url, json=data)
            return json.loads(response.text)["message"]["content"]
        elif self.client_mode == "ONLINE":
            # Use OpenAI or Groq client based on the model
            if "gpt" in model:
                return self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format
                ).choices[0].message.content
            else:
                return self.groq_client.chat.completions.create(
                    model=groq[model],
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format
                ).choices[0].message.content


class EmbeddingAdapter:
    def __init__(self, client_mode):
        self.client_mode = client_mode
        
        if self.client_mode == "LOCAL":
            from fastembed import TextEmbedding  # Import fastembed only when running project locally
            self.fastembed_model = TextEmbedding(model_name="BAAI/bge-base-en")
        elif self.client_mode == "ONLINE":
            self.mxbai_client = MixedbreadAI(api_key=os.getenv("MXBAI_API_KEY"))

    def embeddings(self, text):
        if self.client_mode == "LOCAL":
            # Use the fastembed model to generate embeddings
            result = np.array(list(self.fastembed_model.embed([text])))[-1].tolist()
            return result
        elif self.client_mode == "ONLINE":
            # Use the MixedbreadAI client to generate embeddings
            result = self.mxbai_client.embeddings(
                model='mixedbread-ai/mxbai-embed-large-v1',
                input=[text],
                normalized=True,
                encoding_format='float',
                truncation_strategy='end'
            )
            return result.data[0].embedding


client_mode = os.getenv("CLIENT_MODE")
ollama_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
ai_client = AIClientAdapter(client_mode, ollama_url)
embedding_client = EmbeddingAdapter(client_mode)

app = Robyn(__file__)
websocket = WebSocket(app, "/ws")


ALLOW_CORS(app, origins = ["*"])


url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)


def parse_array_string(s):
    # Remove brackets and split in one operation
    return np.fromstring(s[1:-1], sep=',', dtype=float)


@app.exception
def handle_exception(error):
    logger.error(f"Application error: {str(error)}", exc_info=True)
    return Response(status_code=500, description=f"error msg: {error}", headers={})



def get_cache_key(transcript: str) -> str:
    """Generate a deterministic cache key from the transcript"""
    return f"transcript:{sha256(transcript.encode()).hexdigest()}"


@lru_cache
def extract_action_items(transcript):
    # Sample prompt to instruct the model on extracting action items per person
    messages = [
        {
            "role": "user",
            "content": """You are an executive assistant tasked with extracting action items from a meeting transcript.
            For each person involved in the transcript, list their name with their respective action items, or state "No action items"
            if there are none for that person.
            
            Write it as an html list in a json body. For example:
            {"html":"
            <h3>Arsen</h3>
            <ul>
              <li>action 1 bla bla</li>
              <li>action 2 bla</li>
            </ul>
            <h3>Sanskar</h3>
            <ul>
              <li>action 1 bla bla</li>
              <li>action 2 bla</li>
            </ul>"
            }
            
            Transcript: """ + transcript
        }
    ]

    # Sending the prompt to the AI model using chat completions
    response = ai_client.chat_completions_create(
        model="llama-3.3",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    if response is None:
        logger.error("Error extracting action items")
        return "No action items found."


    action_items = json.loads(response)["html"]
    return action_items


@lru_cache
def generate_notes(transcript):
    messages = [
        {
            "role": "user",
            "content": f"""You are an executive assistant tasked with taking notes from an online meeting transcript. You must produce the notes in Markdown format
                Full transcript: {transcript}. Follow the JSON structure:""" + "{notes: meeting notes}" +
                """Here's an example: ### Meeting Notes

                    **Date:** January 15, 2025

                    **Participants:**
                    - You
                    - Sanskar Jethi

                    **Summary:**
                    - Discussion about an option being fully received.
                    - Confirmation that the system is running properly now.
                    - Network issues have been resolved and are working perfectly.

                    **Key Points:**
                    - Option was fully received and confirmed.
                    - System is confirmed to be running properly.
                    - Network is functioning correctly."""
        }
    ]

    response = ai_client.chat_completions_create(
        model="llama-3.3",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    notes = json.loads(response)["notes"]
    return notes


@lru_cache
def generate_title(summary):
    messages = [
        {
            "role": "system",
            "content": f"""You are an executive assistant tasked with generating titles for meetings based on the meeting summaries."""
        },
        {
            "role": "user",
            "content": "Generate a title for the following meeting summary. You must follow the JSON schema: {title: generated title}" + f"Full summary: {summary}"
        }
    ]

    response = ai_client.chat_completions_create(
        model="llama-3.3",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    title = json.loads(response)["title"]
    return title


def send_email_summary(list_emails, actions, meeting_summary = None):
    url = "https://api.resend.com/emails"
    successful_emails = []
    resend_key = os.getenv("RESEND_API_KEY")
    resend_email = os.getenv("RESEND_NOREPLY")

    if not meeting_summary:
        html = f"""
        <h1>Action Items</h1>
        {actions}
        """

    else:
        html = f"""
        <h1>Meeting Summary</h1>
        <p>{meeting_summary}</p>
        <h1>Action Items</h1>
        {actions}
        """

    if list_emails:
        current_time = time.localtime()
        formatted_time = time.strftime("%d %b %Y %I:%M%p", current_time)
        for email in list_emails:
            payload = {
                "from": resend_email,
                "to": email,
                "subject": f"Summary | Meeting on {formatted_time} | Amurex",
                "html": html
            }
            headers = {
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json"
            }

            response = requests.request("POST", url, json=payload, headers=headers)

            if response.status_code != 200:
                return {"type": "error", "error": f"Error sending email to {email}: {response.text}", "emails": None}
            else:
                successful_emails.append(email)

    return {"type": "success", "error": None, "emails": successful_emails}


def send_email(email, email_type, **kwargs):
    url = "https://api.resend.com/emails"
    resend_key = os.getenv("RESEND_API_KEY")
    resend_email = os.getenv("RESEND_FOUNDERS_EMAIL")

    if not email:
        return {"error": "no email provided"}

    if email_type == "signup":
        html = """
                <div>
                    <div>
                        <p><b>Hello there 👋</b></p>
                    </div>
                    <div>
                        <p>First off, a big thank you for signing up for Amurex! We're excited to have you join our mission to create the world's first AI meeting copilot.</p>

                        <p>Amurex is on a mission to become the world's first AI meeting copilot and ultimately your complete executive assistant. We're thrilled to have you join us on this journey.</p>

                        <p>As a quick heads-up, here's what's coming next:</p>
                        <ul>
                            <li>Sneak peeks into new features</li>
                            <li>Early access opportunities</li>
                            <li>Ways to share your feedback and shape the future of Amurex</li>
                        </ul>

                        <p>Want to learn more about how Amurex can help you? <a href="https://cal.com/founders-the-personal-ai-company/15min" >Just Book a Demo →</a></p>

                        <p>If you have any questions or just want to say hi, hit reply – we're all ears! We'd love to talk to you. Or better yet, join our conversation on <a href="https://discord.gg/ftUdQsHWbY">Discord</a>.</p>

                        <p>Thanks for being part of our growing community.</p>

                        <p>Cheers,<br>Sanskar 🦖</p>
                    </div>
                </div>
                """

        subject = "Welcome to Amurex – We're Glad You're Here!"
    
    elif email_type == "meeting_share":
        share_url = kwargs['share_url']
        owner_email = kwargs['owner_email']
        meeting_obj_id = kwargs['meeting_obj_id']

        resend_email = os.getenv("RESEND_NOREPLY")
        subject = f"{owner_email} shared their notes with you | Amurex"
        html = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
            <html dir="ltr" lang="en">
            <div
                style="display:none;overflow:hidden;line-height:1px;opacity:0;max-height:0;max-width:0"
            >
                {owner_email} shared their notes with you
                <div>
                 ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿ ‌​‍‎‏﻿
                </div>
            </div>
            <body
                style='background-color:rgb(255,255,255);margin-top:auto;margin-bottom:auto;margin-left:auto;margin-right:auto;font-family:ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";padding-left:0.5rem;padding-right:0.5rem'
            >
                <table
                align="center"
                width="100%"
                border="0"
                cellpadding="0"
                cellspacing="0"
                role="presentation"
                style="border-width:1px;border-style:solid;border-color:rgb(234,234,234);border-radius:0.25rem;margin-top:40px;margin-bottom:40px;margin-left:auto;margin-right:auto;padding:20px;max-width:465px"
                >
                <tbody>
                    <tr style="width:100%">
                    <td>
                        <table
                        align="center"
                        width="100%"
                        border="0"
                        cellpadding="0"
                        cellspacing="0"
                        role="presentation"
                        style="margin-top:32px"
                        >
                        <tbody>
                            <tr>
                            <td>
                                <div
                                style="text-align:center;margin-top:0px;margin-bottom:0px;margin-left:auto;margin-right:auto;display:block;outline:none;border:none;text-decoration:none"
                                >
                                <a href="https://amurex.ai" style="text-decoration: none; color: inherit;" target="_blank">
                                    <p style="font-size: 50px; display: inline-block; margin-right: 10px;">🦖</p>
                                    <span style="font-size: 50px; display: inline-block;">Amurex</span>
                                </a>
                                </div>
                            </td>
                            </tr>
                        </tbody>
                        </table>
                        <h1
                        style="color:rgb(0,0,0);font-size:24px;font-weight:400;text-align:center;padding:0px;margin-top:30px;margin-bottom:30px;margin-left:0px;margin-right:0px"
                        >
                        <strong>{owner_email}</strong> shared their meeting notes with you
                        </h1>
                        <p
                        style="color:rgb(0,0,0);font-size:14px;line-height:24px;margin:16px 0"
                        >
                        Hey,
                        </p>
                        <p
                        style="color:rgb(0,0,0);font-size:14px;line-height:24px;margin:16px 0"
                        >
                        <a
                            href="mailto:{owner_email}"
                            style="color:rgb(37,99,235);text-decoration-line:none"
                            target="_blank"
                            >{owner_email}</a
                        > has granted you access to their meeting notes.<!-- -->
                        </p>
                        <table
                        align="center"
                        width="100%"
                        border="0"
                        cellpadding="0"
                        cellspacing="0"
                        role="presentation"
                        style="text-align:center;margin-top:32px;margin-bottom:32px"
                        >
                        <tbody>
                            <tr>
                            <td>
                                <a
                                href="{share_url}"
                                style="background-color:rgb(0,0,0);border-radius:0.25rem;color:rgb(255,255,255);font-size:12px;font-weight:600;text-decoration-line:none;text-align:center;padding-left:1.25rem;padding-right:1.25rem;padding-top:0.75rem;padding-bottom:0.75rem;line-height:100%;text-decoration:none;display:inline-block;max-width:100%;mso-padding-alt:0px;padding:12px 20px 12px 20px"
                                target="_blank"
                                ><span
                                    ><!--[if mso
                                    ]><i
                                        style="mso-font-width:500%;mso-text-raise:18"
                                        hidden
                                        >&#8202;&#8202;</i
                                    ><!
                                    [endif]--></span
                                ><span
                                    style="max-width:100%;display:inline-block;line-height:120%;mso-padding-alt:0px;mso-text-raise:9px"
                                    >Open the document</span
                                ><span
                                    ><!--[if mso
                                    ]><i style="mso-font-width:500%" hidden
                                        >&#8202;&#8202;&#8203;</i
                                    ><!
                                    [endif]--></span
                                ></a
                                >
                            </td>
                            </tr>
                        </tbody>
                        </table>
                        <p
                        style="color:rgb(0,0,0);font-size:14px;line-height:24px;margin:16px 0"
                        >
                        or copy and paste this URL into your browser:<!-- -->
                        <a
                            href="{share_url}"
                            style="color:rgb(37,99,235);text-decoration-line:none"
                            target="_blank"
                            >{share_url}</a
                        >
                        </p>
                        <hr
                        style="border-width:1px;border-style:solid;border-color:rgb(234,234,234);margin-top:26px;margin-bottom:26px;margin-left:0px;margin-right:0px;width:100%;border:none;border-top:1px solid #eaeaea"
                        />
                        <p
                        style="color:rgb(102,102,102);font-size:12px;line-height:24px;margin:16px 0"
                        >
                        This invitation was intended for<!-- -->
                        <span style="color:rgb(0,0,0)">{email}</span>. If you
                        were not expecting this invitation, you can ignore this email. If
                        you are concerned about your account&#x27;s safety, please reply
                        to this email to get in touch with us.
                        </p>
                    </td>
                    </tr>
                </tbody>
                </table>
                <!--/$-->
            </body>
            </html>"""

        shared_emails = supabase.table("late_meeting")\
            .select("shared_with")\
            .eq("id", meeting_obj_id)\
            .execute().data[0]["shared_with"]

        if shared_emails:
            if email not in shared_emails:
                result = supabase.table("late_meeting")\
                    .update({"shared_with": shared_emails + [email]})\
                    .eq("id", meeting_obj_id)\
                    .execute()
            else:
                pass
        else:
            result = supabase.table("late_meeting")\
                .update({"shared_with": [email]})\
                .eq("id", meeting_obj_id)\
                .execute()

    payload = {
        "from": resend_email,
        "to": email,
        "subject": subject,
        "html": html
    }

    headers = {
        "Authorization": f"Bearer {resend_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code != 200:
        return {"type": "error", "error": f"Error sending email to {email}: {response.text}"}

    return {"type": "success", "error": None}


def extract_text(file_path):
    with fitz.open(file_path) as pdf_document:
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
    return text


def get_chunks(text):
    max_chars = 200
    overlap = 50
    
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + max_chars]
        chunks.append(chunk)
        start += max_chars - overlap
    
    if start < len(text):
        chunks.append(text[start:])

    return chunks


def embed_text(text):
    embeddings = embedding_client.embeddings(text)
    return embeddings


def calc_centroid(embeddings):
    return np.mean(embeddings, axis=0)


@app.post("/upload_meeting_file/:meeting_id/:user_id/")
async def upload_meeting_file(request):
    meeting_id = request.path_params.get("meeting_id")
    user_id = request.path_params.get("user_id")
    logger.info(f"Processing file upload for meeting_id: {meeting_id}, user_id: {user_id}")

    files = request.files
    file_name = list(files.keys())[0] if len(files) > 0 else None

    if not file_name:
        logger.warning("No file provided in request")
        return Response(status_code=400, description="No file provided", headers={})

    # Check file size limit (20MB)
    file_contents = files[file_name]
    file_limit = 20 * 1024 * 1024
    if len(file_contents) > file_limit:
        logger.warning(f"File size {len(file_contents)} exceeds limit of {file_limit}")
        return Response(status_code=413, description="File size exceeds 20MB limit", headers={})

    logger.info(f"Processing file: {file_name}")
    
    # Generate unique filename
    file_extension = file_name.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    
    # Read file contents
    file_contents = files[file_name]
    
    # Upload to Supabase Storage
    storage_response = supabase.storage.from_("meeting_context_files").upload(
        unique_filename,
        file_contents
    )
    
    # Get public URL for the uploaded file
    file_url = supabase.storage.from_("meeting_context_files").get_public_url(unique_filename)


    new_entry = supabase.table("meetings").upsert(
        {
            "meeting_id": meeting_id,
            "user_id": user_id,
            "context_files": [file_url]
        },
        on_conflict="meeting_id, user_id"
    ).execute()

    pdf_stream = BytesIO(file_contents)
    reader = PdfReader(pdf_stream)

    file_text = ''
    for page in reader.pages:
        file_text += page.extract_text()

    file_chunks = get_chunks(file_text)
    embedded_chunks = [str(embed_text(chunk)) for chunk in file_chunks]

    result = supabase.table("meetings")\
        .update({"embeddings": embedded_chunks, "chunks": file_chunks})\
        .eq("meeting_id", meeting_id)\
        .eq("user_id", user_id)\
        .execute()
    
    return {
        "status": "success",
        "file_url": file_url,
        "updated_meeting": result.data[0]
    }


class TranscriptRequest(Body):
    transcript: str
    meeting_id: str
    user_id: str


class ActionRequest(Body):
    transcript: str


class EndMeetingRequest(Body):
    transcript: str
    user_id: str
    meeting_id: str


class ActionItemsRequest(Body):
    action_items: str
    emails: List[str]
    meeting_summary: Optional[str] = None


def create_memory_object(transcript):
    # Try to get from cache
    logger.info("Cache miss - generating new results")
    
    # Generate new results if not in cache
    action_items = extract_action_items(transcript)
    notes_content = generate_notes(transcript)
    title = generate_title(notes_content)
    
    result = {
        "action_items": action_items,
        "notes_content": notes_content,
        "title": title
    }
    
    return result

@lru_cache(maxsize=1000)
def check_memory_enabled(user_id):
    return supabase.table("users").select("memory_enabled").eq("id", user_id).execute().data[0]["memory_enabled"]

@app.post("/end_meeting")
async def end_meeting(request, body: EndMeetingRequest):
    # the logic here could be simplified as well
    # TODO: simplify the logic
    data = json.loads(body)
    transcript = data["transcript"]
    user_id = data.get("user_id", None)
    meeting_id = data.get("meeting_id", None)


    if not user_id:
        # this is a temporary fix for the issue
        # we need to fix this in the future
        # TODO: figure out why tf are we not sending user_id from the chrome extension
        return {
            "notes_content": generate_notes(transcript),
            "action_items": extract_action_items(transcript)
        }
    
    if not meeting_id:
        action_items = extract_action_items(transcript)
        notes_content = generate_notes(transcript)
        
        return {
            "notes_content": notes_content,
            "action_items": action_items
        }
    
    
    is_memory_enabled = check_memory_enabled(user_id)

    if not is_memory_enabled:
        notes_content = generate_notes(transcript)
        action_items = extract_action_items(transcript)
        return {
            "notes_content": notes_content,
            "action_items": action_items
        }


    meeting_obj = supabase.table("late_meeting").select("id, transcript").eq("meeting_id", meeting_id).execute().data
    if not meeting_obj or len(meeting_obj) == 0 or meeting_obj[0]["transcript"] is None:
        result = supabase.table("late_meeting").upsert({
                "meeting_id": meeting_id,
                "user_ids": [user_id],
                "meeting_start_time": time.time()
            }, on_conflict="meeting_id").execute()

        meeting_obj_id = result.data[0]["id"]
        meeting_obj_transcript_exists = None
    else:
        meeting_obj_id = meeting_obj[0]["id"]
        meeting_obj_transcript_exists = meeting_obj[0]["transcript"]

    if not meeting_obj_transcript_exists:
        # Fire and forget transcript storage
        asyncio.create_task(store_transcript_file(transcript, meeting_obj_id))

    memory = supabase.table("memories").select("*").eq("meeting_id", meeting_obj_id).execute().data

    if memory and memory[0]["content"] and "ACTION_ITEMS" in memory[0]["content"]:
        summary = memory[0]["content"].split("DIVIDER")[0]
        action_items = memory[0]["content"].split("DIVIDER")[1]
        return {
            "action_items": action_items,
            "notes_content": summary
        }
    else:
        memory_obj = create_memory_object(transcript=transcript)
        
        # Fire and forget memory storage
        asyncio.create_task(store_memory_data(memory_obj, user_id, meeting_obj_id))
        
        return {
            "action_items": memory_obj["action_items"],
            "notes_content": memory_obj["notes_content"]
        }
 
@app.post("/generate_actions")
async def generate_actions(request, body: ActionRequest):
    data = json.loads(body)
    transcript = data["transcript"]
    cache_key = get_cache_key(transcript)
    
    logger.info(f"Generating actions for transcript with cache key: {cache_key}")
    
    # Try to get from cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        logger.info("Retrieved result from cache")
        return json.loads(cached_result)
    
    logger.info("Cache miss - generating new results")
    # Generate new results if not in cache
    action_items = extract_action_items(transcript)
    notes_content = generate_notes(transcript)
    
    result = {
        "action_items": action_items,
        "notes_content": notes_content
    }
    
    # Cache the result
    redis_client.setex(
        cache_key,
        CACHE_EXPIRATION,
        json.dumps(result)
    )
    
    return result


@app.post("/submit")
async def submit(request: Request, body: ActionItemsRequest):
    data = json.loads(body)
    action_items = data["action_items"]
    meeting_summary = data["meeting_summary"]
    
    # notion_url = create_note(notes_content)
    emails = data["emails"]
    successful_emails = send_email_summary(emails, action_items, meeting_summary)

    if successful_emails["type"] == "error":
        return {
            "successful_emails": None,
            "error": successful_emails["error"]
        }
    
    return {"successful_emails": successful_emails["emails"]}

class TrackingRequest(Body):
    uuid: str
    event_type: str
    meeting_id: Optional[str] = None

@app.post("/track")
async def track(request: Request, body: TrackingRequest):
    try:
        data = json.loads(body)
        uuid = data["uuid"]
        event_type = data["event_type"]
        meeting_id = data.get("meeting_id")
        result = supabase.table("analytics").insert({
            "uuid": uuid,
            "event_type": event_type,
            "meeting_id": meeting_id
        }).execute()
        return {"result": "success"}
    except Exception as e:
        return Response(
            status_code=500,
            description=f"Error tracking event: {str(e)}",
            headers={}
        )

@app.get("/")
def home():
    return "Welcome to the Amurex backend!"


def find_closest_chunk(query_embedding, chunks_embeddings, chunks):
    query_embedding = np.array(query_embedding)
    chunks_embeddings = np.array(chunks_embeddings)

    similarities = cosine_similarity([query_embedding], chunks_embeddings)

    closest_indices = np.argsort(similarities, axis=1)[0, -5:][::-1] # Five the closest indices of embeddings
    closest_chunks = [chunks[i] for i in closest_indices]

    return closest_chunks


def generate_realtime_suggestion(context, transcript):
    messages = [
        {
            "role": "system",
            "content": """
                You are a personal online meeting assistant, and your task is to give instant help for a user during a call.
                Possible cases when user needs help or a suggestion:
                - They are struggling to answer a question
                - They were asked a question that requires recalling something
                - They need to recall something from their memory (e.g. 'what was the company you told us about 3 weeks ago?')
                
                You have to generate the most important suggestion or help for a user based on the information retrieved from user's memory and latest transcript chunk.
            """
        },
        {
            "role": "user",
            "content": f"""
                Information retrieved from user's memory: {context},
                Latest chunk of the transcript: {transcript},
                

                Be super short. Just give some short facts or key words that could help your user to answer the question.
                Do not use any intro words, like 'Here's the suggestion' etc.
            """
        }
    ]

    response = ai_client.chat_completions_create(
        model="llama-3.2",
        messages=messages,
        temperature=0
    )

    return response


def check_suggestion(request_dict): 
    try:
        transcript = request_dict["transcript"]
        meeting_id = request_dict["meeting_id"]
        user_id = request_dict["user_id"]
        is_file_uploaded = request_dict.get("isFileUploaded", None)

        if is_file_uploaded:
            sb_response = supabase.table("meetings").select("context_files, embeddings, chunks, suggestion_count").eq("meeting_id", meeting_id).eq("user_id", user_id).execute().data

            if not sb_response:
                return {
                    "files_found": False,
                    "generated_suggestion": None,
                    "last_question": None,
                    "type": "no_record_found"
                    }
            
            sb_response = sb_response[0]
            if not sb_response["context_files"] or not sb_response["chunks"]:
                return {
                    "files_found": False,
                    "generated_suggestion": None,
                    "last_question": None,
                    "type": "no_file_found"
                    }

            # logger.info("This is the suggestion count: %s ", sb_response["suggestion_count"])
            # if int(sb_response["suggestion_count"]) == 10:
                # return {
                    # "files_found": True,
                    # "generated_suggestion": None,
                    # "last_question": None,
                    # "type": "exceeded_response"
                # }
            
            file_chunks = sb_response["chunks"]
            embedded_chunks = sb_response["embeddings"]
            embedded_chunks = [parse_array_string(item) for item in embedded_chunks]

            messages_list = [
                {
                    "role": "system",
                    "content": """You are a personal online meeting copilot, and your task is to detect if a speaker needs help during a call. 

                        Possible cases when user needs help in real time:
                        - They need to recall something from their memory (e.g. 'what was the company you told us about 3 weeks ago?')
                        - They need to recall something from files or context they have prepared for the meeting (we are able handle the RAG across their documents)

                        If the user was not asked a question or is not trying to recall something, then they don't need any help or suggestions.
                        
                        You have to identify if they need help based on the call transcript,
                        If your user has already answered the question, there is no need to help.
                        If the last sentence in the transcript was a question, then your user probably needs help. If it's not a question, then don't.
                        
                        You are strictly required to follow this JSON structure:
                        {"needs_help":true/false, "last_question": json null or the last question}
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                        Latest chunk from the transcript: {transcript}.
                    """
                }
            ]

            response = ai_client.chat_completions_create(
                model="llama-3.2",
                messages=messages_list,
                temperature=0,
                response_format={"type": "json_object"}
            )

            response_content = json.loads(response)
            last_question = response_content["last_question"]

            if 'needs_help' in response_content and response_content["needs_help"]:
                embedded_query = embed_text(last_question)
                closest_chunks = find_closest_chunk(query_embedding=embedded_query, chunks_embeddings=embedded_chunks, chunks=file_chunks)

                suggestion = generate_realtime_suggestion(context=closest_chunks, transcript=transcript)

                # result = supabase.table("meetings")\
                #         .update({"suggestion_count": int(sb_response["suggestion_count"]) + 1})\
                #         .eq("meeting_id", meeting_id)\
                #         .eq("user_id", user_id)\
                #         .execute()

                return {
                    "files_found": True,
                    "generated_suggestion": suggestion,
                    "last_question": last_question,
                    "type": "suggestion_response"
                    }
            else:
                return {
                    "files_found": False,
                    "generated_suggestion": None,
                    "last_question": None,
                    "type": "suggestion_response"
                    }
        else:
            # follow up question logic to be implemented
            # print("no uploaded files")
            return {
                "files_found": True,
                "generated_suggestion": None,
                "last_question": None,
                "type": "no_file_uploaded"
                }


    
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred. Please try again later. bitch: {e}"}


async def sync_meeting_with_supabase(meeting_id: str, user_id: str) -> str:
    """Sync meeting data with Supabase and return meeting object ID"""
    try:
        # First check if meeting exists in Supabase
        meeting_obj = supabase.table("late_meeting")\
            .select("id, user_ids")\
            .eq("meeting_id", meeting_id)\
            .execute().data

        if not meeting_obj:
            # Create new meeting in Supabase
            result = supabase.table("late_meeting").upsert({
                "meeting_id": meeting_id,
                "user_ids": [user_id],
                "meeting_start_time": time.time()
            }, on_conflict="meeting_id").execute()
            logger.info(f"Created new meeting in Supabase for meeting_id: {meeting_id}")
            return result.data[0]["id"]
        else:
            # Update existing meeting
            existing_user_ids = meeting_obj[0]["user_ids"] or []
            if user_id not in existing_user_ids:
                new_user_ids = list(set(existing_user_ids + [user_id]))
                result = supabase.table("late_meeting")\
                    .update({"user_ids": new_user_ids})\
                    .eq("meeting_id", meeting_id)\
                    .execute()
                logger.info(f"Added user {user_id} to existing meeting {meeting_id}")
            return meeting_obj[0]["id"]

    except Exception as e:
        logger.error(f"Error in Supabase operation for meeting {meeting_id}: {str(e)}", exc_info=True)
        raise

@websocket.on("connect")
async def on_connect(ws, msg):
    meeting_id = ws.query_params.get("meeting_id")
    user_id = ws.query_params.get("user_id")
    logger.info(f"WebSocket connection request - meeting_id: {meeting_id}, user_id: {user_id}")

    try:
        # Create meeting in SQLite if it doesn't exist
        db.create_meeting(meeting_id)
        
        if user_id and user_id not in ("undefined", "null"):
            # Add connection to SQLite database
            db.add_connection(ws.id, meeting_id, user_id)
            
            # Set as primary user if none exists
            if not db.get_primary_user(meeting_id):
                db.set_primary_user(meeting_id, ws.id)
                logger.info(f"Set primary user for meeting {meeting_id}: {ws.id}")

            # Sync with Supabase
            try:
                meeting_obj_id = await sync_meeting_with_supabase(meeting_id, user_id)
                logger.info(f"Successfully synced meeting {meeting_id} with Supabase")
            except Exception as e:
                logger.error(f"Failed to sync with Supabase: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in connection handling: {str(e)}", exc_info=True)

    return ""


@websocket.on("message")
async def on_message(ws, msg):
    try:
        if isinstance(msg, str):
            msg_data = json.loads(msg)
        else:
            msg_data = msg

        meeting_id = ws.query_params.get("meeting_id")
        data = msg_data.get("data")
        type_ = msg_data.get("type")

        logger.info(f"WebSocket message received - type: {type_}, meeting_id: {meeting_id}")

        if not isinstance(msg_data, dict) or data is None or type_ is None:
            logger.warning("Invalid message format received")
            return ""

        if type_ == "transcript_update":
            try:
                # Check if this is the primary user
                primary_user = db.get_primary_user(meeting_id)
                if primary_user != ws.id or not data:
                    return ""

                # Get current transcript and update
                meeting = db.get_meeting(meeting_id)
                current_transcript = meeting.get('transcript', '') if meeting else ''
                updated_transcript = current_transcript + data
                
                # Update transcript in database
                db.update_transcript(meeting_id, updated_transcript)
                logger.debug(f"Updated transcript for meeting {meeting_id}")

            except Exception as e:
                logger.error(f"Error updating transcript: {str(e)}", exc_info=True)
                return ""

        elif type_ == "check_suggestion":
            data["meeting_id"] = meeting_id
            response = check_suggestion(data)
            return json.dumps(response)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}", exc_info=True)
        return f"JSON parsing error: {e}"
    except Exception as e:
        logger.error(f"WebSocket message error: {str(e)}", exc_info=True)
        return f"WebSocket error: {e}"

    return ""


@websocket.on("close")
async def close(ws, msg):
    try:
        meeting_id = ws.query_params.get("meeting_id")
        
        # Remove connection from database
        db.remove_connection(ws.id)
        logger.info(f"Closed connection for websocket {ws.id}")
        
    except Exception as e:
        logger.error(f"Error in closing connection: {str(e)}", exc_info=True)
    
    return ""


@app.get("/late_summary/:meeting_id")
async def get_late_summary(path_params):
    meeting_id = path_params["meeting_id"]
    if meeting_id == "undefined":
        return {"late_summary": ""}

    meeting = db.get_meeting(meeting_id)
    if not meeting or not meeting['transcript']:
        return {"late_summary": ""}

    late_summary = generate_notes(meeting['transcript'])
    return {"late_summary": late_summary}


@app.get("/check_meeting/:meeting_id")
async def check_meeting(path_params):
    meeting_id = path_params["meeting_id"]
    meeting = db.get_meeting(meeting_id)
    return {"is_meeting": meeting is not None}


@app.post("/send_user_email")
async def send_user_email(request):

    email_type = json.loads(request.body).get("type")
    user_email = json.loads(request.body).get("email")

    if email_type == "signup":
        response = send_email(user_email, email_type)
        return response
    elif email_type == "meeting_share":
        share_url = json.loads(request.body).get("share_url")
        owner_email = json.loads(request.body).get("owner_email")
        meeting_obj_id = json.loads(request.body).get("meeting_id")
        response = send_email(email=user_email, email_type=email_type, share_url=share_url, meeting_obj_id=meeting_obj_id, owner_email=owner_email)
        return response
    else:
        logger.info('oh no')

    return ""


@app.post("/update_meeting_obj")
async def update_meeting_obj(request):
    json_body = json.loads(request.body)
    transcript = json_body.get("transcript")
    meeting_obj_id = json_body.get("meeting_obj_id")
    summary = json_body.get("summary")
    action_items = json_body.get("action_items")

    supabase_update_object = {}

    if not action_items:
        action_items = extract_action_items(transcript)
        supabase_update_object["action_items"] = action_items

    if not summary:
        summary = generate_notes(transcript)
        supabase_update_object["summary"] = summary

    if transcript:
        unique_filename = f"{uuid.uuid4()}.txt"
        file_contents = transcript
        file_bytes = file_contents.encode('utf-8')
        
        storage_response = supabase.storage.from_("transcripts").upload(
            path=unique_filename,
            file=file_bytes,
        )
        file_url = supabase.storage.from_("transcripts").get_public_url(unique_filename)
        
        supabase_update_object["transcript"] = file_url

    result = supabase.table("late_meeting")\
                .update(supabase_update_object)\
                .eq("id", meeting_obj_id)\
                .execute()

    return {"status": "ok"}


@app.get("/get_history")
async def get_history(request):

    return {"status": "ok"}


@app.get("/health_check")
async def health_check():
    return {"status": "ok"}


async def store_transcript_file(transcript: str, meeting_obj_id: str):
    """Store transcript file and update meeting record asynchronously"""
    try:
        unique_filename = f"{uuid.uuid4()}.txt"
        file_bytes = transcript.encode('utf-8')
        
        storage_response = supabase.storage.from_("transcripts").upload(
            path=unique_filename,
            file=file_bytes,
        )
        file_url = supabase.storage.from_("transcripts").get_public_url(unique_filename)
        
        supabase.table("late_meeting")\
            .update({"transcript": file_url})\
            .eq("id", meeting_obj_id)\
            .execute()
    except Exception as e:
        logger.error(f"Failed to store transcript: {str(e)}")

async def store_memory_data(memory_obj: dict, user_id: str, meeting_obj_id: str):
    """Store memory data asynchronously"""
    try:
        content = memory_obj["notes_content"] + memory_obj["action_items"]
        content_chunks = get_chunks(content)
        embeddings = [embed_text(chunk) for chunk in content_chunks]
        centroid = str(calc_centroid(np.array(embeddings)).tolist())
        embeddings = list(map(str, embeddings))
        final_content = memory_obj["notes_content"] + f"\nDIVIDER\n" + memory_obj["action_items"]

        supabase.table("memories").insert({
            "user_id": user_id,
            "meeting_id": meeting_obj_id,
            "content": final_content,
            "chunks": content_chunks,
            "embeddings": embeddings,
            "centroid": centroid,
        }).execute()

        supabase.table("late_meeting")\
            .update({
                "summary": memory_obj["notes_content"], 
                "action_items": memory_obj["action_items"], 
                "meeting_title": memory_obj["title"]
            })\
            .eq("id", meeting_obj_id)\
            .execute()
    except Exception as e:
        logger.error(f"Failed to store memory data: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.start(port=port, host="0.0.0.0")
