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
import redis
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
import sentry_sdk
from sentry_sdk import capture_exception
import logging

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
)

load_dotenv()


class AIClientAdapter:
    def __init__(self, client_mode):
        self.client_mode = client_mode
        self.ollama_url = "http://localhost:11434/api/chat"
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
ai_client = AIClientAdapter(client_mode)
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
    capture_exception(error)
    return Response(status_code=500, description=f"error msg: {error}", headers={})


redis_user = os.getenv("REDIS_USERNAME")
redis_host = os.getenv("REDIS_URL")
redis_password = os.getenv("REDIS_PASSWORD")
redis_port = int(os.getenv("REDIS_PORT"))
redis_url = f"rediss://{redis_user}:{redis_password}@{redis_host}:{redis_port}"
redis_client = redis.Redis.from_url(redis_url)
CACHE_EXPIRATION = 60 * 60 * 24  # 24 hours in seconds


def get_cache_key(transcript: str) -> str:
    """Generate a deterministic cache key from the transcript"""
    return f"transcript:{sha256(transcript.encode()).hexdigest()}"


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
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    action_items = json.loads(response)["html"]
    return action_items


def generate_notes(transcript):
    messages = [
        {
            "role": "user",
            "content": f"""You are an executive assistant tasked with taking notes from an online meeting transcript.
                Full transcript: {transcript}"""
        }
    ]

    response = ai_client.chat_completions_create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    notes = response
    return notes


def generate_title(summary):
    messages = [
        {
            "role": "user",
            "content": f"""You are an executive assistant tasked with generating titles for meetings based on the meeting summaries.
                Full summary: {summary}"""
        }
    ]

    response = ai_client.chat_completions_create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    title = response
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
        meeting_obj_id = kwargs['meeting_obj_id']
        html = f"""someone shared their meeting notes with you: {share_url}"""
        resend_email = os.getenv("RESEND_NOREPLY")
        subject = "Someone shared their notes with you | Amurex"

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
                return ""
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


def create_memory_object(user_id, meeting_id, transcript, cache_key):
    # Try to get from cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        logger.info("Retrieved result from cache")
        return json.loads(cached_result)
    
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
    
    # Cache the result
    redis_client.setex(
        cache_key,
        CACHE_EXPIRATION,
        json.dumps(result)
    )
    
    return result


@app.post("/end_meeting")
async def end_meeting(request, body: EndMeetingRequest):
    data = json.loads(body)
    transcript = data["transcript"]
    cache_key = get_cache_key(transcript)

    if not "meeting_id" in data:
        if not "user_id" in data:
            return {
                # this is our problem
                "notes_content": generate_notes(transcript),
                "actions_items": extract_action_items(transcript)
            }
        
        action_items = extract_action_items(transcript)
        notes_content = generate_notes(transcript)
        
        return {
            "notes_content": notes_content,
            "actions_items": action_items
        }
    

    user_id = data["user_id"]
    meeting_id = data["meeting_id"]

    is_memory_enabled = supabase.table("users").select("memory_enabled").eq("id", user_id).execute().data[0]["memory_enabled"]

    if is_memory_enabled is True:
        meeting_obj = supabase.table("late_meeting").select("id, transcript").eq("meeting_id", meeting_id).execute().data
        if not meeting_obj:
            result = supabase.table("late_meeting").upsert({
                    "meeting_id": meeting_id,
                    "user_ids": [user_id],
                    "meeting_start_time": time.time()
                }, on_conflict="meeting_id").execute()

            meeting_obj_transcript_exists = None
            meeting_obj_id = result.data[0]["id"]

        meeting_obj_transcript_exists = meeting_obj[0]["transcript"] # None or str url
        meeting_obj_id = meeting_obj[0]["id"]

        if not meeting_obj_transcript_exists:
            unique_filename = f"{uuid.uuid4()}.txt"
            file_contents = transcript
            file_bytes = file_contents.encode('utf-8')
            
            storage_response = supabase.storage.from_("transcripts").upload(
                path=unique_filename,
                file=file_bytes,
            )
            file_url = supabase.storage.from_("transcripts").get_public_url(unique_filename)
            
            result = supabase.table("late_meeting")\
                    .update({"transcript": file_url})\
                    .eq("id", meeting_obj_id)\
                    .execute()

        memory = supabase.table("memories").select("*").eq("meeting_id", meeting_obj_id).execute().data

        if memory:
            if memory[0]["content"] and "ACTION_ITEMS" in memory[0]["content"]:
                summary = memory[0]["content"].split("DIVIDER")[0]
                action_items = memory[0]["content"].split("DIVIDER")[1]

                result = {
                    "action_items": action_items,
                    "notes_content": notes_content
                }

                return result
        else:
            memory_obj = create_memory_object(user_id=user_id, meeting_id=meeting_id, transcript=transcript, cache_key=cache_key)

            content = memory_obj["notes_content"] + memory_obj["action_items"]
            content_chunks = get_chunks(content)
            embeddings = [embed_text(chunk) for chunk in content_chunks]
            centroid = str(calc_centroid(np.array(embeddings)).tolist())
            embeddings = list(map(str, embeddings))
            content = memory_obj["notes_content"] + f"\nDIVIDER\n" + memory_obj["action_items"]
            title = memory_obj["title"]

            supabase.table("memories").insert({
                    "user_id": user_id,
                    "meeting_id": meeting_obj_id,
                    "content": content,
                    "chunks": content_chunks,
                    "embeddings": embeddings,
                    "centroid": centroid,
                }).execute()

            supabase.table("late_meeting")\
                .update({"summary": memory_obj["notes_content"], "action_items": memory_obj["action_items"], "meeting_title": title})\
                .eq("id", meeting_obj_id)\
                .execute()

            return {
                "action_items": memory_obj["action_items"],
                "notes_content": memory_obj["notes_content"]
            }
    else:
        action_items = extract_action_items(transcript)
        notes_content = generate_notes(transcript)
        
        return {
            "notes_content": notes_content,
            "actions_items": action_items
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
    print(emails, type(emails), data)
    successful_emails = send_email_summary(emails, action_items, meeting_summary)

    if successful_emails["type"] == "error":
        return {
            "successful_emails": None,
            "error": successful_emails["error"]
        }
    
    return {"successful_emails": successful_emails["emails"]}


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

            logger.info("This is the suggestion count: %s ", sb_response["suggestion_count"])
            if int(sb_response["suggestion_count"]) == 10:
                return {
                    "files_found": True,
                    "generated_suggestion": None,
                    "last_question": None,
                    "type": "exceeded_response"
                }
            
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

                result = supabase.table("meetings")\
                        .update({"suggestion_count": int(sb_response["suggestion_count"]) + 1})\
                        .eq("meeting_id", meeting_id)\
                        .eq("user_id", user_id)\
                        .execute()

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


@websocket.on("connect")
async def on_connect(ws, msg):
    meeting_id = ws.query_params.get("meeting_id")
    user_id = ws.query_params.get("user_id")
    logger.info(f"WebSocket connection request - meeting_id: {meeting_id}, user_id: {user_id}")

    try:
        if not redis_client.exists(f"meeting:{meeting_id}"):
            redis_client.set(f"meeting:{meeting_id}", "")
    except Exception as e:
        logger.error(f"Error in setting meeting:{meeting_id} in Redis: {str(e)}", exc_info=True)
        pass

    primary_user_key = f"primary_user:{meeting_id}"
    if user_id is not None and user_id != "undefined" and user_id != "null":
        if not redis_client.exists(primary_user_key):
            logger.info(f"Setting primary user for meeting {meeting_id}")
            try:
                redis_client.set(primary_user_key, ws.id)
            except Exception as e:
                logger.error(f"Error in setting primary_user:{meeting_id}: {str(e)}", exc_info=True)
                pass

            try:
                # Create new meeting metric entry when first user connects
                result = supabase.table("late_meeting").upsert({
                        "meeting_id": meeting_id,
                        "user_ids": [user_id],
                        "meeting_start_time": time.time()
                    }, on_conflict="meeting_id").execute()
            except Exception as e:
                logger.error(f"Error in creating new late_meeting ({meeting_id}) record: {str(e)}", exc_info=True)
                pass
        else:
            logger.info(f"Updating existing late_meeting ({meeting_id}) record to add new user ({user_id})")
            
            # Update existing late_meeting record to add new user
            try:
                user_ids = supabase.table("late_meeting").select("user_ids").eq("meeting_id", meeting_id).execute().data
                if user_ids:
                    user_ids = user_ids[0]["user_ids"]
                else:
                    user_ids = []
                new_user_ids = set(user_ids + [user_id])
                
                result = supabase.table("late_meeting")\
                    .update({"user_ids": list(new_user_ids)}, count="exact")\
                    .eq("meeting_id", meeting_id)\
                    .execute()
            except Exception as e:
                logger.error(f"Error in updating existing late_meeting ({meeting_id}) record: {str(e)}", exc_info=True)
                pass

    return ""


@websocket.on("message")
async def on_message(ws, msg):
    try:
        # First ensure the message is properly parsed as JSON
        if isinstance(msg, str):
            msg_data = json.loads(msg)
        else:
            msg_data = msg

        meeting_id = ws.query_params.get("meeting_id")
        data = msg_data.get("data")
        type_ = msg_data.get("type")

        logger.info(f"WebSocket message received - type: {type_}, meeting_id: {meeting_id}")

        # Safely access the data field
        if not isinstance(msg_data, dict) or data is None or type_ is None:
            logger.warning("Invalid message format received")
            return ""

        if type_ == "transcript_update":
            primary_user_key = f"primary_user:{meeting_id}"
            meeting_key = f"meeting:{meeting_id}"
            
            try:
                # Use pipeline for primary user check and set
                pipe = redis_client.pipeline()
                pipe.exists(primary_user_key)
                pipe.get(primary_user_key)
                exists, primary_user = pipe.execute()

                if not exists:
                    redis_client.set(primary_user_key, ws.id)
                    is_primary = True
                else:
                    is_primary = primary_user.decode() == ws.id

                # Only proceed if this is the primary user
                if not is_primary or data is None:
                    return ""

                # Use pipeline for transcript operations
                pipe = redis_client.pipeline()
                pipe.get(meeting_key)
                pipe.exists(meeting_key)
                current_transcript, exists = pipe.execute()

                # Combine existing and new transcript
                updated_transcript = (current_transcript.decode() if exists else "") + data

                # Set updated transcript with expiration
                redis_client.setex(
                    name=meeting_key,
                    time=CACHE_EXPIRATION,
                    value=updated_transcript
                )

                logger.debug(f"Successfully updated transcript for meeting {meeting_id}")

            except redis.RedisError as e:
                logger.error(f"Redis error during transcript update: {str(e)}", exc_info=True)
                capture_exception(e)
            except Exception as e:
                logger.error(f"Unexpected error during transcript update: {str(e)}", exc_info=True)
                capture_exception(e)

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
        capture_exception(e)  # Send error to Sentry
        return Response(
            status_code=500,
            description=f"Error tracking event: {str(e)}",
            headers={}
        )

@websocket.on("close")
async def close(ws, msg):
    meeting_id = ws.query_params.get("meeting_id")
    primary_user_key = f"primary_user:{meeting_id}"
    if redis_client.get(primary_user_key).decode() == ws.id:
        redis_client.delete(primary_user_key)

    return ""


@app.get("/late_summary/:meeting_id")
async def get_late_summary(path_params):
    meeting_id = path_params["meeting_id"]
    if meeting_id == "undefined":
        return {"late_summary": ""}

    transcript = redis_client.get(f"meeting:{meeting_id}")
    late_summary = generate_notes(transcript)

    return {"late_summary": late_summary}


@app.get("/check_meeting/:meeting_id")
async def check_meeting(path_params):
    meeting_id = path_params["meeting_id"]
    is_meeting = redis_client.exists(f"meeting:{meeting_id}")

    return {"is_meeting": is_meeting}
    

@app.post("/send_user_email")
async def send_user_email(request):

    email_type = json.loads(request.body).get("type")
    user_email = json.loads(request.body).get("email")

    if email_type == "signup":
        send_email(user_email, email_type)
    elif email_type == "meeting_share":
        share_url = json.loads(request.body).get("share_url")
        meeting_obj_id = json.loads(request.body).get("meeting_id")
        send_email(email=user_email, email_type=email_type, share_url=share_url, meeting_obj_id=meeting_obj_id)
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


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.start(port=port, host="0.0.0.0")
