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
from io import BytesIO
from PyPDF2 import PdfReader
from robyn import Robyn, ALLOW_CORS, WebSocket, Response, Request
from robyn.types import Body
import logging
from database.db_manager import DatabaseManager
from functools import lru_cache
import asyncio
import redis
from mistralai import Mistral
import re
import multiprocessing


redis_user = os.getenv("REDIS_USERNAME")
redis_host = os.getenv("REDIS_URL")
redis_password = os.getenv("REDIS_PASSWORD")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_url = f"rediss://{redis_user}:{redis_password}@{redis_host}:{redis_port}"
redis_client = redis.Redis.from_url(
    redis_url,
    health_check_interval=10,
    socket_connect_timeout=5,
    socket_keepalive=True,
    retry_on_timeout=True,
    max_connections=250  # this is the max number of connections to the redis server
)
CACHE_EXPIRATION = 60 * 60 * 24  # 24 hours in seconds


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
            # Initialize Mistral client instead of MixedbreadAI
            self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def embeddings(self, text):
        if self.client_mode == "LOCAL":
            # Use the fastembed model to generate embeddings
            result = np.array(list(self.fastembed_model.embed([text])))[-1].tolist()
            return result
        elif self.client_mode == "ONLINE":
            # Use the Mistral client to generate embeddings
            model = "mistral-embed"
            response = self.mistral_client.embeddings.create(
                model=model,
                inputs=[text]
            )

            return response.data[0].embedding


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


def markdown_to_html(markdown: str) -> str:
    # yes, I've built our own markdown to html converter.

    """
    Convert a Markdown string to an HTML string, without using third-party libraries.
    Supports a large subset of core Markdown features:
      - Headings
      - Bold, italic
      - Inline code
      - Fenced code blocks
      - Links, images
      - Blockquotes
      - Unordered and ordered lists
      - Horizontal rules
      - Paragraphs
    """
    # Split input into lines
    lines = markdown.split('\n')
    
    # State variables
    in_code_block = False
    code_block_delimiter = None
    code_lines = []
    paragraph_lines = []
    html_output = []
    
    def flush_paragraph():
        """Close out the current paragraph buffer and convert it into an HTML <p> block."""
        if paragraph_lines:
            # Join all paragraph lines with a space, then run inline parsing.
            paragraph_text = ' '.join(paragraph_lines)
            paragraph_text = parse_inline(paragraph_text)
            html_output.append(f"<p>{paragraph_text}</p>")
            paragraph_lines.clear()
    
    def parse_inline(text: str) -> str:
        """
        Perform inline replacements for:
         - Images: ![alt](url)
         - Links: [text](url)
         - Bold: **text** or __text__
         - Italic: *text* or _text_
         - Inline code: `code`
        """
        # Images
        text = re.sub(
            r'!\[([^\]]*)\]\(([^)]+)\)',
            r'<img alt="\1" src="\2">',
            text
        )
        # Links
        text = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2">\1</a>',
            text
        )
        # Bold (greedy match for ** or __)
        text = re.sub(
            r'(\*\*|__)(.+?)\1',
            r'<strong>\2</strong>',
            text
        )
        # Italic (greedy match for * or _)
        text = re.sub(
            r'(\*|_)(.+?)\1',
            r'<em>\2</em>',
            text
        )
        # Inline code
        text = re.sub(
            r'`([^`]+)`',
            r'<code>\1</code>',
            text
        )
        return text
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for fenced code block (start or end)
        fence_match = re.match(r'^(```|~~~)\s*$', line)
        if fence_match:
            fence = fence_match.group(1)
            if not in_code_block:
                # Entering code block
                flush_paragraph()
                in_code_block = True
                code_block_delimiter = fence
                code_lines = []
            else:
                # Exiting code block
                if fence == code_block_delimiter:
                    in_code_block = False
                    # Escape HTML special chars inside code
                    escaped_code = '\n'.join(code_lines)
                    escaped_code = (
                        escaped_code
                        .replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                    )
                    html_output.append(f"<pre><code>{escaped_code}</code></pre>")
            i += 1
            continue
        
        # If we're currently in a fenced code block, gather lines until fence end
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Check for headings: (#{1,6} + text)
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = parse_inline(heading_match.group(2))
            html_output.append(f"<h{level}>{heading_text}</h{level}>")
            i += 1
            continue
        
        # Check for horizontal rule
        hr_match = re.match(r'^(\*[\s\*]*|\-[\s\-]*|_[\s_]*)$', line.strip())
        if hr_match:
            # Heuristically requires 3 or more symbols to be valid
            # We'll do a simpler check by removing spaces and verifying length >= 3
            chars_only = re.sub(r'\s+', '', line)
            if len(chars_only) >= 3:
                flush_paragraph()
                html_output.append("<hr/>")
                i += 1
                continue
        
        # Check for blockquote
        bq_match = re.match(r'^>\s?(.*)', line)
        if bq_match:
            flush_paragraph()
            # Accumulate blockquote lines
            quote_lines = [bq_match.group(1)]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_bq = re.match(r'^>\s?(.*)', next_line)
                if next_bq:
                    quote_lines.append(next_bq.group(1))
                    j += 1
                else:
                    break
            # Recursively parse the blockquote content
            inner_html = markdown_to_html('\n'.join(quote_lines))
            html_output.append(f"<blockquote>{inner_html}</blockquote>")
            i = j
            continue
        
        # Check for lists (unordered or ordered)
        # Unordered: -, +, or * at start
        # Ordered: number followed by a period
        ul_match = re.match(r'^(\*|\-|\+)\s+(.*)', line)
        ol_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if ul_match or ol_match:
            flush_paragraph()
            # Determine list type
            if ul_match:
                list_tag = "ul"
            else:
                list_tag = "ol"
            list_buffer = []
            
            # Gather subsequent lines
            while i < len(lines):
                l = lines[i]
                # Check if we still match the same type of list
                if list_tag == "ul":
                    m = re.match(r'^(\*|\-|\+)\s+(.*)', l)
                else:
                    m = re.match(r'^(\d+)\.\s+(.*)', l)
                
                if m:
                    item_content = m.group(2)
                    list_buffer.append(item_content)
                    i += 1
                else:
                    break
            
            # Convert the gathered lines into list items
            html_output.append(f"<{list_tag}>")
            for item in list_buffer:
                html_output.append(f"  <li>{parse_inline(item)}</li>")
            html_output.append(f"</{list_tag}>")
            continue
        
        # If the line is empty, it signals a paragraph break
        if not line.strip():
            flush_paragraph()
            i += 1
            continue
        
        # Otherwise, treat it as part of a paragraph
        paragraph_lines.append(line.strip())
        i += 1
    
    # Flush any remaining paragraph at the end
    flush_paragraph()
    
    # Join everything into one HTML string
    return "\n".join(html_output)


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
        # model="gpt-4o",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    if response is None:
        logger.error("Error extracting action items")
        return "No action items found."


    action_items = json.loads(response)["html"]
    return action_items


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
                    - Network is functioning correctly.""" + 
                    "Here's an example of what your JSON output should look like: " +
                    """{
                        "notes": "### Meeting Notes\n\n**Date:** February 19, 2025\n\n**Participants:**\n- You\n- Sanskar Jethi\n\n**Summary:**\n- Discussion about an option being fully received.\n- Confirmation that the system is running properly now.\n- Network issues have been resolved and are working perfectly.\n\n**Key Points:**\n- Option was fully received and confirmed.\n- System is confirmed to be running properly.\n- Network is functioning correctly."
                    }"""
        }
    ]

    try:
        response = ai_client.chat_completions_create(
            model="llama-3.3",
            # model="gpt-4o",
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        if "failed_generation" in str(e):
            # extract the failed_generation from the error message anyways
            response = e["failed_generation"]
        else:
            return "No notes found."

    notes = json.loads(response)["notes"]
    return notes


def generate_title(summary):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an executive assistant tasked with generating concise meeting titles. "
                "Use the participants' names and the meeting date from the summary when available. "
                "Keep the title relevant and limited to 10 words."
            )
        },
        {
            "role": "user",
            "content": (
                'Generate a title for the following meeting summary. '
                'Return the response in JSON format following this schema: {"title": "<generated title>"}. '
                f'Full summary: {summary}'
            )
        }
    ]

    response = ai_client.chat_completions_create(
        model="llama-3.3",
        # model="gpt-4o",
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
                                    <a
                                        href="https://app.amurex.ai"
                                        style="text-decoration: none; color: inherit"
                                        target="_blank"
                                    >
                                        <p
                                        style="
                                            font-size: 40px;
                                            display: inline-block;
                                            margin: 0 5px 0 0;
                                        "
                                        >
                                            <img 
                                                src="https://www.amurex.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FAmurexLogo.56901b87.png&w=64&q=75"
                                                alt="Amurex Logo"
                                                style="
                                                width: 40px;
                                                height: 40px;
                                                vertical-align: middle;
                                                border-radius: 50%;
                                                "
                                            />
                                        </p>
                                        <span style="font-size: 40px; display: inline-block">
                                            Amurex
                                        </span>
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


    elif email_type == "post_meeting_summary":
        meeting_id = kwargs['meeting_id']
        result = supabase.table("late_meeting")\
            .select("summary, action_items")\
            .eq("id", meeting_id)\
            .execute().data[0]
        
        summary = result["summary"]
        action_items = result["action_items"]

        resend_email = os.getenv("RESEND_NOREPLY")

        subject = f"Your notes are ready | Amurex"
        html = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
                    <html dir="ltr" lang="en">
                        <div
                            style="
                            display: none;
                            overflow: hidden;
                            line-height: 1px;
                            opacity: 0;
                            max-height: 0;
                            max-width: 0;
                            "
                        >
                            Your notes are ready | Amurex
                        </div>
                        
                        <body
                            style="
                            background-color: rgb(255, 255, 255);
                            margin-top: auto;
                            margin-bottom: auto;
                            margin-left: auto;
                            margin-right: auto;
                            font-family: ui-sans-serif, system-ui, sans-serif, 'Apple Color Emoji',
                                'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
                            padding-left: 0.5rem;
                            padding-right: 0.5rem;
                            "
                        >
                            <table
                            align="center"
                            width="100%"
                            border="0"
                            cellpadding="0"
                            cellspacing="0"
                            role="presentation"
                            style="
                                border-width: 1px;
                                border-style: solid;
                                border-color: rgb(234, 234, 234);
                                border-radius: 0.25rem;
                                margin-top: 40px;
                                margin-bottom: 40px;
                                margin-left: auto;
                                margin-right: auto;
                                padding: 20px;
                                max-width: 465px;
                            "
                            >
                                <tbody>
                                    <tr style="width: 100%">
                                        <td>
                                            <table
                                            align="center"
                                            width="100%"
                                            border="0"
                                            cellpadding="0"
                                            cellspacing="0"
                                            role="presentation"
                                            style="margin-top: 32px"
                                            >
                                                <tbody>
                                                    <tr>
                                                        <td>
                                                            <div style="
                                                                text-align: center;
                                                                margin-top: 0px;
                                                                margin-bottom: 0px;
                                                                margin-left: auto;
                                                                margin-right: auto;
                                                                display: block;
                                                                outline: none;
                                                                border: none;
                                                                text-decoration: none;
                                                                "
                                                            >
                                                                <a
                                                                    href="https://app.amurex.ai"
                                                                    style="text-decoration: none; color: inherit"
                                                                    target="_blank"
                                                                >
                                                                    <p
                                                                    style="
                                                                        font-size: 40px;
                                                                        display: inline-block;
                                                                        margin: 0 5px 0 0;
                                                                    "
                                                                    >
                                                                        <img 
                                                                            src="https://www.amurex.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2FAmurexLogo.56901b87.png&w=64&q=75"
                                                                            alt="Amurex Logo"
                                                                            style="
                                                                            width: 40px;
                                                                            height: 40px;
                                                                            vertical-align: middle;
                                                                            border-radius: 50%;
                                                                            "
                                                                        />
                                                                    </p>
                                                                    <span style="font-size: 40px; display: inline-block">
                                                                        Amurex
                                                                    </span>
                                                                </a>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                            
                                            <p
                                            style="
                                                color: rgb(0, 0, 0);
                                                font-size: 14px;
                                                line-height: 24px;
                                                margin: 16px 0;
                                                "
                                            >
                                                Hey 👋
                                            </p>
                                            
                                            <p
                                            style="
                                                color: rgb(0, 0, 0);
                                                font-size: 14px;
                                                line-height: 24px;
                                                margin: 16px 0;
                                            "
                                            >
                                                Here's a quick recap of your meeting:
                                            </p>

                                            <p
                                            style="
                                                color: rgb(0, 0, 0);
                                                font-size: 16px;
                                                font-weight: 600;
                                                line-height: 24px;
                                                margin: 24px 0 8px 0;
                                                "
                                            >
                                                Summary
                                            </p>

                                            <div style="
                                                max-height: 130px;
                                                overflow: hidden;
                                                position: relative;
                                                margin-bottom: 20px;
                                                padding: 16px;
                                                border: 1px solid #eaeaea;
                                                border-radius: 8px;
                                                box-shadow: 0 4px 40px 10px rgba(147, 51, 234, 0.8), 0 0 20px 5px rgba(255, 0, 255, 0.6);
                                            ">
                                                {markdown_to_html(summary)}
                                                <div style="
                                                    position: absolute;
                                                    bottom: 0;
                                                    left: 0;
                                                    width: 100%;
                                                    height: 80px;
                                                    background: linear-gradient(rgba(255,255,255,0), rgba(255,255,255,0.8) 40%, rgba(255,255,255,1) 90%);
                                                    pointer-events: none;
                                                    border-bottom-left-radius: 8px;
                                                    border-bottom-right-radius: 8px;
                                                "></div>
                                            </div>

                                            <p
                                            style="
                                                color: rgb(0, 0, 0);
                                                font-size: 16px;
                                                font-weight: 600;
                                                line-height: 24px;
                                                margin: 24px 0 8px 0;
                                                "
                                            >
                                                Action Items
                                            </p>
                                            
                                            <div style="
                                                max-height: 130px;
                                                overflow: hidden;
                                                position: relative;
                                                margin-bottom: 20px;
                                                padding: 16px;
                                                border: 1px solid #eaeaea;
                                                border-radius: 8px;
                                            ">
                                                {action_items}
                                                <div style="
                                                    position: absolute;
                                                    bottom: 0;
                                                    left: 0;
                                                    width: 100%;
                                                    height: 80px;
                                                    background: linear-gradient(rgba(255,255,255,0), rgba(255,255,255,1) 90%;
                                                    pointer-events: none;
                                                "></div>
                                            </div>

                                            <table
                                                align="center"
                                                width="100%"
                                                border="0"
                                                cellpadding="0"
                                                cellspacing="0"
                                                role="presentation"
                                                style="text-align: center; margin-top: 32px; margin-bottom: 32px"
                                            >
                                                <tbody>
                                                    <tr>
                                                        <td>
                                                            <p
                                                            style="
                                                                color: rgb(0, 0, 0);
                                                                font-size: 14px;
                                                                line-height: 24px;
                                                                margin: 16px 0;
                                                            "
                                                            >
                                                                For the full summary, access it in our web app:
                                                            </p>
                                                            
                                                            <a
                                                            href="https://app.amurex.ai/meetings/{meeting_id}"
                                                            style="
                                                                background-color: rgb(0, 0, 0);
                                                                border-radius: 0.25rem;
                                                                color: rgb(255, 255, 255);
                                                                font-size: 12px;
                                                                font-weight: 600;
                                                                text-decoration-line: none;
                                                                text-align: center;
                                                                padding-left: 1.25rem;
                                                                padding-right: 1.25rem;
                                                                padding-top: 0.75rem;
                                                                padding-bottom: 0.75rem;
                                                                line-height: 100%;
                                                                text-decoration: none;
                                                                display: inline-block;
                                                                max-width: 100%;
                                                                mso-padding-alt: 0px;
                                                                padding: 12px 20px 12px 20px;
                                                            "
                                                            target="_blank"
                                                            >
                                                                <span
                                                                    style="
                                                                    max-width: 100%;
                                                                    display: inline-block;
                                                                    line-height: 120%;
                                                                    mso-padding-alt: 0px;
                                                                    mso-text-raise: 9px;
                                                                    "
                                                                >
                                                                    
                                                                    View full summary
                                                                </span>
                                                            </a>
                                                        </td>
                                                    </tr>
                                                </tbody>
                                            </table>

                                            <hr
                                            style="
                                                border-width: 1px;
                                                border-style: solid;
                                                border-color: rgb(234, 234, 234);
                                                margin-top: 26px;
                                                margin-bottom: 26px;
                                                margin-left: 0px;
                                                margin-right: 0px;
                                                width: 100%;
                                                border: none;
                                                border-top: 1px solid #eaeaea;
                                                "
                                            />

                                            <p
                                            style="
                                                color: rgb(102, 102, 102);
                                                font-size: 12px;
                                                line-height: 24px;
                                                margin: 16px 0;
                                                "
                                            >
                                                This invitation was intended for<!-- --> <span style="color: rgb(0, 0, 0)">{email}</span>. 
                                                If you were not expecting this invitation, you can ignore this email. If you are
                                                concerned about your account&#x27;s safety, please get in touch with <a href="mailto:founders@thepersonalaicompany.com">founders@thepersonalaicompany.com</a>.
                                            </p>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                            <!--/$-->
                        </body>
                    </html>"""


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
    
    # Ensure notes_content is a string before generating title
    if isinstance(notes_content, list):
        notes_content = '\n'.join(notes_content)
    
    title = generate_title(notes_content)
    
    result = {
        "action_items": action_items,
        "notes_content": notes_content,
        "title": title
    }
    
    return result

@lru_cache(maxsize=1000)
def check_memory_enabled(user_id):
    try:
        result = supabase.table("users").select("memory_enabled").eq("id", user_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0].get("memory_enabled", False)
        logger.warning(f"No user found with id {user_id}")
        return False
    except Exception as e:
        logger.error(f"Error checking memory enabled for user {user_id}: {str(e)}")
        return False

@app.post("/end_meeting")
async def end_meeting(request, body: EndMeetingRequest):
    try:
        data = json.loads(body)
        transcript = data["transcript"]
        user_id = data.get("user_id")
        meeting_id = data.get("meeting_id")

        # If no user_id or meeting_id, return basic response
        if not user_id or user_id == "undefined" or user_id == "null":
            logger.info("No valid user_id provided, returning basic response")
            return {
                "notes_content": generate_notes(transcript),
                "action_items": extract_action_items(transcript)
            }
        
        if not meeting_id:
            logger.info("No meeting_id provided, returning basic response")
            action_items = extract_action_items(transcript)
            notes_content = generate_notes(transcript)
            return {
                "notes_content": notes_content,
                "action_items": action_items
            }
        
        # Check if memory is enabled for this user
        try:
            is_memory_enabled = check_memory_enabled(user_id)
            logger.info(f"Memory enabled status for user {user_id}: {is_memory_enabled}")
        except Exception as e:
            logger.error(f"Error checking memory enabled status: {str(e)}")
            is_memory_enabled = False

        if not is_memory_enabled:
            logger.info(f"Memory not enabled for user {user_id}, returning basic response")
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

        if memory and memory[0]["content"] and "DIVIDER" in memory[0]["content"]:
            summary = memory[0]["content"].split("DIVIDER")[0]
            action_items = memory[0]["content"].split("DIVIDER")[1]
            return {
                "action_items": action_items,
                "notes_content": summary
            }
        else:
            memory_obj = create_memory_object(transcript=transcript)
            
            response = {
                "action_items": memory_obj["action_items"],
                "notes_content": memory_obj["notes_content"]
            }

            # Create async task for storing memory data
            asyncio.create_task(store_memory_data(memory_obj, user_id, meeting_obj_id))

            return response

    except Exception as e:
        logger.error(f"Error in end_meeting: {str(e)}")
        return Response(
            status_code=500,
            description=f"Internal server error: {str(e)}",
            headers={}
        )

async def store_memory_data(memory_obj: dict, user_id: str, meeting_obj_id: str, max_retries: int = 1):
    """Store memory data asynchronously using asyncio with retries"""
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    while retry_count < max_retries:
        try:
            # Step 1: Process content and create chunks
            try:
                content = memory_obj["notes_content"] + memory_obj["action_items"]
                content_chunks = get_chunks(content)
            except Exception as content_error:
                logger.error(f"Failed to process content: {str(content_error)}")
                raise content_error

            # Step 2: Create embeddings with retry for each chunk
            try:
                async def create_embedding_with_retry(chunk, max_chunk_retries=2):
                    chunk_retry = 0
                    while chunk_retry < max_chunk_retries:
                        try:
                            return await embed_text(chunk)
                        except Exception as e:
                            chunk_retry += 1
                            if chunk_retry == max_chunk_retries:
                                raise e
                            await asyncio.sleep(1)
                
                embeddings = await asyncio.gather(
                    *[create_embedding_with_retry(chunk) for chunk in content_chunks],
                    return_exceptions=True
                )

                # Check if any embeddings failed
                failed_embeddings = [i for i, e in enumerate(embeddings) if isinstance(e, Exception)]
                if failed_embeddings:
                    raise Exception(f"Failed to create embeddings for chunks: {failed_embeddings}")

            except Exception as embedding_error:
                logger.error(f"Failed to create embeddings: {str(embedding_error)}")
                raise embedding_error

            # Step 3: Calculate centroid
            try:
                centroid = str(calc_centroid(np.array(embeddings)).tolist())
                embeddings = list(map(str, embeddings))
                final_content = memory_obj["notes_content"] + f"\nDIVIDER\n" + memory_obj["action_items"]
            except Exception as processing_error:
                logger.error(f"Failed to process embeddings: {str(processing_error)}")
                raise processing_error

            # Step 4: Database operations with individual error handling
            try:
                # Execute database operations concurrently
                db_results = await asyncio.gather(
                    supabase.table("memories").insert({
                        "user_id": user_id,
                        "meeting_id": meeting_obj_id,
                        "content": final_content,
                        "chunks": content_chunks,
                        "embeddings": embeddings,
                        "centroid": centroid,
                    }).execute(),
                    supabase.table("late_meeting").update({
                        "summary": memory_obj["notes_content"], 
                        "action_items": memory_obj["action_items"], 
                        "meeting_title": memory_obj["title"]
                    }).eq("id", meeting_obj_id).execute(),
                    return_exceptions=True
                )

                # Check for database operation failures
                for i, result in enumerate(db_results):
                    if isinstance(result, Exception):
                        raise Exception(f"Database operation {i} failed: {str(result)}")

            except Exception as db_error:
                logger.error(f"Database operations failed: {str(db_error)}")
                raise db_error

            # Step 5: Email handling with retries
            try:
                # Get user email and preferences
                user_data = await supabase.table("users").select("email,emails_enabled").eq("id", user_id).execute()
                if not user_data.data:
                    raise Exception(f"No user found with id {user_id}")
                
                user_email = user_data.data[0]["email"]
                emails_enabled = user_data.data[0]["emails_enabled"]
                
                email_data = await supabase.table("late_meeting").select("post_email_sent").eq("id", meeting_obj_id).execute()
                if not email_data.data:
                    raise Exception(f"No meeting found with id {meeting_obj_id}")
                
                email_already_sent = email_data.data[0]["post_email_sent"]

                if not email_already_sent and emails_enabled:
                    email_results = await asyncio.gather(
                        send_email(email=user_email, email_type="post_meeting_summary", meeting_id=meeting_obj_id),
                        supabase.table("late_meeting").update({
                            "post_email_sent": True
                        }).eq("id", meeting_obj_id).execute(),
                        return_exceptions=True
                    )

                    # Check for email operation failures
                    for i, result in enumerate(email_results):
                        if isinstance(result, Exception):
                            raise Exception(f"Email operation {i} failed: {str(result)}")

            except Exception as email_error:
                logger.error(f"Email operations failed: {str(email_error)}")
                # Don't retry the whole operation for email failures
                # Just log and continue
                logger.warning("Continuing despite email failure")

            # If we get here, everything succeeded
            logger.info(f"Successfully stored memory data for meeting {meeting_obj_id}")
            return True

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"Retrying memory storage in {wait_time} seconds... (attempt {retry_count + 1})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to store memory data after {max_retries} attempts: {str(e)}")
                # Update the meeting record to indicate failure
                try:
                    await supabase.table("late_meeting").update({
                        "memory_storage_failed": True,
                        "memory_storage_error": str(e)
                    }).eq("id", meeting_obj_id).execute()
                except Exception as notify_error:
                    logger.error(f"Failed to notify about memory storage failure: {str(notify_error)}")
                return False

    return False  # Should not reach here, but just in case

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
        # model="llama-3.2",
        model="gpt-4o",
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
                # model="llama-3.2",
                model="gpt-4o",
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
            is_file_uploaded = data.get("isFileUploaded", None)
            if is_file_uploaded is True:
                response = check_suggestion(data)
                return json.dumps(response)
            else:
                return json.dumps({"files_found": False, "generated_suggestion": None, "last_question": None, "type": "no_file_uploaded"})

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

    print("This is the late meeting transcript: ", meeting_id,  meeting['transcript'])
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


async def store_transcript_file(transcript: str, meeting_obj_id: str, max_retries: int = 1):
    """Store transcript file and update meeting record asynchronously with retries"""
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    while retry_count < max_retries:
        try:
            unique_filename = f"{uuid.uuid4()}.txt"
            file_bytes = transcript.encode('utf-8')
            
            # First try to upload to storage
            try:
                storage_response = supabase.storage.from_("transcripts").upload(
                    path=unique_filename,
                    file=file_bytes,
                )
            except Exception as storage_error:
                logger.error(f"Failed to upload transcript to storage (attempt {retry_count + 1}): {str(storage_error)}")
                raise storage_error

            # If storage upload succeeds, get the URL
            try:
                file_url = supabase.storage.from_("transcripts").get_public_url(unique_filename)
            except Exception as url_error:
                logger.error(f"Failed to get public URL (attempt {retry_count + 1}): {str(url_error)}")
                # Try to cleanup the uploaded file
                try:
                    supabase.storage.from_("transcripts").remove([unique_filename])
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup uploaded file after URL error: {str(cleanup_error)}")
                raise url_error

            # If we got the URL, update the database
            try:
                await supabase.table("late_meeting")\
                    .update({"transcript": file_url})\
                    .eq("id", meeting_obj_id)\
                    .execute()
                
                logger.info(f"Successfully stored transcript for meeting {meeting_obj_id}")
                return True  # Success
                
            except Exception as db_error:
                logger.error(f"Failed to update database (attempt {retry_count + 1}): {str(db_error)}")
                # Try to cleanup the uploaded file
                try:
                    supabase.storage.from_("transcripts").remove([unique_filename])
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup uploaded file after database error: {str(cleanup_error)}")
                raise db_error

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"Retrying transcript storage in {wait_time} seconds... (attempt {retry_count + 1})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to store transcript after {max_retries} attempts: {str(e)}")
                # Notify the application about the failure
                try:
                    await supabase.table("late_meeting")\
                        .update({
                            "transcript_storage_failed": True,
                            "transcript_storage_error": str(e)
                        })\
                        .eq("id", meeting_obj_id)\
                        .execute()
                except Exception as notify_error:
                    logger.error(f"Failed to notify about transcript storage failure: {str(notify_error)}")
                return False  # Failed after all retries

    return False  # Should not reach here, but just in case


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.start(port=port, host="0.0.0.0")
