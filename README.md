<div align="center">
  <h2>Amurex Backend</h2>

  <p>
    <a href="https://github.com/thepersonalaicompany/amurex/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License" />
    </a>
    <a href="https://chrome.google.com/webstore/detail/amurex/dckidmhhpnfhachdpobgfbjnhfnmddmc">
      <img src="https://img.shields.io/chrome-web-store/v/dckidmhhpnfhachdpobgfbjnhfnmddmc.svg" alt="Chrome Web Store" />
    </a>
    <a href="https://twitter.com/thepersonalaico">
      <img src="https://img.shields.io/twitter/follow/thepersonalaico?style=social" alt="Twitter Follow" />
    </a>
    <a href="https://discord.gg/ftUdQsHWbY">
      <img alt="Discord" src="https://img.shields.io/discord/1306591395804348476">
    </a>
  </p>
</div>



## Amurex Backend

This is the backend for the entire Amurex project. You can use it to host your own backend instance of Amurex.

## Prerequisites

- Python 3.11
- Redis server
- Docker (optional)
- Required API keys:
  - Supabase credentials
  - OpenAI API key (optional if using CLIENT_MODE=LOCAL)
  - Groq API key (optional if using CLIENT_MODE=LOCAL)
  - Mistral AI key (optional if using CLIENT_MODE=LOCAL)

Note: When using CLIENT_MODE=LOCAL, you'll need to:
- Install Ollama for local model inference
- Install fast-embed for local embeddings generation

## Supabase Setup

1. Create a new project on [Supabase](https://supabase.com)

2. Create the following tables in your Supabase database:

### Meetings Table

You can find the SQL for this table in `supabase/migrations/20241201195715_meetings.sql`

3. Set up Storage:
   - Create a new bucket named `meeting_context_files`
   - Set the bucket's privacy settings according to your needs
   - Make sure to configure CORS if needed

4. Get your credentials:
   - Go to Project Settings > API
   - Copy your `Project URL` (this will be your SUPABASE_URL)
   - Copy your `anon/public` key (this will be your SUPABASE_ANON_KEY)

Create a `.env` file in the root directory with the following variables:

```env
# Required
SUPABASE_URL=your_project_url
SUPABASE_ANON_KEY=your_anon_key
REDIS_USERNAME=your_redis_username
REDIS_URL=your_redis_host
REDIS_PASSWORD=your_redis_password
REDIS_PORT=your_redis_port
RESEND_API_KEY=your_resend_api_key
RESEND_NOREPLY=your_resend_noreply

# Optional if CLIENT_MODE=LOCAL
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_api_key

# Mode Selection
CLIENT_MODE=ONLINE #set LOCAL to run local Ollama instead of OpenAI and Groq API and fast-embed instead of mistralai

#Ollama mode : 
OLLAMA_ENDPOINT=<your_ollama_endpoint>
```

## Installation

### Option 1: Local Installation
0. Change directory to `server`:

```
cd server
```

1. Create a virtual environment:

```
python -m venv venv
```

2. Install dependencies:

```
⚠ **Caution:**  
If running the project **locally**, make sure to **comment out** `fastembed==0.4.2` from `requirements.txt`.  

pip install -r requirements.txt
```

3 Start the application

```
python index.py
```

### Option 2: Docker

1. Clone the repository

```
git clone https://github.com/thepersonalaicompany/amurex-backend
```

2. Change directory to `amurex-backend`:

```
cd amurex-backend
```

3. Edit the .env file

   Add the various keys for the service
   
```
vim .env
```

4. Build the Docker image:

```
docker build -t amurex-backend .
```

5. Run the Docker container:

```
docker run -d --name amurex-backend --restart unless-stopped amurex-backend:latest
```

Alternatively, use docker compose:

```bash
docker compose up
```

<div align="center">
  Made with ❤️ for better <del>meetings</del> life
</div>

