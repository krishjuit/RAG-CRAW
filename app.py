import os
import sys

# Safeguard against protobuf _upb metaclass incompatibility on Python >= 3.14
if sys.version_info >= (3, 14):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
from rag import RAG


# Ensure Windows terminal handles UTF-8 correctly
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

load_dotenv('.env')

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment or .env file!")
    print("Please add GOOGLE_API_KEY=your_key to your .env file.")
    sys.exit(1)

website_url = "https://winone.in/"
question = "summarize the website"

print(f"Loading and processing {website_url}...")
chatbot = RAG(website_url, api_key)

print(f"\nAsking question: {question}\n")
response = chatbot.get_response(question)

print("=" * 60)
print("ANSWER:")
print("=" * 60)
print(response.get("answer", "No answer generated."))

sources = response.get("sources", [])
if sources:
    print("\n" + "=" * 60)
    print(f"SOURCES ({len(sources)}):")
    print("=" * 60)
    for i, s in enumerate(sources, 1):
        print(f"\n--- Source {i} ---")
        print(s[:400] + ("..." if len(s) > 400 else ""))