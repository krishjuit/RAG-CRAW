from rag import RAG
import os
from dotenv import load_dotenv

load_dotenv('.env')

website_url = "https://winone.in/"
# website_url = "https://degrees.apps.asu.edu/masters-phd/major/ASU00/ESCOMSCMS/computer-science-ms?init=false&nopassive=true"
# question = "what are the deadlines for the applicaiton?"
question = "summarize the website"

# chatbot = RAG(website_url, os.getenv('GOOGLE_API_KEY'))
chatbot = RAG(website_url, os.getenv('GOOGLE_API_KEY'), recursive_count=500)
response = chatbot.get_response(question)
print(response)