# How to Run

1. Install Docker

2. Download Repository

3. Create Environment File
Inside the backend folder, create a .env file and set the variables
```
OPENAI_API_KEY=""
TAVILY_API_KEY=""

# LANGGRAPH VARIABLES
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT=""
```

4. Build Docker Image
```
docker compose build
```

5. Compose Docker Container
```
docker compose up
```

6. Open FastAPI service
```
http://0.0.0.0:8000/docs
```

7. Test API
Test the service [GET] /scraping
```
https://es.wikipedia.org/wiki/argentina
```