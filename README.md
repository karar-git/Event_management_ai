# SumerTrip AI Services

AI-powered chatbot and recommendation system for SumerTrip event management platform.

## Features

1. **Chatbot Service** - AI assistant specialized in Iraqi tourism
   - Answers questions about trips, events, and bookings
   - Provides travel recommendations and tips
   - Supports Arabic language

2. **RAG-based Recommendation System** - Personalized recommendations
   - Uses vector embeddings for semantic search
   - Matches user preferences with available trips/events
   - Generates AI explanations for recommendations

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Fal.ai OpenRouter** - LLM API (uses Google Gemini)
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **Railway** - Deployment platform

## API Endpoints

### Health Check
```
GET /health
```

### Chatbot
```
POST /api/chat
{
  "message": "ما هي أفضل الرحلات السياحية؟",
  "temperature": 0.7
}
```

### Recommendations
```
POST /api/recommend
{
  "user_info": {
    "interests": ["heritage", "culture"],
    "budget": "medium",
    "duration": "full_day",
    "location": "any"
  },
  "top_k": 5,
  "include_explanation": true
}
```

### Update Data
```
POST /api/data/update
{
  "trips": [...],
  "events": [...]
}
```

## Deployment on Railway

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Add environment variable:
   - `FAL_KEY` - Your Fal.ai API key (get it from https://fal.ai)
4. Deploy!

Railway will automatically detect the `railway.toml` configuration.

## Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp .env.example .env
# Edit .env and add your FAL_KEY
```

4. Run the server:
```bash
python main.py
```

Server will start at `http://localhost:8000`

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Backend Integration

Your backend can call these endpoints:

```javascript
// Chat example
const response = await fetch('https://your-railway-url/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'أريد معرفة المزيد عن رحلة بابل'
  })
});
const data = await response.json();
console.log(data.response);

// Recommendation example
const recs = await fetch('https://your-railway-url/api/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_info: {
      interests: ['heritage', 'nature'],
      budget: 'medium'
    }
  })
});
const recommendations = await recs.json();
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FAL_KEY` | Fal.ai API key | Required |
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |
| `DEFAULT_MODEL` | LLM model | google/gemini-2.5-flash |
# Event_management_ai
