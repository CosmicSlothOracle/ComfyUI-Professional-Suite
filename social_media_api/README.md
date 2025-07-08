# Social Media Video Generation API

A professional, documented, robust API pipeline with ComfyUI and Google VEO3 for daily, AI-based generation of social media videos based on current trends for the target audience of 16-27 year olds in Germany.

## 🚀 Features

- **Automated Trend Analysis**: Scrapes and analyzes current trends from TikTok and Instagram
- **NLP-based Content Analysis**: Classifies content semantically and clusters content mechanics
- **Video Generation with Google VEO3**: Creates engaging vertical videos (1080x1920) based on trends
- **Post Text Generation**: Creates viral captions with appropriate hashtags
- **Dashboard UI**: Monitor status, view generated content, and control the pipeline
- **Scheduled Generation**: Automatically generates content daily at 21:30
- **Robust Error Handling**: No fake success or dummy data when sources are unavailable

## 📋 System Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Trend Analysis │─────▶│   NLP Analysis  │─────▶│ Video Generation│
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Error Handling Layer                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │   ComfyUI UI    │
                        └─────────────────┘
```

## 🛠️ Technical Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: ComfyUI Dashboard, HTML/CSS/JS
- **AI/ML**: Google VEO3, NLTK, spaCy, scikit-learn
- **Data Processing**: pandas, numpy
- **Scheduling**: Python schedule library
- **Error Handling**: Comprehensive logging and notification system

## ⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/social-media-video-generator.git
cd social-media-video-generator
```

2. Install dependencies
```bash
pip install -r social_media_api/requirements.txt
```

3. Configure API credentials
- Place Google API credentials in `social_media_api/config/google_credentials.json`
- Update configuration in `social_media_api/config/dashboard_config.json`

4. Start the API server
```bash
cd social_media_api
uvicorn api_server.social_media_api:app --reload
```

5. Start the ComfyUI dashboard
```bash
python comfy_integration/comfy_ui_dashboard.py
```

## 🖥️ Dashboard Usage

1. Access the dashboard at `http://localhost:8000/ui`
2. Configure settings for trend analysis, video generation, and post text
3. Monitor the system status and view generated content
4. Trigger manual generation or wait for scheduled generation

## 📊 API Endpoints

- `GET /`: API status and information
- `GET /health`: Health check endpoint
- `POST /trends/analyze`: Analyze current social media trends
- `POST /video/generate`: Generate a social media video
- `POST /post/generate`: Generate post text for a video
- `GET /tasks/{task_id}`: Get task status
- `GET /tasks`: List all tasks
- `GET /download/{filename}`: Download generated file
- `DELETE /tasks/{task_id}`: Delete task and associated files
- `GET /stats`: Get API statistics

## ⚠️ Error Handling

The system is designed to handle errors gracefully and avoid generating fake content:

- If trend analysis fails, the pipeline stops with an error
- If NLP analysis fails, the pipeline stops with an error
- If video generation fails, the pipeline stops with an error
- All errors are logged and displayed in the dashboard

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Google VEO3 API for video generation
- ComfyUI for the dashboard interface
- NLTK and spaCy for NLP processing