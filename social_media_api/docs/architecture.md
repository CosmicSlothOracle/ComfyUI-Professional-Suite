# System Architecture

This document outlines the architecture of the Social Media Video Generation API.

## Architecture Overview

The system is designed as a pipeline with several key components that process data sequentially, with robust error handling at each stage.

```mermaid
flowchart TD
    subgraph "Daily Trigger"
        A[Scheduler - 21:30 Daily]
    end

    subgraph "Trend Analysis"
        B1[TikTok Scraper]
        B2[Instagram Scraper]
        B3[Trend Analyzer]
        B1 --> B3
        B2 --> B3
    end

    subgraph "NLP Analysis"
        C1[Sentiment Analysis]
        C2[Content Classification]
        C3[Content Mechanics]
        C1 --> C4[NLP Result]
        C2 --> C4
        C3 --> C4
    end

    subgraph "Video Generation"
        D1[Script Generator]
        D2[VEO3 API]
        D3[Video Post-processing]
        D1 --> D2
        D2 --> D3
    end

    subgraph "Post Generation"
        E1[Text Generator]
        E2[Hashtag Generator]
    end

    subgraph "Error Handling"
        F1[Input Validation]
        F2[API Error Detection]
        F3[Empty Result Detection]
        F4[Notification System]
        F1 --> F4
        F2 --> F4
        F3 --> F4
    end

    subgraph "Dashboard UI"
        G1[Status Monitor]
        G2[Content Preview]
        G3[Manual Controls]
        G4[Configuration]
    end

    A --> B1
    A --> B2
    B3 --> F1
    F1 -->|Valid| C1
    F1 -->|Invalid| F4
    C4 --> F2
    F2 -->|Valid| D1
    F2 -->|Invalid| F4
    D3 --> F3
    F3 -->|Valid| E1
    F3 -->|Invalid| F4
    E1 --> E2

    F4 --> G1
    D3 --> G2
    E2 --> G2
    G3 --> A
    G4 --> B3
    G4 --> D1
    G4 --> E1

    classDef primary fill:#f9f,stroke:#333,stroke-width:2px
    classDef error fill:#f66,stroke:#333,stroke-width:2px
    classDef success fill:#6f6,stroke:#333,stroke-width:2px

    class A,B3,C4,D3,E2 primary
    class F1,F2,F3,F4 error
    class G1,G2,G3,G4 success
```

## Component Descriptions

### 1. Daily Trigger
- **Scheduler**: Runs daily at 21:30 to initiate the video generation process

### 2. Trend Analysis
- **TikTok Scraper**: Extracts trending content from TikTok
- **Instagram Scraper**: Extracts trending content from Instagram
- **Trend Analyzer**: Processes and ranks trends based on popularity and relevance

### 3. NLP Analysis
- **Sentiment Analysis**: Determines the emotional tone of trends
- **Content Classification**: Categorizes content (educational, entertainment, etc.)
- **Content Mechanics**: Identifies content creation techniques used

### 4. Video Generation
- **Script Generator**: Creates a script based on trend and NLP analysis
- **VEO3 API**: Generates video content using Google's VEO3
- **Video Post-processing**: Finalizes video with effects, captions, etc.

### 5. Post Generation
- **Text Generator**: Creates engaging caption text
- **Hashtag Generator**: Adds relevant hashtags for maximum reach

### 6. Error Handling
- **Input Validation**: Ensures data meets requirements before processing
- **API Error Detection**: Monitors for API failures
- **Empty Result Detection**: Prevents processing with insufficient data
- **Notification System**: Alerts about errors and system status

### 7. Dashboard UI
- **Status Monitor**: Displays real-time system status
- **Content Preview**: Shows generated videos and posts
- **Manual Controls**: Allows manual triggering of processes
- **Configuration**: Settings for all system components

## Data Flow

1. The scheduler triggers the pipeline daily at 21:30
2. Trend analysis collects and ranks current social media trends
3. If valid trends are found, NLP analysis classifies and analyzes the content
4. Based on NLP results, a video script is generated and passed to VEO3
5. The generated video is post-processed with captions, effects, etc.
6. Post text and hashtags are generated to accompany the video
7. The final content is available in the dashboard for review
8. At any point, if an error occurs or data is invalid, the process stops and notifies the user

## Error Handling Philosophy

The system follows a strict "fail fast" approach:
- No processing continues with invalid or empty data
- No fake content is generated when sources are unavailable
- All errors are clearly communicated through the dashboard
- Manual intervention is required when automatic processing fails