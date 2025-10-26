export interface AnalysisProgress {
  step: string
  progress: number
  message: string
  details?: string
}

export interface AnalysisStep {
  name: string
  progress: number
  status: 'pending' | 'in_progress' | 'completed' | 'error'
  message: string
  details?: string
  startTime?: number
  endTime?: number
}

export interface ModularAnalysisResult {
  videoInfo: VideoInfo
  recommendations: Recommendation[]
  processingTime: number
  steps: AnalysisStep[]
  success: boolean
  error?: string
}

export interface VideoInfo {
  title: string
  duration: string
  channel: string
  description?: string
}

export interface Recommendation {
  startTime: string
  endTime: string
  summary: string
  relevanceScore: number
  topics: string[]
  transcript: string
  transcriptSummary?: string
  confidence?: number
  auto_highlights_result?: any
  sentiment_analysis_results?: any[]
  entities?: any[]
  iab_categories_result?: any
  auto_chapters_result?: any
}
