'use client'

import { useState, useRef, useEffect } from 'react'

interface AnalysisStep {
  name: string
  progress: number
  status: 'pending' | 'in_progress' | 'completed' | 'error'
  message: string
  details?: string
}

interface TimelineSegment {
  time: string
  title: string
  relevanceScore: number
  reasoning: string
  topics: string[]
}

interface Recommendation {
  startTime: string
  endTime: string
  summary: string
  relevanceScore: number
  topics: string[]
  reasoning: string
  transcript: string
}

export default function Home() {
  const [url, setUrl] = useState('')
  const [interests, setInterests] = useState('')
  const [model, setModel] = useState<'openai' | 'gemini'>('openai')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [progressSteps, setProgressSteps] = useState<AnalysisStep[]>([])
  const [selectedSegment, setSelectedSegment] = useState<TimelineSegment | null>(null)
  const [videoId, setVideoId] = useState<string>('')
  const videoRef = useRef<HTMLIFrameElement>(null)

  // Extract YouTube video ID from URL
  const extractVideoId = (url: string): string => {
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/)
    return match ? match[1] : ''
  }

  // Convert time string to seconds
  const timeToSeconds = (timeStr: string): number => {
    const parts = timeStr.split(':')
    if (parts.length === 2) {
      return parseInt(parts[0]) * 60 + parseInt(parts[1])
    } else if (parts.length === 3) {
      return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseInt(parts[2])
    }
    return 0
  }

  // Jump to specific time in video
  const jumpToTime = (timeStr: string) => {
    const seconds = timeToSeconds(timeStr)
    if (videoRef.current) {
      const newUrl = `https://www.youtube.com/embed/${videoId}?start=${seconds}&autoplay=1`
      videoRef.current.src = newUrl
    }
  }

  // Auto-load video at recommended time when results are ready
  useEffect(() => {
    if (result && result.success && result.recommendations && result.recommendations.length > 0 && videoId) {
      const firstRecommendation = result.recommendations[0]
      console.log('Auto-loading video at:', firstRecommendation.startTime)
      jumpToTime(firstRecommendation.startTime)
    }
  }, [result, videoId])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Form submitted with URL:', url, 'Interests:', interests)
    setLoading(true)
    setError('')
    setResult(null)
    setSelectedSegment(null)
    
    const extractedVideoId = extractVideoId(url)
    setVideoId(extractedVideoId)
    console.log('Extracted video ID:', extractedVideoId)
    
    // Initialize progress steps for timeline analysis
    const initialSteps: AnalysisStep[] = [
      {
        name: 'keyword_extraction',
        progress: 0,
        status: 'pending',
        message: `Extracting learning concepts with ${model === 'openai' ? 'OpenAI GPT' : 'Gemini'}...`
      },
      {
        name: 'extract_video_info',
        progress: 0,
        status: 'pending',
        message: 'Extracting video information...'
      },
      {
        name: 'parse_timeline',
        progress: 0,
        status: 'pending',
        message: 'Analyzing video timeline structure...'
      },
      {
        name: 'analyze_relevance',
        progress: 0,
        status: 'pending',
        message: `Matching segments with learning concepts using ${model === 'openai' ? 'OpenAI GPT' : 'Gemini'}...`
      },
      {
        name: 'filter_segments',
        progress: 0,
        status: 'pending',
        message: 'Selecting most relevant video segments...'
      },
      {
        name: 'extract_audio_chunks',
        progress: 0,
        status: 'pending',
        message: 'Extracting relevant audio chunks...'
      },
      {
        name: 'transcribe_chunks',
        progress: 0,
        status: 'pending',
        message: 'Transcribing audio chunks...'
      }
    ]
    setProgressSteps(initialSteps)

    try {
      console.log('Starting API call...')
      const response = await fetch('/api/analyze-timeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url,
          interests,
          model,
        }),
      })

      const data = await response.json()
      console.log('API response received:', data)
      console.log('Response success:', data.success)
      
      // Update progress steps with final results
      if (data.steps) {
        console.log('Updating progress steps:', data.steps)
        setProgressSteps(data.steps)
      }
      
      setResult(data)
      console.log('Result set:', data)
    } catch (err: any) {
      console.error('API Error:', err)
      setError(err.message || 'An error occurred while processing your request')
      
      // Update progress steps to show error
      setProgressSteps(prev => prev.map(step => ({
        ...step,
        status: step.status === 'in_progress' ? 'error' : step.status,
        message: step.status === 'in_progress' ? `Error: ${err.response?.data?.error || 'Unknown error'}` : step.message
      })))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              🎯 WeVid Podcast Analyzer
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Get personalized podcast recommendations with AI-powered analysis. 
              Watch the video while exploring relevant segments based on your interests!
            </p>
          </div>

          {/* Main Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* URL Input */}
                <div>
                  <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
                    📹 YouTube URL
                  </label>
                  <input
                    type="url"
                    id="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    required
                  />
                </div>

                {/* Model Selection */}
                <div>
                  <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-2">
                    🤖 AI Model
                  </label>
                  <select
                    id="model"
                    value={model}
                    onChange={(e) => setModel(e.target.value as 'openai' | 'gemini')}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  >
                    <option value="openai">OpenAI GPT-3.5 Turbo</option>
                    <option value="gemini">Google Gemini Pro</option>
                  </select>
                </div>
              </div>

              {/* Interests Input - Full Width */}
              <div>
                <label htmlFor="interests" className="block text-sm font-medium text-gray-700 mb-2">
                  🎯 Describe What You Want to Learn
                </label>
                <textarea
                  id="interests"
                  value={interests}
                  onChange={(e) => setInterests(e.target.value)}
                  placeholder="I want to learn about reinforcement learning from human feedback (RLHF) and how it's used to train AI models to be more helpful and safe..."
                  rows={3}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 resize-none"
                  required
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || !url || !interests}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-4 px-6 rounded-lg transition-all duration-200 flex items-center justify-center text-lg"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Content...
                  </>
                ) : (
                  '🚀 Analyze & Get Recommendations'
                )}
              </button>
            </form>
          </div>

          {/* Simplified Progress Display */}
          {loading && (
            <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
              <div className="text-center">
                <div className="flex items-center justify-center mb-4">
                  <svg className="animate-spin w-8 h-8 text-blue-600 mr-3" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <h2 className="text-2xl font-bold text-gray-900">🤖 AI Analysis in Progress</h2>
                </div>
                <p className="text-lg text-gray-600 mb-4">
                  Our AI is analyzing your interests and matching them with video content...
                </p>
                <div className="bg-blue-50 rounded-lg p-4 max-w-md mx-auto">
                  <p className="text-sm text-blue-800">
                    <strong>Current Step:</strong> {progressSteps.find(step => step.status === 'in_progress')?.message || 'Processing...'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}

          {/* Results Display */}
          {result && result.success && (
            <div className="space-y-8">
              {/* User Interests & Keywords */}
              {result.analysisDetails && (
                <div className="bg-white rounded-2xl shadow-xl p-8">
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">🎯 Your Analysis Profile</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-blue-50 rounded-lg p-4">
                      <p className="text-sm font-medium text-blue-800 mb-2">Original Interests:</p>
                      <p className="text-blue-700 italic">"{result.analysisDetails.userInterests}"</p>
                    </div>
                    <div className="bg-green-50 rounded-lg p-4">
                      <p className="text-sm font-medium text-green-800 mb-2">Learning Concepts:</p>
                      <div className="flex flex-wrap gap-2">
                        {result.analysisDetails.extractedKeywords?.map((concept: string, index: number) => (
                          <span key={index} className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full font-medium">
                            {concept}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Top Recommendations - MOVED TO TOP */}
              {result.recommendations && result.recommendations.length > 0 && (
                <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl shadow-xl p-8">
                  <h3 className="text-2xl font-bold text-gray-800 mb-6">🏆 Top Recommendations</h3>
                  <div className="grid gap-8">
                    {result.recommendations.map((rec: Recommendation, index: number) => (
                      <div key={index} className="bg-white rounded-lg p-6 border border-green-200 shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                          <button
                            onClick={() => jumpToTime(rec.startTime)}
                            className="text-xl font-bold text-green-600 hover:text-green-700 transition-colors"
                          >
                            ▶️ {rec.startTime} - {rec.endTime}
                          </button>
                          <span className="text-sm font-medium text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                            {rec.relevanceScore}% Match
                          </span>
                        </div>
                        <h4 className="text-lg font-semibold text-gray-800 mb-3">{rec.summary}</h4>
                        <div className="bg-blue-50 rounded-lg p-4 mb-4">
                          <p className="text-sm text-blue-800">
                            <strong>Why this segment is relevant:</strong> {rec.reasoning}
                          </p>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-4 mb-4">
                          <p className="text-sm text-gray-700">
                            <strong>Transcript Preview:</strong> {rec.transcript}
                          </p>
                          {rec.transcriptSummary && (
                            <div className="mt-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                              <p className="text-sm text-blue-800">
                                <strong>📊 AI Analysis Summary:</strong> {rec.transcriptSummary}
                              </p>
                              {rec.confidence && (
                                <p className="text-xs text-blue-600 mt-1">
                                  Confidence: {Math.round(rec.confidence * 100)}%
                                </p>
                              )}
                            </div>
                          )}
                          
                          {/* Enhanced AssemblyAI Analysis Details */}
                          {rec.auto_highlights_result && (
                            <div className="mt-3 p-3 bg-green-50 rounded-lg border-l-4 border-green-400">
                              <p className="text-sm text-green-800">
                                <strong>🔑 Key Highlights:</strong> {rec.auto_highlights_result.results?.map((r: any) => r.text).join(', ') || 'No highlights available'}
                              </p>
                            </div>
                          )}
                          
                          {rec.sentiment_analysis_results && rec.sentiment_analysis_results.length > 0 && (
                            <div className="mt-3 p-3 bg-purple-50 rounded-lg border-l-4 border-purple-400">
                              <p className="text-sm text-purple-800">
                                <strong>😊 Sentiment Analysis:</strong> {rec.sentiment_analysis_results[0].sentiment} 
                                {rec.sentiment_analysis_results[0].confidence && (
                                  <span className="ml-2 text-xs">
                                    (Confidence: {Math.round(rec.sentiment_analysis_results[0].confidence * 100)}%)
                                  </span>
                                )}
                              </p>
                            </div>
                          )}
                          
                          {rec.entities && rec.entities.length > 0 && (
                            <div className="mt-3 p-3 bg-orange-50 rounded-lg border-l-4 border-orange-400">
                              <p className="text-sm text-orange-800">
                                <strong>🏷️ Key Entities:</strong> {rec.entities.slice(0, 5).map((e: any) => `${e.text} (${e.entity_type})`).join(', ')}
                              </p>
                            </div>
                          )}
                          
                          {rec.iab_categories_result && (
                            <div className="mt-3 p-3 bg-indigo-50 rounded-lg border-l-4 border-indigo-400">
                              <p className="text-sm text-indigo-800">
                                <strong>📂 Content Categories:</strong> {rec.iab_categories_result.results?.slice(0, 3).map((c: any) => c.label).join(', ') || 'No categories available'}
                              </p>
                            </div>
                          )}
                        </div>
                        
                        {/* Embedded Video Clip */}
                        {videoId && (
                          <div className="mb-4">
                            <h5 className="text-md font-semibold text-gray-700 mb-2">📺 Watch This Segment:</h5>
                            <div className="aspect-video rounded-lg overflow-hidden">
                              <iframe
                                src={`https://www.youtube.com/embed/${videoId}?start=${timeToSeconds(rec.startTime)}&autoplay=0`}
                                title={`Video segment: ${rec.startTime} - ${rec.endTime}`}
                                frameBorder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                                className="w-full h-full"
                              ></iframe>
                            </div>
                          </div>
                        )}
                        
                        <div className="flex flex-wrap gap-2">
                          {rec.topics?.map((topic: string, i: number) => (
                            <span key={i} className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full font-medium">
                              {topic}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}


              {/* Video Information */}
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">📋 Video Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-600">Title</p>
                    <p className="text-gray-800 font-semibold">{result.videoInfo?.title}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-600">Duration</p>
                    <p className="text-gray-800 font-semibold">{result.videoInfo?.duration}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-600">Channel</p>
                    <p className="text-gray-800 font-semibold">{result.videoInfo?.channel}</p>
                  </div>
                </div>
              </div>

              {/* Condensed Timeline Analysis - MOVED TO END */}
              {result.analysisDetails?.timelineAnalysis && (
                <div className="bg-white rounded-2xl shadow-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">⏰ All Timeline Segments</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-80 overflow-y-auto">
                    {result.analysisDetails.timelineAnalysis.map((segment: TimelineSegment, index: number) => (
                      <div 
                        key={index} 
                        className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:shadow-md ${
                          segment.relevanceScore >= 70 ? 'bg-green-50 border-green-200 hover:bg-green-100' : 
                          segment.relevanceScore >= 40 ? 'bg-yellow-50 border-yellow-200 hover:bg-yellow-100' : 
                          'bg-gray-50 border-gray-200 hover:bg-gray-100'
                        } ${selectedSegment === segment ? 'ring-2 ring-blue-500' : ''}`}
                        onClick={() => {
                          setSelectedSegment(segment)
                          jumpToTime(segment.time)
                        }}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <button 
                            className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full hover:bg-blue-200 transition-colors"
                            onClick={(e) => {
                              e.stopPropagation()
                              jumpToTime(segment.time)
                            }}
                          >
                            ▶️ {segment.time}
                          </button>
                          <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                            segment.relevanceScore >= 70 ? 'bg-green-100 text-green-800' : 
                            segment.relevanceScore >= 40 ? 'bg-yellow-100 text-yellow-800' : 
                            'bg-gray-100 text-gray-600'
                          }`}>
                            {segment.relevanceScore}%
                          </span>
                        </div>
                        <p className="text-sm font-medium text-gray-800 mb-1 line-clamp-2">{segment.title}</p>
                        <p className="text-xs text-gray-600 line-clamp-2">{segment.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Analysis Summary */}
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">📊 Analysis Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{result.analysisDetails?.totalSegments || 0}</div>
                    <div className="text-sm text-gray-600">Total Segments</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{result.analysisDetails?.relevantSegments || 0}</div>
                    <div className="text-sm text-gray-600">Relevant Segments</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{result.recommendations?.length || 0}</div>
                    <div className="text-sm text-gray-600">Recommendations</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">{Math.round(result.processingTime / 1000)}s</div>
                    <div className="text-sm text-gray-600">Processing Time</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}