import OpenAI from 'openai'
import { spawn } from 'child_process'
import { promises as fs } from 'fs'
import path from 'path'
import os from 'os'
import { AssemblyAI } from 'assemblyai'

export interface TimelineSegment {
  startTime: string
  endTime: string
  title: string
  description?: string
}

export interface AnalyzedSegment extends TimelineSegment {
  relevanceScore: number
  reasoning: string
  topics: string[]
}

export interface TimelineAnalysisResult {
  videoInfo: {
    title: string
    duration: string
    channel: string
    description: string
  }
  segments: AnalyzedSegment[]
  relevantSegments: AnalyzedSegment[]
  processingTime: number
  userInterests: string
  extractedKeywords: string[]
  source: 'youtube' | 'assemblyai_python' | 'llm_fallback' | 'fallback'
  audioAnalyzerData?: {
    segments: any[]
    processingTime: number
  }
}

export interface ProgressUpdate {
  step: string
  progress: number
  message: string
  details?: any
}

export class TimelineAnalyzer {
  private openai: OpenAI
  private gemini: any
  private assemblyAI: AssemblyAI
  private progressCallback?: (update: ProgressUpdate) => void
  private selectedModel: 'openai' | 'gemini'

  constructor(progressCallback?: (update: ProgressUpdate) => void, model: 'openai' | 'gemini' = 'openai') {
    this.selectedModel = model
    
    // Initialize OpenAI
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    })
    
    // Initialize Gemini
    if (model === 'gemini') {
      const { GoogleGenerativeAI } = require('@google/generative-ai')
      this.gemini = new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
    }
    
    this.assemblyAI = new AssemblyAI({ 
      apiKey: process.env.ASSEMBLYAI_API_KEY || ''
    })
    this.progressCallback = progressCallback
  }

  private updateProgress(step: string, progress: number, message: string, details?: any) {
    if (this.progressCallback) {
      this.progressCallback({ step, progress, message, details })
    }
    console.log(`[${step}] ${progress}% - ${message}`)
  }

  private async extractKeywordsWithLLM(interests: string): Promise<string[]> {
    try {
      console.log(`Using ${this.selectedModel.toUpperCase()} for keyword extraction...`)
      console.log('User interests:', interests)
      
      if (this.selectedModel === 'gemini') {
        return await this.extractKeywordsWithGemini(interests)
      } else {
        return await this.extractKeywordsWithOpenAI(interests)
      }
    } catch (error) {
      console.error(`${this.selectedModel.toUpperCase()} keyword extraction failed:`, error)
      console.log('Falling back to enhanced keyword extraction...')
      // Fallback to enhanced extraction
      return this.extractKeywordsFallback(interests)
    }
  }

  private async extractKeywordsWithOpenAI(interests: string): Promise<string[]> {
    const response = await this.openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: "You are an expert at understanding user learning intent. Your job is to extract 2-4 CORE LEARNING TOPICS that represent what the user actually wants to learn about. Focus on SUBSTANTIVE CONCEPTS, TECHNOLOGIES, or DOMAINS - not individual words, verbs, or filler words. Examples: 'LLM reasoning' (not 'reasoning' alone), 'post-training techniques' (not 'post-training' alone), 'reinforcement learning from human feedback' (not separate words). Return ONLY a comma-separated list of meaningful learning topics, no explanations."
        },
        {
          role: "user",
          content: `Extract core learning topics from this user interest: "${interests}"`
        }
      ],
      max_tokens: 100,
      temperature: 0.1
    })

    const extractedText = response.choices[0]?.message?.content || ""
    const concepts = extractedText
      .split(',')
      .map((concept: string) => concept.trim().toLowerCase())
      .filter((concept: string) => concept.length > 3) // Filter out very short concepts
      .filter((concept: string) => !this.isStopWord(concept)) // Additional filtering
      .slice(0, 4) // Limit to 4 core concepts

    console.log('OpenAI extracted learning topics:', concepts)
    return concepts
  }

  private async extractKeywordsWithGemini(interests: string): Promise<string[]> {
    const model = this.gemini.getGenerativeModel({ model: "gemini-pro" })
    
    const prompt = `You are an expert at understanding user learning intent. Your job is to extract 2-4 CORE LEARNING TOPICS that represent what the user actually wants to learn about. Focus on SUBSTANTIVE CONCEPTS, TECHNOLOGIES, or DOMAINS - not individual words, verbs, or filler words. Examples: 'LLM reasoning' (not 'reasoning' alone), 'post-training techniques' (not 'post-training' alone), 'reinforcement learning from human feedback' (not separate words). Return ONLY a comma-separated list of meaningful learning topics, no explanations.

Extract core learning topics from this user interest: "${interests}"`

    const result = await model.generateContent(prompt)
    const response = await result.response
    const extractedText = response.text()

    const concepts = extractedText
      .split(',')
      .map((concept: string) => concept.trim().toLowerCase())
      .filter((concept: string) => concept.length > 3) // Filter out very short concepts
      .filter((concept: string) => !this.isStopWord(concept)) // Additional filtering
      .slice(0, 4) // Limit to 4 core concepts

    console.log('Gemini extracted learning topics:', concepts)
    return concepts
  }

  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      'interested', 'about', 'these', 'topics', 'learning', 'exploring', 'actively', 
      'catching', 'up', 'with', 'latest', 'research', 'updates', 'want', 'to', 'learn',
      'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
      'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
      'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
      'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
      'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below',
      'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
      'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
      'just', 'should', 'now', 'like', 'from'
    ])
    return stopWords.has(word.toLowerCase())
  }

  private extractKeywordsFallback(interests: string): string[] {
    console.log('Using enhanced fallback keyword extraction for:', interests)
    
    // Enhanced concept extraction focusing on meaningful learning topics
    const interestText = interests.toLowerCase()
    
    // Define comprehensive stop words
    const stopWords = new Set([
      'interested', 'about', 'these', 'topics', 'learning', 'exploring', 'actively', 
      'catching', 'up', 'with', 'latest', 'research', 'updates', 'want', 'to', 'learn',
      'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
      'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
      'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
      'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
      'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below',
      'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
      'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
      'just', 'should', 'now', 'like', 'from'
    ])
    
    // Extract meaningful concepts using pattern matching
    const concepts: string[] = []
    
    // Look for specific technical patterns
    const technicalPatterns = [
      { pattern: /llm\s+reasoning/gi, concept: 'llm reasoning' },
      { pattern: /post-training\s+techniques?/gi, concept: 'post-training techniques' },
      { pattern: /reinforcement\s+learning\s+from\s+human\s+feedback/gi, concept: 'reinforcement learning from human feedback' },
      { pattern: /machine\s+learning/gi, concept: 'machine learning' },
      { pattern: /artificial\s+intelligence/gi, concept: 'artificial intelligence' },
      { pattern: /neural\s+networks?/gi, concept: 'neural networks' },
      { pattern: /deep\s+learning/gi, concept: 'deep learning' },
      { pattern: /natural\s+language\s+processing/gi, concept: 'natural language processing' },
      { pattern: /computer\s+vision/gi, concept: 'computer vision' },
      { pattern: /data\s+science/gi, concept: 'data science' },
      { pattern: /software\s+engineering/gi, concept: 'software engineering' }
    ]
    
    // Check for technical patterns first
    for (const { pattern, concept } of technicalPatterns) {
      if (pattern.test(interestText)) {
        concepts.push(concept)
      }
    }
    
    // If no technical patterns found, extract meaningful words
    if (concepts.length === 0) {
      const words = interestText.split(/[,\s]+/)
        .filter(word => word.length > 3) // Filter out short words
        .filter(word => !stopWords.has(word)) // Filter out stop words
        .filter(word => /^[a-zA-Z]+$/.test(word)) // Only alphabetic words
      
      // Look for compound concepts (2-3 word phrases)
      const phrases = []
      const wordArray = interestText.split(/[,\s]+/)
      for (let i = 0; i < wordArray.length - 1; i++) {
        const phrase = `${wordArray[i]} ${wordArray[i + 1]}`
        if (phrase.length > 5 && 
            !stopWords.has(wordArray[i]) && 
            !stopWords.has(wordArray[i + 1])) {
          phrases.push(phrase)
        }
      }
      
      concepts.push(...words.slice(0, 3), ...phrases.slice(0, 2))
    }
    
    // Remove duplicates and limit to 4 concepts
    const uniqueConcepts = Array.from(new Set(concepts)).slice(0, 4)
    
    console.log('Enhanced fallback extracted concepts:', uniqueConcepts)
    return uniqueConcepts
  }

  async analyzeVideo(url: string, interests: string, useAudioAnalyzer: boolean = false): Promise<TimelineAnalysisResult> {
    const startTime = Date.now()
    
    try {
      console.log('Starting analyzeVideo with:', { url, interests, useAudioAnalyzer, model: this.selectedModel })
      
      // Extract learning concepts from user interests using LLM
      this.updateProgress('keyword_extraction', 10, `Extracting learning concepts with ${this.selectedModel.toUpperCase()}...`)
      const extractedKeywords = await this.extractKeywordsWithLLM(interests)
      this.updateProgress('keyword_extraction', 100, 'Learning concepts extracted', {
        concepts: extractedKeywords,
        originalInterests: interests
      })
      
      // Step 1: Extract video info and description
      this.updateProgress('extract_video_info', 10, 'Extracting video information...')
      const videoInfo = await this.extractVideoInfo(url)
      this.updateProgress('extract_video_info', 100, 'Video information extracted', {
        title: videoInfo.title,
        duration: videoInfo.duration,
        channel: videoInfo.channel
      })
      
      // Step 2: Parse timeline from description
      this.updateProgress('parse_timeline', 10, 'Analyzing video timeline structure...')
      let segments = this.parseTimelineFromDescription(videoInfo.description)
      let timelineSource: 'youtube' | 'assemblyai_python' | 'llm_fallback' | 'fallback' = 'youtube'
      let audioAnalyzerData: { segments: any[], processingTime: number } | undefined = undefined
      
      // Fallback to Python AssemblyAI analysis with timeout and LLM fallback
      if (segments.length === 0) {
        if (useAudioAnalyzer) {
          this.updateProgress('parse_timeline', 50, 'No YouTube segments found, starting audio analysis...')
          
          try {
            // Try AudioAnalyzer with 60-minute timeout (audio analysis can take 30+ minutes)
            const analysisPromise = this.analyzeWithPythonBackend(url, interests)
            const timeoutPromise = new Promise((_, reject) => 
              setTimeout(() => reject(new Error('Audio analysis timed out after 60 minutes')), 3600000)
            )
            
            const pythonResult = await Promise.race([analysisPromise, timeoutPromise]) as any
            
            // Check if we got actual segments (not just single "entire video")
            const hasMultipleSegments = pythonResult.segments && pythonResult.segments.length > 1
            
            if (hasMultipleSegments) {
              // Store AudioAnalyzer data for later use
              audioAnalyzerData = {
                segments: pythonResult.segments,
                processingTime: pythonResult.processingTime || 0
              }
              
              // Convert to timeline format but keep original AudioAnalyzer data
              segments = pythonResult.segments.map((segment: any) => ({
                startTime: segment.startTime,
                endTime: segment.endTime,
                title: segment.title,
                description: segment.description
              }))
              
              timelineSource = 'assemblyai_python'
              
              this.updateProgress('parse_timeline', 100, 'Python AssemblyAI analysis complete', {
                totalSegments: segments.length,
                segments: segments.map(s => ({ time: s.startTime, title: s.title })),
                source: 'assemblyai_python'
              })
            } else {
              // AudioAnalyzer returned single segment, use LLM fallback
              throw new Error('AudioAnalyzer returned single segment, using LLM fallback')
            }
          } catch (pythonError) {
            console.error('Audio analysis failed or timed out, trying to get transcript for LLM:', pythonError)
            
            try {
              // NEW: Get transcript directly from AssemblyAI for LLM analysis
              this.updateProgress('parse_timeline', 70, 'Getting transcript from AssemblyAI for LLM analysis...')
              const transcript = await this.getTranscriptFromAudio(url)
              
              // Use LLM to create segments from the transcript
              this.updateProgress('parse_timeline', 80, 'Using LLM to analyze transcript and create segments...')
              segments = await this.generateSegmentsWithLLMFromTranscript(videoInfo, interests, extractedKeywords, transcript)
              timelineSource = 'llm_fallback'
              
              this.updateProgress('parse_timeline', 100, 'LLM-generated segments from transcript ready', {
                totalSegments: segments.length,
                segments: segments.map(s => ({ time: s.startTime, title: s.title })),
                source: 'llm_fallback'
              })
            } catch (transcriptError) {
              console.error('Failed to get transcript or LLM analysis failed, using description-only:', transcriptError)
              
              try {
                // Fallback to description-only LLM analysis
                this.updateProgress('parse_timeline', 75, 'Using LLM to generate segments from video description...')
                segments = await this.generateSegmentsWithLLM(videoInfo, interests, extractedKeywords)
                timelineSource = 'llm_fallback'
                
                this.updateProgress('parse_timeline', 100, 'LLM-generated segments ready', {
                  totalSegments: segments.length,
                  segments: segments.map(s => ({ time: s.startTime, title: s.title })),
                  source: 'llm_fallback'
                })
              } catch (llmError) {
                console.error('LLM segment generation also failed, using single segment:', llmError)
                
                // Final fallback: Create a single segment for the entire video
                segments = [{
                  startTime: '0:00',
                  endTime: videoInfo.duration,
                  title: videoInfo.title,
                  description: 'Full video content - no segments available'
                }]
                
                timelineSource = 'fallback'
                
                this.updateProgress('parse_timeline', 100, 'Fallback to single segment', {
                  totalSegments: 1,
                  segments: [{ time: '0:00', title: videoInfo.title }],
                  source: 'fallback'
                })
              }
            }
          }
        } else {
          // Audio analyzer not enabled, skip directly to LLM fallback
          this.updateProgress('parse_timeline', 50, 'No YouTube segments found, using LLM to generate segments...')
          
          try {
            // Fallback to description-only LLM analysis
            this.updateProgress('parse_timeline', 75, 'Using LLM to generate segments from video description...')
            segments = await this.generateSegmentsWithLLM(videoInfo, interests, extractedKeywords)
            timelineSource = 'llm_fallback'
            
            this.updateProgress('parse_timeline', 100, 'LLM-generated segments ready', {
              totalSegments: segments.length,
              segments: segments.map(s => ({ time: s.startTime, title: s.title })),
              source: 'llm_fallback'
            })
          } catch (llmError) {
            console.error('LLM segment generation failed, using single segment:', llmError)
            
            // Final fallback: Create a single segment for the entire video
            segments = [{
              startTime: '0:00',
              endTime: videoInfo.duration,
              title: videoInfo.title,
              description: 'Full video content - no segments available'
            }]
            
            timelineSource = 'fallback'
            
            this.updateProgress('parse_timeline', 100, 'Fallback to single segment', {
              totalSegments: 1,
              segments: [{ time: '0:00', title: videoInfo.title }],
              source: 'fallback'
            })
          }
        }
      } else {
        this.updateProgress('parse_timeline', 100, 'Video timeline analyzed', {
          totalSegments: segments.length,
          segments: segments.map(s => ({ time: s.startTime, title: s.title })),
          source: 'youtube'
        })
      }
      
      // Step 3: Use selected model to analyze relevance with extracted concepts
      this.updateProgress('analyze_relevance', 10, `Matching segments with learning concepts using ${this.selectedModel.toUpperCase()}...`)
      const analyzedSegments = await this.analyzeRelevanceWithLLM(segments, interests, extractedKeywords)
      this.updateProgress('analyze_relevance', 100, 'Segment relevance analysis completed', {
        analyzedSegments: analyzedSegments.map(s => ({
          time: s.startTime,
          title: s.title,
          score: s.relevanceScore,
          reasoning: s.reasoning
        }))
      })
      
      // Step 4: Filter most relevant segments
      this.updateProgress('filter_segments', 10, 'Selecting most relevant video segments...')
      const relevantSegments = analyzedSegments
        .filter(segment => segment.relevanceScore >= 25) // Lowered threshold to be more inclusive
        .sort((a, b) => b.relevanceScore - a.relevanceScore)
        .slice(0, 3) // Top 3 most relevant segments
      
      this.updateProgress('filter_segments', 100, 'Top segments selected for recommendations', {
        relevantSegments: relevantSegments.map(s => ({
          time: s.startTime,
          title: s.title,
          score: s.relevanceScore,
          reasoning: s.reasoning
        }))
      })
      
      console.log(`Found ${relevantSegments.length} highly relevant segments`)
      
      return {
        videoInfo,
        segments: analyzedSegments,
        relevantSegments,
        processingTime: Date.now() - startTime,
        userInterests: interests,
        extractedKeywords,
        source: timelineSource,
        audioAnalyzerData: audioAnalyzerData
      }
    } catch (error: any) {
      console.error('Timeline analysis failed:', error)
      this.updateProgress('error', 100, `Analysis failed: ${error.message}`)
      throw new Error(`Timeline analysis failed: ${error.message}`)
    }
  }

  private async extractVideoInfo(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
      // Check if cookies file exists
      const cookiesPath = path.join(process.cwd(), 'cookies.txt')
      const args = [
        '--dump-json',
        '--no-download',
        '--no-check-certificate',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      ]
      
      // Add cookies if file exists
      try {
        if (require('fs').existsSync(cookiesPath)) {
          args.push('--cookies', cookiesPath)
          console.log('Using cookies from:', cookiesPath)
        } else {
          console.log('No cookies.txt found, proceeding without cookies')
        }
      } catch (e) {
        console.log('Could not check for cookies file:', e)
      }
      
      args.push(url)
      
      const ytdlp = spawn('yt-dlp', args)

      let output = ''
      let errorOutput = ''

      ytdlp.stdout.on('data', (data) => {
        output += data.toString()
      })

      ytdlp.stderr.on('data', (data) => {
        errorOutput += data.toString()
      })

      ytdlp.on('close', (code) => {
        if (code === 0) {
          try {
            const info = JSON.parse(output)
            resolve({
              title: info.title,
              duration: this.formatDuration(info.duration),
              channel: info.uploader,
              description: info.description || ''
            })
          } catch (parseError) {
            reject(new Error(`Failed to parse video info: ${parseError}`))
          }
        } else {
          // Check for specific YouTube bot detection error
          if (errorOutput.includes('Sign in to confirm') || errorOutput.includes('not a bot')) {
            reject(new Error(`YouTube bot detection triggered. Please update yt-dlp: run 'yt-dlp -U' or 'pip install --upgrade yt-dlp'`))
          } else {
            reject(new Error(`yt-dlp failed with code ${code}: ${errorOutput}`))
          }
        }
      })
    })
  }

  private parseTimelineFromDescription(description: string): TimelineSegment[] {
    const segments: TimelineSegment[] = []
    
    // Try multiple timestamp patterns
    const patterns = [
      // Pattern with start-end range: "0:00:00 - 0:04:18 Title"
      /(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)(?=\n|$)/g,
      // (0:00) or (0:00:00)
      /\((\d{1,2}:\d{2}(?::\d{2})?)\)\s*([^\n]+)/g,
      // [0:00] or [0:00:00]
      /\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^\n]+)/g,
      // 0:00 Title (at line start)
      /^(\d{1,2}:\d{2}(?::\d{2})?)\s+([^\n]+)/gm,
      // \n0:00 Title
      /\n(\d{1,2}:\d{2}(?::\d{2})?)\s+([^\n]+)/g
    ]
    
    for (const pattern of patterns) {
      const matches = Array.from(description.matchAll(pattern))
      if (matches.length > 0) {
        console.log(`Found ${matches.length} timestamps with pattern: ${pattern}`)
        
        // Check if this is the range pattern (has 3 capture groups)
        const isRangePattern = matches[0].length === 4 // [fullMatch, start, end, title]
        
        for (let i = 0; i < matches.length; i++) {
          const match = matches[i]
          
          if (isRangePattern) {
            // Format: "0:00:00 - 0:04:18 Title"
            segments.push({
              startTime: match[1],
              endTime: match[2],
              title: match[3].trim()
            })
          } else {
            // Format: "0:00 Title" (calculate end time from next segment)
            const startTime = match[1]
            const title = match[2].trim()
            const nextMatch = matches[i + 1]
            const endTime = nextMatch ? nextMatch[1] : null
            
            segments.push({
              startTime,
              endTime: endTime || this.calculateEndTime(startTime, segments),
              title
            })
          }
        }
        break // Use first pattern that finds segments
      }
    }
    
    console.log(`Parsed ${segments.length} timeline segments from description`)
    return segments
  }

  private calculateEndTime(startTime: string, existingSegments: TimelineSegment[]): string {
    // Simple heuristic: assume 5 minutes per segment if no end time
    const [minutes, seconds] = startTime.split(':').map(Number)
    const totalSeconds = minutes * 60 + seconds + 300 // Add 5 minutes
    const endMinutes = Math.floor(totalSeconds / 60)
    const endSeconds = totalSeconds % 60
    return `${endMinutes}:${endSeconds.toString().padStart(2, '0')}`
  }

  private async analyzeRelevanceWithLLM(segments: TimelineSegment[], interests: string, extractedKeywords: string[]): Promise<AnalyzedSegment[]> {
    try {
      console.log(`Using ${this.selectedModel.toUpperCase()} for semantic relevance analysis...`)
      
      if (this.selectedModel === 'gemini') {
        return await this.analyzeRelevanceWithGemini(segments, interests, extractedKeywords)
      } else {
        return await this.analyzeRelevanceWithOpenAI(segments, interests, extractedKeywords)
      }
    } catch (error) {
      console.error(`${this.selectedModel.toUpperCase()} analysis failed:`, error)
      console.log('Falling back to semantic keyword matching...')
      // Use semantic matching instead of simple keyword matching
      return this.semanticRelevanceAnalysis(segments, interests, extractedKeywords)
    }
  }

  private async analyzeRelevanceWithOpenAI(segments: TimelineSegment[], interests: string, extractedKeywords: string[]): Promise<AnalyzedSegment[]> {
    const prompt = `
You are analyzing video segments to determine their semantic relevance to user learning intent.

User Learning Intent: "${interests}"
Core Learning Topics: ${extractedKeywords.join(', ')}

Video Segments:
${segments.map((seg, i) => `${i + 1}. ${seg.startTime} - ${seg.title}`).join('\n')}

For each segment, analyze the SEMANTIC RELEVANCE to the user's learning intent, not just keyword matches. Consider:
- Does this segment teach about the core learning topics?
- Is the content conceptually related to what the user wants to learn?
- Would this segment help the user achieve their learning goals?

For each segment, provide:
1. Relevance score (0-100) based on semantic alignment with learning intent
2. Brief reasoning explaining the conceptual connection
3. Key topics covered in the segment

Focus on conceptual relevance, not just word matching. Higher scores for segments that directly address the user's learning goals.

Respond in JSON format:
{
  "segments": [
    {
      "index": 0,
      "relevanceScore": 85,
      "reasoning": "This segment directly covers LLM reasoning techniques, which matches the user's interest in understanding how LLMs think and reason.",
      "topics": ["LLM reasoning", "neural networks", "attention mechanisms"]
    }
  ]
}
`

    try {
      console.log('Using OpenAI GPT for semantic relevance analysis...')
      const completion = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are an expert at analyzing video content and determining relevance to user interests. Always respond with valid JSON."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.3,
        max_tokens: 2000
      })

      const responseText = completion.choices[0]?.message?.content || ''
      console.log('OpenAI response:', responseText)
      
      // Parse JSON response
      const jsonMatch = responseText.match(/\{[\s\S]*\}/)
      if (!jsonMatch) {
        throw new Error('No JSON found in OpenAI response')
      }
      
      const analysis = JSON.parse(jsonMatch[0])
      
      // Map analysis back to segments
      return segments.map((segment, index) => {
        const analysisItem = analysis.segments.find((item: any) => item.index === index)
        return {
          ...segment,
          relevanceScore: analysisItem?.relevanceScore || 0,
          reasoning: analysisItem?.reasoning || 'No analysis provided',
          topics: analysisItem?.topics || []
        }
      })
    } catch (error) {
      console.error('OpenAI analysis failed:', error)
      console.log('Falling back to keyword matching...')
      // Fallback: semantic keyword matching
      return this.semanticRelevanceAnalysis(segments, interests, extractedKeywords)
    }
  }

  private async analyzeRelevanceWithGemini(segments: TimelineSegment[], interests: string, extractedKeywords: string[]): Promise<AnalyzedSegment[]> {
    const prompt = `
You are analyzing video segments to determine their semantic relevance to user learning intent.

User Learning Intent: "${interests}"
Core Learning Topics: ${extractedKeywords.join(', ')}

Video Segments:
${segments.map((seg, i) => `${i + 1}. ${seg.startTime} - ${seg.title}`).join('\n')}

For each segment, analyze the SEMANTIC RELEVANCE to the user's learning intent, not just keyword matches. Consider:
- Does this segment teach about the core learning topics?
- Is the content conceptually related to what the user wants to learn?
- Would this segment help the user achieve their learning goals?

For each segment, provide:
1. Relevance score (0-100) based on semantic alignment with learning intent
2. Brief reasoning explaining the conceptual connection
3. Key topics covered in the segment

Focus on conceptual relevance, not just word matching. Higher scores for segments that directly address the user's learning goals.

Respond in JSON format:
{
  "segments": [
    {
      "index": 0,
      "relevanceScore": 85,
      "reasoning": "This segment directly covers LLM reasoning techniques, which matches the user's interest in understanding how LLMs think and reason.",
      "topics": ["LLM reasoning", "neural networks", "attention mechanisms"]
    }
  ]
}
`

    try {
      console.log('Using Gemini for semantic relevance analysis...')
      const model = this.gemini.getGenerativeModel({ model: "gemini-pro" })
      
      const result = await model.generateContent(prompt)
      const response = await result.response
      const responseText = response.text()
      
      console.log('Gemini response:', responseText)
      
      // Parse JSON response
      const jsonMatch = responseText.match(/\{[\s\S]*\}/)
      if (!jsonMatch) {
        throw new Error('No JSON found in Gemini response')
      }
      
      const analysis = JSON.parse(jsonMatch[0])
      
      // Map the analysis back to segments
      return segments.map((segment, index) => {
        const analysisItem = analysis.segments.find((item: any) => item.index === index)
        if (analysisItem) {
          return {
            ...segment,
            relevanceScore: analysisItem.relevanceScore,
            reasoning: analysisItem.reasoning,
            topics: analysisItem.topics || []
          }
        } else {
          // Fallback for segments not analyzed
          return {
            ...segment,
            relevanceScore: 0,
            reasoning: 'Not analyzed',
            topics: []
          }
        }
      })
    } catch (error) {
      console.error('Gemini analysis failed:', error)
      console.log('Falling back to keyword matching...')
      // Fallback: semantic keyword matching
      return this.semanticRelevanceAnalysis(segments, interests, extractedKeywords)
    }
  }

  private semanticRelevanceAnalysis(segments: TimelineSegment[], interests: string, extractedKeywords: string[]): AnalyzedSegment[] {
    console.log('Using semantic relevance analysis with extracted topics:', extractedKeywords)
    console.log('User interests:', interests)
    console.log('Number of segments to analyze:', segments.length)
    
    return segments.map((segment, index) => {
      const segmentText = segment.title.toLowerCase()
      let score = 0
      let matchedConcepts: string[] = []
      let reasoning = ''
      
      console.log(`Analyzing segment ${index + 1}: "${segment.title}"`)
      
      // Semantic matching based on learning topics
      for (const topic of extractedKeywords) {
        const topicLower = topic.toLowerCase()
        
        // Direct topic match
        if (segmentText.includes(topicLower)) {
          score += 40
          matchedConcepts.push(topic)
          reasoning += `Direct match with learning topic: ${topic}. `
          console.log(`  ✓ Direct match found: ${topic}`)
        }
        
        // Semantic variations and related terms
        const semanticMatches = this.getSemanticMatches(topicLower, segmentText)
        if (semanticMatches.length > 0) {
          score += semanticMatches.length * 20  // Increased from 15 to 20
          matchedConcepts.push(...semanticMatches)
          reasoning += `Semantic match with ${topic}: ${semanticMatches.join(', ')}. `
          console.log(`  ✓ Semantic matches for ${topic}: ${semanticMatches.join(', ')}`)
        }
      }
      
      // Special handling for technical concepts
      const technicalMatches = this.getTechnicalMatches(interests.toLowerCase(), segmentText)
      if (technicalMatches.length > 0) {
        score += technicalMatches.length * 20
        matchedConcepts.push(...technicalMatches)
        reasoning += `Technical concept match: ${technicalMatches.join(', ')}. `
        console.log(`  ✓ Technical matches: ${technicalMatches.join(', ')}`)
      }
      
      // Cap the score at 100
      score = Math.min(score, 100)
      
      // Generate reasoning if none provided
      if (!reasoning) {
        reasoning = `No direct semantic match found with learning topics: ${extractedKeywords.join(', ')}`
        console.log(`  ✗ No matches found for segment: "${segment.title}"`)
      }
      
      console.log(`  Final score: ${score}%`)
      
      return {
        ...segment,
        relevanceScore: score,
        reasoning: reasoning.trim(),
        topics: Array.from(new Set(matchedConcepts))
      }
    })
  }

  private getSemanticMatches(topic: string, segmentText: string): string[] {
    const semanticMap: { [key: string]: string[] } = {
      // AI/ML terms
      'llm reasoning': ['reasoning', 'thinking', 'cognition', 'inference', 'logic', 'chain of thought', 'step-by-step'],
      'post-training techniques': ['fine-tuning', 'alignment', 'rlhf', 'reinforcement learning', 'human feedback', 'post-training', 'training'],
      'machine learning': ['ml', 'ai', 'artificial intelligence', 'neural networks', 'deep learning', 'algorithms', 'models'],
      'neural networks': ['neural', 'deep learning', 'ai', 'machine learning', 'transformer', 'attention', 'layers'],
      'reinforcement learning': ['rl', 'rlhf', 'reward', 'policy', 'agent', 'reinforcement', 'feedback'],
      'natural language processing': ['nlp', 'language', 'text', 'transformer', 'bert', 'gpt', 'llm', 'language model'],
      'computer vision': ['vision', 'image', 'visual', 'cnn', 'detection', 'recognition', 'computer vision'],
      'artificial intelligence': ['ai', 'artificial intelligence', 'machine learning', 'neural networks', 'intelligence'],
      'data science': ['data', 'analytics', 'statistics', 'analysis', 'insights', 'data science'],
      'software engineering': ['engineering', 'development', 'programming', 'coding', 'software', 'engineering'],
      // Health/Fitness/Wellness terms
      'fitness': ['wellness', 'health', 'exercise', 'workout', 'training', 'physical', 'body', 'nutrition', 'diet', 'influencer', 'healthy'],
      'health': ['wellness', 'fitness', 'medical', 'healthcare', 'healthy', 'medicine', 'doctor', 'nutrition', 'diet', 'body'],
      'wellness': ['health', 'fitness', 'wellbeing', 'healthy', 'holistic', 'lifestyle', 'self-care', 'mental health'],
      'nutrition': ['diet', 'food', 'eating', 'nutrients', 'health', 'wellness', 'supplements', 'vitamins'],
      'exercise': ['workout', 'training', 'fitness', 'physical', 'movement', 'gym', 'sports', 'activity'],
      'diet': ['nutrition', 'food', 'eating', 'health', 'weight', 'calories', 'meal', 'fasting']
    }
    
    const variations = semanticMap[topic] || []
    return variations.filter(variation => segmentText.includes(variation))
  }

  private getTechnicalMatches(interests: string, segmentText: string): string[] {
    const technicalTerms = [
      'rlhf', 'reinforcement learning from human feedback',
      'transformer', 'attention mechanism', 'self-attention',
      'fine-tuning', 'pre-training', 'post-training',
      'alignment', 'safety', 'bias', 'fairness',
      'prompt engineering', 'few-shot learning', 'zero-shot',
      'chain of thought', 'reasoning', 'inference'
    ]
    
    return technicalTerms.filter(term => 
      interests.includes(term) && segmentText.includes(term)
    )
  }

  private fallbackRelevanceAnalysis(segments: TimelineSegment[], interests: string, extractedKeywords?: string[]): AnalyzedSegment[] {
    // Use provided extracted keywords if available, otherwise extract them
    let concepts: string[]
    
    if (extractedKeywords && extractedKeywords.length > 0) {
      concepts = extractedKeywords
      console.log('Using provided extracted keywords for fallback analysis:', concepts)
    } else {
      // Extract meaningful concepts from user interests using smarter logic
      const interestText = interests.toLowerCase()
      
      // Define stop words to ignore (including common pronouns and articles)
      const stopWords = new Set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through',
        'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just', 'should', 'now', 'want', 'like', 'learn', 'about', 'interested', 'in', 'to', 'from'
      ])
      
      // Extract meaningful words (filter out stop words and short words)
      const words = interestText.split(/[,\s]+/).filter(word => word.length > 2 && !stopWords.has(word))
      
      // Look for compound concepts (2-3 word phrases)
      const phrases = []
      const wordArray = interestText.split(/[,\s]+/)
      for (let i = 0; i < wordArray.length - 1; i++) {
        const phrase = `${wordArray[i]} ${wordArray[i + 1]}`
        if (phrase.length > 5 && !phrase.includes('i ') && !phrase.includes(' me ') && !phrase.includes(' my ')) {
          phrases.push(phrase)
        }
      }
      
      // Combine individual words and phrases
      concepts = [...words, ...phrases]
      console.log('Fallback extracted concepts:', concepts)
    }
    
    return segments.map(segment => {
      const segmentText = segment.title.toLowerCase()
      let score = 0
      let matchedConcepts: string[] = []
      
      // Score based on concept matches
      for (const concept of concepts) {
        if (segmentText.includes(concept)) {
          // Phrase matches get higher scores
          if (concept.includes(' ')) {
            score += 30
          } else {
            score += 15
          }
          matchedConcepts.push(concept)
        }
      }
      
      // Special handling for technical terms
      const technicalTerms = ['ai', 'ml', 'machine learning', 'artificial intelligence', 'rlhf', 'reinforcement learning', 'neural network', 'deep learning', 'nlp', 'computer vision']
      for (const term of technicalTerms) {
        if (interests.toLowerCase().includes(term) && segmentText.includes(term)) {
          score += 25
          matchedConcepts.push(term)
        }
      }
      
      // Cap the score at 100
      score = Math.min(score, 100)
      
      return {
        ...segment,
        relevanceScore: score,
        reasoning: score > 0 ? `Matches concepts: ${matchedConcepts.join(', ')}` : 'No relevant concepts found',
        topics: matchedConcepts
      }
    })
  }

  async extractRelevantAudioChunks(url: string, segments: AnalyzedSegment[]): Promise<string[]> {
    console.log('Extracting relevant audio chunks...')
    const audioPaths: string[] = []
    
    // Create realistic mock audio files for demonstration
    // In production, this would extract actual audio chunks using ffmpeg
    console.log('Creating realistic mock audio chunks for AssemblyAI demonstration')
    
    for (const segment of segments) {
      const mockPath = path.join(os.tmpdir(), `demo_audio_chunk_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.wav`)
      
      // Create a minimal WAV file header for testing
      try {
        const fs = require('fs')
        // Create a minimal WAV file (44 bytes header + some silence)
        const wavHeader = Buffer.from([
          0x52, 0x49, 0x46, 0x46, // "RIFF"
          0x24, 0x00, 0x00, 0x00, // File size - 8
          0x57, 0x41, 0x56, 0x45, // "WAVE"
          0x66, 0x6D, 0x74, 0x20, // "fmt "
          0x10, 0x00, 0x00, 0x00, // Subchunk1Size
          0x01, 0x00,             // AudioFormat (PCM)
          0x01, 0x00,             // NumChannels
          0x44, 0xAC, 0x00, 0x00, // SampleRate (44100)
          0x88, 0x58, 0x01, 0x00, // ByteRate
          0x02, 0x00,             // BlockAlign
          0x10, 0x00,             // BitsPerSample
          0x64, 0x61, 0x74, 0x61, // "data"
          0x00, 0x00, 0x00, 0x00  // Subchunk2Size
        ])
        
        fs.writeFileSync(mockPath, wavHeader)
        audioPaths.push(mockPath)
        console.log(`Created demo audio chunk for: ${segment.startTime} - ${segment.endTime}`)
      } catch (error) {
        console.error(`Failed to create demo audio file:`, error)
        // Fallback to just the path
        audioPaths.push(mockPath)
      }
    }
    
    return audioPaths
  }

  private async downloadAudioChunk(url: string, outputPath: string, startSeconds: number, endSeconds: number): Promise<void> {
    // Simplified approach - just create a mock file for now
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(`Mock audio chunk created: ${outputPath}`)
        resolve()
      }, 100)
    })
  }

  async transcribeAudioChunks(audioPaths: string[]): Promise<any[]> {
    const transcripts = []
    
    console.log('Starting AssemblyAI transcription for audio chunks...')
    
    for (const audioPath of audioPaths) {
      try {
        console.log(`Transcribing: ${audioPath}`)
        const transcript = await this.transcribeWithAssemblyAI(audioPath)
        
        // Extract key information for the recommendation
        const summary = this.generateTranscriptSummary(transcript)
        
        transcripts.push({
          text: transcript.text,
          utterances: transcript.utterances || [],
          summary: summary,
          confidence: transcript.confidence || 0.95,
          words: transcript.words || [],
          auto_highlights_result: transcript.auto_highlights_result,
          sentiment_analysis_results: transcript.sentiment_analysis_results,
          entities: transcript.entities,
          iab_categories_result: transcript.iab_categories_result,
          auto_chapters_result: transcript.auto_chapters_result,
          assemblyai_summary: transcript.summary
        })
        
        console.log(`Successfully transcribed: ${audioPath}`)
      } catch (error) {
        console.error(`Failed to transcribe ${audioPath}:`, error)
        
        // Push error information instead of mock data
        transcripts.push({
          text: 'Transcription failed - unable to process audio',
          utterances: [],
          summary: 'Audio transcription failed. Please try again or check your AssemblyAI API key.',
          confidence: 0,
          words: [],
          auto_highlights_result: null,
          sentiment_analysis_results: [],
          entities: [],
          iab_categories_result: null,
          auto_chapters_result: null,
          assemblyai_summary: null,
          error: error instanceof Error ? error.message : 'Unknown error'
        })
        console.log(`Transcription failed for: ${audioPath}`)
      }
    }
    
    return transcripts
  }

  private generateTranscriptSummary(transcript: any): string {
    try {
      // Extract key insights from AssemblyAI transcript
      const text = transcript.text || ''
      const utterances = transcript.utterances || []
      
      // Get key terms if available
      const keyTerms = transcript.auto_highlights_result?.results?.map((r: any) => r.text) || []
      
      // Get sentiment if available
      const sentiment = transcript.sentiment_analysis_results?.[0]?.sentiment || 'neutral'
      
      // Get speaker information
      const speakers = Array.from(new Set(utterances.map((u: any) => u.speaker).filter(Boolean)))
      
      // Get entities if available
      const entities = transcript.entities || []
      const importantEntities = entities.filter((e: any) => e.entity_type === 'PERSON' || e.entity_type === 'ORG' || e.entity_type === 'PRODUCT')
      
      // Get IAB categories if available
      const categories = transcript.iab_categories_result?.results || []
      const topCategories = categories.slice(0, 3).map((c: any) => c.label)
      
      // Generate comprehensive summary
      let summary = `📊 Audio Analysis Summary: `
      
      if (keyTerms.length > 0) {
        summary += `Key topics: ${keyTerms.slice(0, 5).join(', ')}. `
      }
      
      if (speakers.length > 0) {
        summary += `Speakers: ${speakers.join(', ')}. `
      }
      
      if (importantEntities.length > 0) {
        const entityNames = importantEntities.map((e: any) => e.text).slice(0, 3)
        summary += `Key entities: ${entityNames.join(', ')}. `
      }
      
      if (topCategories.length > 0) {
        summary += `Content categories: ${topCategories.join(', ')}. `
      }
      
      summary += `Sentiment: ${sentiment}. `
      
      // Add a brief excerpt from the transcript
      const excerpt = text.length > 150 ? text.substring(0, 150) + '...' : text
      summary += `Content preview: "${excerpt}"`
      
      return summary
    } catch (error) {
      console.error('Error generating transcript summary:', error)
      return 'Audio analysis completed. Content discusses relevant topics from the video segment.'
    }
  }

  private async transcribeWithAssemblyAI(audioPath: string): Promise<any> {
    try {
      const audioUrl = await this.assemblyAI.files.upload(audioPath)
      
      const config = {
        audio_url: audioUrl,
        speaker_labels: true,
        auto_highlights: true,
        sentiment_analysis: true,
        entity_detection: true,
        iab_categories: true,
        auto_chapters: true,
        summarization: true,
        summary_type: 'bullets' as const
      }
      
      const transcript = await this.assemblyAI.transcripts.transcribe(config)
      
      // Wait for completion
      let attempts = 0
      const maxAttempts = 30
      
      while (transcript.status !== 'completed' && transcript.status !== 'error' && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        const updatedTranscript = await this.assemblyAI.transcripts.get(transcript.id)
        Object.assign(transcript, updatedTranscript)
        attempts++
      }
      
      if (transcript.status === 'error') {
        throw new Error(`AssemblyAI transcription failed: ${transcript.error}`)
      }
      
      return transcript
    } catch (error) {
      console.error('AssemblyAI transcription error:', error)
      throw error
    }
  }

  private timeToSeconds(timeStr: string): number {
    const [minutes, seconds] = timeStr.split(':').map(Number)
    return minutes * 60 + seconds
  }

  private formatDuration(seconds: number): string {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  async cleanup(audioPaths: string[]): Promise<void> {
    for (const audioPath of audioPaths) {
      try {
        // Check if file exists before trying to delete
        await fs.access(audioPath)
        await fs.unlink(audioPath)
      } catch (error: any) {
        // Only log error if it's not a "file not found" error
        if (error.code !== 'ENOENT') {
          console.error(`Failed to cleanup ${audioPath}:`, error)
        }
      }
    }
  }

  /**
   * Fallback method to analyze video using Python backend when no YouTube segments are found
   * This method preserves the core functionality by only being called as a fallback
   */
  private async analyzeWithPythonBackend(url: string, interests: string): Promise<any> {
    console.log('Calling Python backend for AssemblyAI analysis...')
    
    const response = await fetch('/api/analyze-python', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url, interests }),
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Python backend API failed: ${response.status} ${response.statusText} - ${errorText}`)
    }
    
    const result = await response.json()
    
    if (!result.success) {
      throw new Error(result.error || 'Python backend analysis failed')
    }
    
    console.log(`Python backend returned ${result.segments?.length || 0} segments`)
    return result
  }

  /**
   * Generate segments using LLM when audio analysis fails or times out
   */
  private async generateSegmentsWithLLM(
    videoInfo: any, 
    interests: string, 
    keywords: string[]
  ): Promise<TimelineSegment[]> {
    
    const durationSeconds = this.parseDurationToSeconds(videoInfo.duration)
    const prompt = `You are analyzing a YouTube video to create intelligent segments.

Video Title: ${videoInfo.title}
Video Duration: ${videoInfo.duration}
Video Description: ${videoInfo.description?.substring(0, 500) || 'No description available'}

User Interests: ${interests}
Key Topics: ${keywords.join(', ')}

Based on the video information above, create 5-8 intelligent segments that would be useful for learning about the user's interests.

Rules:
- Each segment should be 2-5 minutes long
- Start times must be in MM:SS format
- End times must be in MM:SS format  
- Segments should be logically related to the video content
- Focus on segments relevant to the user's interests
- Total duration should not exceed video duration (${videoInfo.duration})

Respond in JSON format:
{
  "segments": [
    {
      "startTime": "0:00",
      "endTime": "3:45",
      "title": "Introduction to Topic X",
      "description": "Brief overview of the key concepts"
    }
  ]
}`

    try {
      if (this.selectedModel === 'gemini') {
        const model = this.gemini.getGenerativeModel({ model: "gemini-pro" })
        const result = await model.generateContent(prompt)
        const response = result.response.text()
        
        // Extract JSON from response
        const jsonMatch = response.match(/\{[\s\S]*\}/)
        if (jsonMatch) {
          const data = JSON.parse(jsonMatch[0])
          return data.segments || []
        }
      } else {
        const completion = await this.openai.chat.completions.create({
          model: "gpt-3.5-turbo",
          messages: [
            { role: "system", content: "You are an expert at analyzing educational video content. Always respond with valid JSON." },
            { role: "user", content: prompt }
          ],
          temperature: 0.7,
          max_tokens: 2000
        })
        
        const responseText = completion.choices[0]?.message?.content || ''
        const jsonMatch = responseText.match(/\{[\s\S]*\}/)
        if (jsonMatch) {
          const data = JSON.parse(jsonMatch[0])
          return data.segments || []
        }
      }
      
      throw new Error('Failed to parse LLM response')
    } catch (error) {
      console.error('LLM segment generation failed:', error)
      throw error
    }
  }

  private parseDurationToSeconds(duration: string): number {
    const parts = duration.split(':').map(Number)
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2]
    } else if (parts.length === 2) {
      return parts[0] * 60 + parts[1]
    }
    return 0
  }

  /**
   * Get transcript from AssemblyAI for LLM analysis when AudioAnalyzer fails
   */
  private async getTranscriptFromAudio(url: string): Promise<string> {
    console.log('Getting transcript from AssemblyAI for LLM analysis...')
    
    try {
      // Download audio (faster quality for just getting transcript)
      const audioPath = await this.downloadAudioForTranscript(url)
      
      // Upload to AssemblyAI
      const audioUrl = await this.assemblyAI.files.upload(audioPath)
      
      // Transcribe with minimal config (just get text, faster)
      const config = {
        audio_url: audioUrl,
        summarization: true,
        summary_type: 'bullets' as const
      }
      
      const transcript = await this.assemblyAI.transcripts.transcribe(config)
      
      // Wait for completion
      let attempts = 0
      const maxAttempts = 30
      
      while (transcript.status !== 'completed' && transcript.status !== 'error' && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        const updatedTranscript = await this.assemblyAI.transcripts.get(transcript.id)
        Object.assign(transcript, updatedTranscript)
        attempts++
      }
      
      if (transcript.status === 'error') {
        throw new Error(`AssemblyAI transcription failed: ${transcript.error}`)
      }
      
      // Clean up
      await this.cleanup([audioPath])
      
      // Return transcript text (first 5000 chars for LLM)
      return transcript.text?.substring(0, 5000) || ''
    } catch (error) {
      console.error('Failed to get transcript from audio:', error)
      throw error
    }
  }

  private async downloadAudioForTranscript(url: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const tempDir = os.tmpdir()
      const outputPath = path.join(tempDir, `audio_transcript_${Date.now()}.mp3`)
      
      // Check if cookies file exists
      const cookiesPath = path.join(process.cwd(), 'cookies.txt')
      const args = [
        '--extract-audio',
        '--audio-format', 'mp3',
        '--audio-quality', '64K',  // Lower quality for faster download
        '--output', outputPath,
      ]
      
      // Add cookies if file exists
      try {
        if (require('fs').existsSync(cookiesPath)) {
          args.push('--cookies', cookiesPath)
        }
      } catch (e) {
        // Ignore
      }
      
      args.push(url)
      
      const ytdlp = spawn('yt-dlp', args)
      
      ytdlp.on('close', (code) => {
        if (code === 0) {
          resolve(outputPath)
        } else {
          reject(new Error(`yt-dlp failed with code ${code}`))
        }
      })
      
      ytdlp.on('error', (error) => {
        reject(error)
      })
    })
  }

  /**
   * Generate segments using LLM from transcript when AudioAnalyzer fails
   */
  private async generateSegmentsWithLLMFromTranscript(
    videoInfo: any,
    interests: string,
    keywords: string[],
    transcript: string
  ): Promise<TimelineSegment[]> {
    
    const durationSeconds = this.parseDurationToSeconds(videoInfo.duration)
    
    const prompt = `You are analyzing a YouTube video transcript to create intelligent segments.

Video Title: ${videoInfo.title}
Video Duration: ${videoInfo.duration}
User Interests: ${interests}
Key Topics: ${keywords.join(', ')}

TRANSCRIPT:
${transcript}

Based on the transcript above, create 5-8 intelligent segments that would be useful for learning about the user's interests.

Rules:
- Analyze the transcript content to understand what's being discussed
- Each segment should be 2-5 minutes long
- Start times must be in MM:SS format
- End times must be in MM:SS format
- Segments should cover different topics discussed in the video
- Focus on segments relevant to the user's interests
- Total duration should not exceed video duration (${videoInfo.duration})

Respond in JSON format:
{
  "segments": [
    {
      "startTime": "0:00",
      "endTime": "3:45",
      "title": "Introduction to Topic X",
      "description": "Brief overview of the key concepts"
    }
  ]
}`

    try {
      if (this.selectedModel === 'gemini') {
        const model = this.gemini.getGenerativeModel({ model: "gemini-pro" })
        const result = await model.generateContent(prompt)
        const response = result.response.text()
        
        const jsonMatch = response.match(/\{[\s\S]*\}/)
        if (jsonMatch) {
          const data = JSON.parse(jsonMatch[0])
          return data.segments || []
        }
      } else {
        const completion = await this.openai.chat.completions.create({
          model: "gpt-3.5-turbo",
          messages: [
            { role: "system", content: "You are an expert at analyzing educational video content. Always respond with valid JSON." },
            { role: "user", content: prompt }
          ],
          temperature: 0.7,
          max_tokens: 3000
        })
        
        const responseText = completion.choices[0]?.message?.content || ''
        const jsonMatch = responseText.match(/\{[\s\S]*\}/)
        if (jsonMatch) {
          const data = JSON.parse(jsonMatch[0])
          return data.segments || []
        }
      }
      
      throw new Error('Failed to parse LLM response')
    } catch (error) {
      console.error('LLM segment generation from transcript failed:', error)
      throw error
    }
  }
}
