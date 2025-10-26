import { NextRequest, NextResponse } from 'next/server'
import { TimelineAnalyzer, ProgressUpdate } from '@/lib/timeline-analyzer'

export async function POST(request: NextRequest) {
  try {
    const { url, interests, model = 'openai' } = await request.json()

    if (!url || !interests) {
      return NextResponse.json(
        { success: false, error: 'URL and interests are required' },
        { status: 400 }
      )
    }

    console.log(`Starting timeline analysis for URL: ${url}`)
    console.log(`User interests: ${interests}`)
    console.log(`Selected model: ${model}`)

    // Create analyzer with progress callback
    const progressUpdates: ProgressUpdate[] = []
    const analyzer = new TimelineAnalyzer((update: ProgressUpdate) => {
      progressUpdates.push(update)
      console.log(`Progress Update: ${update.step} - ${update.progress}% - ${update.message}`)
    }, model)
    
    // Step 1: Analyze timeline and get relevant segments
    const timelineResult = await analyzer.analyzeVideo(url, interests)
    
    console.log(`Found ${timelineResult.relevantSegments.length} relevant segments`)
    
    // Step 2: Extract audio chunks for relevant segments only
    console.log('Extracting relevant audio chunks...')
    const audioPaths = await analyzer.extractRelevantAudioChunks(url, timelineResult.relevantSegments)
    
    // Step 3: Transcribe only the relevant chunks
    console.log('Transcribing relevant audio chunks...')
    const transcripts = await analyzer.transcribeAudioChunks(audioPaths)
    
    // Step 4: Combine results
    const recommendations = timelineResult.relevantSegments.map((segment, index) => ({
      startTime: segment.startTime,
      endTime: segment.endTime,
      summary: segment.title,
      relevanceScore: segment.relevanceScore,
      topics: segment.topics,
      reasoning: segment.reasoning,
      transcript: transcripts[index]?.text || 'Transcription not available',
      transcriptSummary: transcripts[index]?.summary || 'Analysis summary not available',
      confidence: transcripts[index]?.confidence || 0.95,
      auto_highlights_result: transcripts[index]?.auto_highlights_result,
      sentiment_analysis_results: transcripts[index]?.sentiment_analysis_results,
      entities: transcripts[index]?.entities,
      iab_categories_result: transcripts[index]?.iab_categories_result,
      auto_chapters_result: transcripts[index]?.auto_chapters_result
    }))

    // Cleanup
    await analyzer.cleanup(audioPaths)

    return NextResponse.json({
      success: true,
      videoInfo: timelineResult.videoInfo,
      recommendations,
      analysisDetails: {
        totalSegments: timelineResult.segments.length,
        relevantSegments: timelineResult.relevantSegments.length,
        audioChunksExtracted: audioPaths.length,
        transcriptsGenerated: transcripts.length,
        processingTime: timelineResult.processingTime,
        userInterests: timelineResult.userInterests,
        extractedKeywords: timelineResult.extractedKeywords,
        timelineAnalysis: timelineResult.segments.map(seg => ({
          time: seg.startTime,
          title: seg.title,
          relevanceScore: seg.relevanceScore,
          reasoning: seg.reasoning,
          topics: seg.topics
        })),
        progressUpdates: progressUpdates
      },
      steps: [
        {
          name: 'keyword_extraction',
          progress: 100,
          status: 'completed',
          message: 'Keywords extracted from user interests',
          details: `Keywords: ${timelineResult.extractedKeywords.join(', ')}`
        },
        {
          name: 'extract_video_info',
          progress: 100,
          status: 'completed',
          message: 'Video information extracted',
          details: `Title: ${timelineResult.videoInfo.title} | Duration: ${timelineResult.videoInfo.duration}`
        },
        {
          name: 'parse_timeline',
          progress: 100,
          status: 'completed',
          message: 'Timeline parsed from description',
          details: `Found ${timelineResult.segments.length} timeline segments`
        },
        {
          name: 'analyze_relevance',
          progress: 100,
          status: 'completed',
          message: 'Relevance analysis completed with OpenAI GPT',
          details: `${timelineResult.relevantSegments.length} highly relevant segments identified`
        },
        {
          name: 'filter_segments',
          progress: 100,
          status: 'completed',
          message: 'Most relevant segments filtered',
          details: `Selected top ${timelineResult.relevantSegments.length} segments for processing`
        },
        {
          name: 'extract_audio_chunks',
          progress: 100,
          status: 'completed',
          message: 'Relevant audio chunks extracted',
          details: `Extracted ${audioPaths.length} audio chunks`
        },
        {
          name: 'transcribe_chunks',
          progress: 100,
          status: 'completed',
          message: 'Audio chunks transcribed',
          details: `Generated ${transcripts.length} transcripts`
        }
      ]
    })

  } catch (error: any) {
    console.error('Timeline analysis error:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: error.message || 'Timeline analysis failed',
        steps: [
          {
            name: 'error',
            progress: 100,
            status: 'error',
            message: `Analysis failed: ${error.message || 'Unknown error'}`,
            details: 'Check server logs for more details'
          }
        ]
      },
      { status: 500 }
    )
  }
}
