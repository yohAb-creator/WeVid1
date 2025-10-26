#!/usr/bin/env python3
"""
Bridge script to integrate Python audio analysis with Next.js application.
This script uses the existing audio_analyzer.py and youtube_processor.py without modification.
"""

import json
import sys
import os
import time
from pathlib import Path

# Import existing modules without modification
from audio_analyzer import AudioAnalyzer, ContentMatcher
from youtube_processor import analyze_youtube_podcast

def convert_segments_to_timeline_format(segments):
    """Convert AudioSegment objects to TimelineSegment format expected by Next.js"""
    timeline_segments = []
    
    for segment in segments:
        # Convert milliseconds to MM:SS format
        start_minutes = int(segment.start_ms // 60000)
        start_seconds = int((segment.start_ms % 60000) // 1000)
        end_minutes = int(segment.end_ms // 60000)
        end_seconds = int((segment.end_ms % 60000) // 1000)
        
        timeline_segment = {
            'startTime': f"{start_minutes}:{start_seconds:02d}",
            'endTime': f"{end_minutes}:{end_seconds:02d}",
            'title': segment.text[:100] + "..." if len(segment.text) > 100 else segment.text,
            'description': segment.text,
            'topics': segment.topics,
            'sentiment': segment.sentiment,
            'confidence': segment.confidence,
            'startMs': segment.start_ms,
            'endMs': segment.end_ms,
            'source': 'assemblyai_python'
        }
        timeline_segments.append(timeline_segment)
    
    return timeline_segments

def extract_video_info_from_url(url):
    """Extract basic video info from URL (placeholder - could be enhanced)"""
    return {
        'title': 'YouTube Video',
        'duration': 'Unknown',
        'channel': 'Unknown',
        'description': 'Analyzed using AssemblyAI Python backend'
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_youtube_bridge.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Read input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        url = input_data['url']
        interests = input_data['interests']
        output_file = input_data['output_file']
        
        print(f"Python Bridge: Analyzing YouTube URL: {url}")
        print(f"Python Bridge: User interests: {interests}")
        
        start_time = time.time()
        
        # Use existing AudioAnalyzer without modification
        analyzer = AudioAnalyzer()
        
        # Use existing analyze_youtube_podcast function without modification
        print("Python Bridge: Starting analysis with existing modules...")
        segments = analyze_youtube_podcast(url, analyzer)
        print(f"Python Bridge: Found {len(segments)} segments")
        
        # Convert to timeline format
        timeline_segments = convert_segments_to_timeline_format(segments)
        
        # Extract video info
        video_info = extract_video_info_from_url(url)
        
        processing_time = time.time() - start_time
        
        # Prepare output data
        output_data = {
            'success': True,
            'segments': timeline_segments,
            'videoInfo': video_info,
            'processingTime': processing_time,
            'source': 'assemblyai_python',
            'originalSegmentCount': len(segments),
            'convertedSegmentCount': len(timeline_segments)
        }
        
        # Write results to output file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Python Bridge: Analysis complete!")
        print(f"Python Bridge: Processed {len(segments)} original segments")
        print(f"Python Bridge: Converted to {len(timeline_segments)} timeline segments")
        print(f"Python Bridge: Processing time: {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"Python Bridge Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Write error to output file
        error_data = {
            'success': False,
            'error': str(e),
            'segments': [],
            'videoInfo': {},
            'processingTime': 0,
            'source': 'assemblyai_python'
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(error_data, f)
        except:
            pass  # If we can't write the error file, just exit
            
        sys.exit(1)

if __name__ == "__main__":
    main()
