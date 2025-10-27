from dataclasses import dataclass
from typing import Dict, List, Tuple
import requests
import time
import os
from dotenv import load_dotenv

@dataclass
class AudioSegment:
    start_ms: int
    end_ms: int
    text: str
    topics: List[str]
    sentiment: str
    confidence: float

class AudioAnalyzer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not self.api_key:
            raise ValueError("AssemblyAI API key not found in environment variables")

        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.api_key
        }

    def _upload_file(self, file_path: str) -> str:
        """Upload a local file to AssemblyAI"""
        import os
        
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Uploading file to AssemblyAI: {file_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        print("Uploading... (this may take several minutes)")
        
        uploaded = 0
        chunk_size = 5242880  # 5MB
        
        def read_file_with_progress(file_path):
            nonlocal uploaded
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    uploaded += len(data)
                    percent = (uploaded / file_size) * 100
                    print(f"\rUpload progress: {percent:.1f}% ({uploaded / (1024*1024):.2f} MB uploaded)", end="", flush=True)
                    yield data
            print("\nUpload complete!")

        # Upload file
        upload_response = requests.post(
            f"{self.base_url}/upload",
            headers=self.headers,
            data=read_file_with_progress(file_path)
        )

        if upload_response.status_code != 200:
            raise RuntimeError(f"Upload failed: {upload_response.text}")

        return upload_response.json()["upload_url"]

    def analyze_audio(self, audio_path: str, is_url: bool = False) -> List[AudioSegment]:
        """
        Analyze audio content using AssemblyAI's API

        Args:
            audio_path (str): Path to audio file or URL
            is_url (bool): Whether the audio_path is a URL

        Returns:
            List[AudioSegment]: List of analyzed audio segments
        """
        # First verify the file exists
        if not is_url and not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Processing audio from: {audio_path}")

        # Handle file upload if needed
        try:
            audio_url = audio_path if is_url else self._upload_file(audio_path)
            print(f"Audio successfully uploaded. URL: {audio_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to upload file: {str(e)}")

        # Create transcription request
        # Adjust payload to match AssemblyAI API schema
        data = {
            "audio_url": audio_url,
            "auto_highlights": True,  # Highlights for key moments
            "auto_chapters": True,  # Chapters for better segmentation
            "iab_categories": True,  # Topic detection
            "sentiment_analysis": True,
            "speaker_labels": True  # Speaker diarization for better segmentation
        }

        # Start transcription
        print("Sending transcription request to AssemblyAI...")
        response = requests.post(
            f"{self.base_url}/transcript",
            headers=self.headers,
            json=data
        )

        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            raise RuntimeError(f"Failed to create transcription. Status code: {response.status_code}")

        try:
            transcript_id = response.json()["id"]
            print(f"Transcription started. ID: {transcript_id}")
        except KeyError:
            print(f"Unexpected API response: {response.text}")
            raise RuntimeError("Failed to get transcription ID from API response")

        # Poll for results
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"

        print("\nProcessing audio... This may take a while.")
        print("Progress updates every 30 seconds:")
        print("-" * 60)

        start_time = time.time()
        
        while True:
            response = requests.get(polling_endpoint, headers=self.headers)
            transcript = response.json()

            status = transcript.get("status")
            elapsed_time = int(time.time() - start_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            
            print(f"\n[{minutes:02d}:{seconds:02d}] Status: {status}")
            
            # Show additional info if available
            if status == "processing":
                if 'words' in transcript:
                    print(f"  Words transcribed: {transcript.get('words', 0)}")
                elif 'audio_duration' in transcript and 'confidence' in transcript:
                    print(f"  Confidence: {transcript.get('confidence', 0):.2f}")

            if status == "completed":
                print("\n" + "="*60)
                print("Processing completed!")
                print(f"Total time: {minutes:02d}:{seconds:02d}")
                print("="*60)
                break
            elif status == "error":
                error = transcript.get("error", "Unknown error")
                raise RuntimeError(f"Transcription failed: {error}")
            else:
                time.sleep(30)  # Check every 30 seconds

        # Process results into segments
        segments = []
        
        print("\nExtracting segments from transcript...")
        
        # PRIORITY 1: Try to get chapters first (most reliable segmentation)
        chapters = transcript.get("chapters", [])
        if chapters:
            print(f"Found {len(chapters)} chapters")
            for chapter in chapters:
                segment = AudioSegment(
                    start_ms=int(chapter["start"]),
                    end_ms=int(chapter["end"]),
                    text=chapter.get("headline", "") or chapter.get("summary", ""),
                    topics=[],
                    sentiment="neutral",
                    confidence=1.0
                )
                segments.append(segment)
        
        # If chapters created segments, return them
        if segments:
            print(f"Created {len(segments)} segments from chapters")
            return segments
        
        # PRIORITY 2: Try to get highlights (auto-highlights feature)
        highlights = transcript.get("auto_highlights_result", {}).get("results", [])
        
        if highlights:
            print(f"Found {len(highlights)} highlighted segments")
            # Debug: print first highlight structure
            if highlights:
                print(f"Debug - First highlight keys: {list(highlights[0].keys())}")
            
            for i, highlight in enumerate(highlights):
                try:
                    # Get timestamps - AssemblyAI returns timestamps as array
                    start_ms = 0
                    end_ms = 0
                    timestamps = highlight.get("timestamps", [])
                    
                    # Debug for first few
                    if i < 3:
                        print(f"  Highlight {i}: {highlight.get('text', '')[:50]}...")
                        print(f"    Timestamps raw: {timestamps}")
                        print(f"    Timestamps type: {type(timestamps)}")
                        if timestamps:
                            print(f"    First element: {timestamps[0]}, type: {type(timestamps[0])}")
                    
                    if timestamps and len(timestamps) > 0:
                        if isinstance(timestamps[0], (list, tuple)) and len(timestamps[0]) >= 2:
                            # Format: [[start_ms, end_ms]] or [[start_s, end_s]]
                            start_val = timestamps[0][0]
                            end_val = timestamps[0][1]
                            # Check if values are in seconds (0-5000 range) or milliseconds
                            if start_val < 10000:  # Likely in seconds
                                start_ms = int(start_val * 1000)
                                end_ms = int(end_val * 1000)
                            else:  # Already in milliseconds
                                start_ms = int(start_val)
                                end_ms = int(end_val)
                        elif len(timestamps) >= 2:
                            # Format: [start, end]
                            start_val = timestamps[0]
                            end_val = timestamps[1]
                            if start_val < 10000:  # Likely in seconds
                                start_ms = int(start_val * 1000)
                                end_ms = int(end_val * 1000)
                            else:  # Already in milliseconds
                                start_ms = int(start_val)
                                end_ms = int(end_val)
                    else:
                        # Fallback to individual start/end fields if timestamps not available
                        if "start" in highlight:
                            start_ms = int(highlight["start"])
                        if "end" in highlight:
                            end_ms = int(highlight["end"])
                    
                    # Get text
                    text = highlight.get("text", "") or highlight.get("snippet", "")
                    
                    # Get rank/confidence - normalize rank to 0-1 scale
                    rank = highlight.get("rank", 0)
                    if isinstance(rank, (int, float)) and rank > 0:
                        # Normalize: rank of 1 = 1.0, rank of 2 = 0.8, etc.
                        confidence = max(0.1, 1.0 / float(rank))
                    else:
                        confidence = 1.0
                    
                    # Only add segment if we have timestamps and text
                    if text:
                        segment = AudioSegment(
                            start_ms=start_ms,
                            end_ms=end_ms if end_ms > 0 else start_ms + 5000,  # Default 5 second clip if no end
                            text=text,
                            topics=[],
                            sentiment="neutral",
                            confidence=confidence
                        )
                        segments.append(segment)
                except Exception as e:
                    print(f"Error processing highlight {i}: {e}")
                    print(f"  Highlight data: {highlight}")
                    continue
        
        # If no highlights, try to use IAB categories to create segments
        if not segments:
            iab_results = transcript.get("iab_categories_result", {}).get("summary", {})
            if iab_results:
                print(f"Found IAB category results")
                # Get all timestamps for topic changes
                topics_data = {}
                for category, results in iab_results.items():
                    for result in results:
                        start_time = result.get("start", 0)
                        end_time = result.get("end", 0)
                        if start_time not in topics_data:
                            topics_data[start_time] = {"end": end_time, "topics": []}
                        if category not in topics_data[start_time]["topics"]:
                            topics_data[start_time]["topics"].append(category)
                
                # Create segments from topic timestamps
                for start_ms, data in sorted(topics_data.items()):
                    segment = AudioSegment(
                        start_ms=int(start_ms * 1000),
                        end_ms=int(data["end"] * 1000),
                        text=f"Topic segment: {', '.join(data['topics'])}",
                        topics=data["topics"],
                        sentiment="neutral",
                        confidence=1.0
                    )
                    segments.append(segment)
        
        # If still no segments, try using utterances
        if not segments:
            utterances = transcript.get("utterances", [])
            if utterances:
                print(f"Found {len(utterances)} utterances")
                for utterance in utterances:
                    segment = AudioSegment(
                        start_ms=int(utterance["start"]),
                        end_ms=int(utterance["end"]),
                        text=utterance.get("text", ""),
                        topics=[],
                        sentiment="neutral",
                        confidence=1.0
                    )
                    segments.append(segment)
        
        # If still no segments, fall back to using full transcript
        if not segments:
            print("No automatic segments found, using full transcript as single segment")
            text = transcript.get("text", "")
            audio_duration = transcript.get("audio_duration")
            if text:
                segment = AudioSegment(
                    start_ms=0,
                    end_ms=int(audio_duration * 1000) if audio_duration else 0,
                    text=text[:500] + "..." if len(text) > 500 else text,
                    topics=[],
                    sentiment="neutral",
                    confidence=1.0
                )
                segments.append(segment)
        
        print(f"Created {len(segments)} segments from transcript")
        return segments

class ContentMatcher:
    def __init__(self):
        self.segments: List[AudioSegment] = []

    def add_segments(self, segments: List[AudioSegment]):
        """Add audio segments to the matcher"""
        self.segments.extend(segments)

    def match_content(self, user_preferences: Dict) -> List[Tuple[AudioSegment, float]]:
        """Match audio segments against user preferences"""
        matches = []
        
        user_topics = [t.lower() for t in user_preferences.get('topics', [])]
        preferred_tone = user_preferences.get('preferred_tone', '').lower()

        for segment in self.segments:
            score = 0.0

            # Match topics (if available)
            if segment.topics:
                common_topics = set([t.lower() for t in segment.topics]) & set(user_topics)
                score += len(common_topics) * 2.0  # Increased from 0.5
            
            # Match text content with keywords (fallback when topics are empty)
            if not segment.topics and user_topics:
                text_lower = segment.text.lower()
                for user_topic in user_topics:
                    if user_topic in text_lower:
                        score += 1.5
                        break  # Count each topic once per segment

            # Match sentiment
            if segment.sentiment.lower() == preferred_tone:
                score += 1.0  # Increased from 0.3

            # Weight by confidence
            score *= segment.confidence

            # Add base score for all segments (so they all show up)
            if score == 0:
                score = 0.1 * segment.confidence

            matches.append((segment, score))

        # Normalize scores to 0-100 scale
        if matches:
            max_score = max(score for _, score in matches)
            if max_score > 0:
                # Normalize to 0-100
                normalized_matches = [(segment, (score / max_score) * 100) 
                                    for segment, score in matches]
                return sorted(normalized_matches, key=lambda x: x[1], reverse=True)
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
