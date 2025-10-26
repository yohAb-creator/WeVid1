from audio_analyzer import AudioAnalyzer, ContentMatcher
from youtube_processor import analyze_youtube_podcast

def main():
    # Initialize the audio analyzer
    analyzer = AudioAnalyzer()

    while True:
        try:
            # Get YouTube URL from user
            print("\nEnter a YouTube video URL (or 'q' to quit):")
            youtube_url = input().strip()

            if youtube_url.lower() == 'q':
                break

            if not youtube_url.startswith('http'):
                print("Please enter a valid YouTube URL starting with http:// or https://")
                continue

            # Process and analyze the podcast
            print("\nStarting analysis...")
            segments = analyze_youtube_podcast(youtube_url, analyzer)
            print(f"\nFound {len(segments)} segments")

            # Initialize content matcher
            matcher = ContentMatcher()
            matcher.add_segments(segments)

            # Get user preferences
            print("\nEnter your preferred topics (comma-separated):")
            topics = [topic.strip() for topic in input().split(",")]

            print("\nEnter your preferred tone (positive/negative/neutral):")
            tone = input().strip().lower()

            # Create user preferences
            user_prefs = {
                "topics": topics,
                "preferred_tone": tone
            }

            # Get content recommendations
            print("\nFinding matches based on your preferences...")
            recommendations = matcher.match_content(user_prefs)

            # Display results
            print("\nTop Recommendations:")
            for i, (segment, score) in enumerate(recommendations[:5], 1):
                print(f"\nMatch #{i} (Relevance Score: {score:.1f}/100)")
                print(f"Time: {segment.start_ms/1000:.1f}s - {segment.end_ms/1000:.1f}s")
                print(f"Topics: {', '.join(segment.topics) if segment.topics else 'N/A'}")
                print(f"Sentiment: {segment.sentiment}")
                print(f"Content: {segment.text}")

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different URL or enter 'q' to quit.")
            continue

if __name__ == "__main__":
    main()
