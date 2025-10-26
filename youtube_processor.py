import yt_dlp
import os
from pathlib import Path

class YouTubeProcessor:
    def __init__(self):
        # Set download directory
        self.download_dir = Path('downloads')
        self.download_dir.mkdir(exist_ok=True)
        
        # Configure yt-dlp options for audio extraction
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        # Add cookies file for restricted videos
        self.ydl_opts['cookiefile'] = 'cookies.txt'  # Ensure this file is exported from your browser


    def process_url(self, url: str) -> str:
        """Download audio from YouTube URL and return path to audio file"""
        # Set output template
        self.ydl_opts['outtmpl'] = str(self.download_dir / '%(title)s.%(ext)s')
        self.ydl_opts['progress_hooks'] = [self._download_hook]

        try:
            # Download the video
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                print(f"\nFetching video info from YouTube...")
                info = ydl.extract_info(url, download=True)

                # Get the path of downloaded audio file
                output_path = str(self.download_dir / f"{info['title']}.mp3")

                # Verify file exists and has content
                if not os.path.exists(output_path):
                    raise FileNotFoundError(f"Download failed: File not found at {output_path}")

                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    raise RuntimeError(f"Download failed: File is empty at {output_path}")

                print(f"\nDownload completed successfully!")
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                return output_path

        except Exception as e:
            print(f"\nError during download: {str(e)}")
            print(f"URL attempted: {url}")
            print(f"Download directory: {self.download_dir}")
            raise RuntimeError(f"Failed to download YouTube video: {str(e)}")

    def _download_hook(self, d):
        """Progress hook for yt-dlp"""
        if d['status'] == 'downloading':
            try:
                percent = d['_percent_str']
                speed = d.get('_speed_str', 'N/A')
                print(f"\rDownloading... {percent} at {speed}", end='', flush=True)
            except KeyError:
                pass
        elif d['status'] == 'finished':
            print("\nDownload finished, converting to MP3...")

def analyze_youtube_podcast(youtube_url: str, analyzer) -> list:
    """Download and analyze a YouTube podcast"""
    processor = YouTubeProcessor()
    print("Downloading YouTube audio...")
    audio_path = processor.process_url(youtube_url)
    print(f"Audio downloaded to: {audio_path}")

    print("Analyzing audio content...")
    segments = analyzer.analyze_audio(audio_path, is_url=False)
    return segments
