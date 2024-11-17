from flask import Flask, request, send_file, render_template, jsonify
import threading
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import srt
import os
import assemblyai as aai
import tempfile
from datetime import timedelta
import traceback
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')

# Set up AssemblyAI API key
aai.settings.api_key = "a87bf10740af4402a492e9e7f1f7ba8b"

# Global variables to store processing status and file paths
processing_status = "Not started"
processed_video_path = None
srt_file_path = None

def transcribe_audio(video_file):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(video_file)
    return transcript.words

def edit_subtitles(words):
    subtitles = []
    current_subtitle = []
    current_start = 0
    max_chars_per_line = 40
    max_lines = 2

    for word in words:
        current_subtitle.append(word.text)
        subtitle_text = ' '.join(current_subtitle)
        
        if len(subtitle_text) > max_chars_per_line * max_lines or word.text.endswith(('.', '!', '?')):
            # Split into multiple lines if necessary
            lines = []
            current_line = []
            for w in current_subtitle:
                if len(' '.join(current_line + [w])) > max_chars_per_line:
                    lines.append(' '.join(current_line))
                    current_line = [w]
                else:
                    current_line.append(w)
            if current_line:
                lines.append(' '.join(current_line))
            
            subtitles.append((
                current_start,
                word.end,
                '\n'.join(lines[:max_lines])
            ))
            current_subtitle = []
            current_start = word.end

    if current_subtitle:
        subtitles.append((
            current_start,
            words[-1].end,
            ' '.join(current_subtitle)
        ))

    return subtitles

def generate_srt_file(subtitles):
    srt_subtitles = []
    for i, (start, end, text) in enumerate(subtitles, start=1):
        srt_subtitles.append(
            srt.Subtitle(index=i, 
                         start=timedelta(seconds=start), 
                         end=timedelta(seconds=end), 
                         content=text)
        )
    return srt.compose(srt_subtitles)

def add_subtitles_to_video(video, subtitles):
    subtitle_clips = []

    for start, end, text in subtitles:
        subtitle_clip = (TextClip(text, fontsize=24, font='Arial', color='white', stroke_color='black', stroke_width=1, method='caption', size=(video.w, None))
                         .set_position(('center', 'bottom'))
                         .set_duration(end - start)
                         .set_start(start))
        subtitle_clips.append(subtitle_clip)

    final_video = CompositeVideoClip([video] + subtitle_clips)
    final_video = final_video.set_audio(video.audio)
    
    return final_video

def resize_video(input_path, output_path, target_height=480):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new width to maintain aspect ratio
    new_width = int((target_height / height) * width)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, target_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, target_height))
        out.write(resized_frame)
    
    cap.release()
    out.release()

def process_video(video_file_path):
    global processing_status, processed_video_path, srt_file_path
    
    try:
        # Resize the video
        processing_status = "Resizing video"
        resized_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        resize_video(video_file_path, resized_video_path)
        
        # Load the resized video
        video = VideoFileClip(resized_video_path)
        
        processing_status = "Transcribing audio"
        words = transcribe_audio(video_file_path)
        
        processing_status = "Generating subtitles"
        subtitles = edit_subtitles(words)
        
        processing_status = "Creating SRT file"
        srt_content = generate_srt_file(subtitles)
        srt_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt')
        srt_file.write(srt_content)
        srt_file.close()
        srt_file_path = srt_file.name
        
        processing_status = "Adding subtitles to video"
        final_video = add_subtitles_to_video(video, subtitles)
        
        processing_status = "Saving final video"
        output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        final_video.write_videofile(output_file.name, codec='libx264', audio_codec='aac', threads=4, fps=24)
        processed_video_path = output_file.name
        
        os.unlink(video_file_path)
        processing_status = "Complete"
    except Exception as e:
        processing_status = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(processing_status)  # Print the error to the console

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global processing_status
    processing_status = "Processing started"
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    video_file = request.files['video']
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_video.name)
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_video, args=(temp_video.name,))
    thread.start()

    return jsonify({"message": "Processing started"}), 202

@app.route('/status')
def get_status():
    global processing_status
    return jsonify({"status": processing_status})

@app.route('/download_video')
def download_video():
    global processed_video_path
    if processed_video_path and os.path.exists(processed_video_path):
        return send_file(processed_video_path, as_attachment=True, download_name='video_with_subtitles.mp4')
    else:
        return "Video processing not complete or file not found", 404

@app.route('/download_srt')
def download_srt():
    global srt_file_path
    if srt_file_path and os.path.exists(srt_file_path):
        return send_file(srt_file_path, as_attachment=True, download_name='subtitles.srt')
    else:
        return "SRT file not found", 404

if __name__ == '__main__':
    app.run(debug=True)