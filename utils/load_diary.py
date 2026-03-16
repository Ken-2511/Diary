# Load diary entry as a structured dict with optional image support

import os
import base64
import glob
from datetime import datetime

__all__ = ['load_diary_entry', 'entry_to_content']

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
}


def load_diary_entry(diary_dir: str, dir_name: str, config: dict) -> dict:
    """Load a diary entry from a directory into a structured dict.

    Returns:
        {
            "dir_name": str,
            "timestamp": datetime,
            "text": str | None,
            "title": str | None,
            "comment": str | None,
            "images": [{"path": str, "base64": str}],  # base64 is a data URL
        }
    """
    entry_dir = os.path.join(diary_dir, dir_name)
    timestamp = datetime.strptime(dir_name, '%Y-%m-%d-%H-%M-%S')

    # text
    diary_path = os.path.join(entry_dir, config['diary_name'])
    text = None
    if os.path.exists(diary_path):
        with open(diary_path, 'r', encoding='utf-8') as f:
            text = f.read()

    # title
    title_path = os.path.join(entry_dir, config['title_name'])
    title = None
    if os.path.exists(title_path):
        with open(title_path, 'r', encoding='utf-8') as f:
            title = f.read()

    # comment
    comment_path = os.path.join(entry_dir, config['comment_name'])
    comment = None
    if os.path.exists(comment_path):
        with open(comment_path, 'r', encoding='utf-8') as f:
            comment = f.read()

    # images
    images = _load_images(entry_dir)

    return {
        "dir_name": dir_name,
        "timestamp": timestamp,
        "text": text,
        "title": title,
        "comment": comment,
        "images": images,
    }


def _load_images(entry_dir: str) -> list[dict]:
    """Scan directory for .jpg/.png files and encode them as base64 data URLs."""
    images = []
    for fname in sorted(os.listdir(entry_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            fpath = os.path.join(entry_dir, fname)
            mime = MIME_TYPES[ext]
            with open(fpath, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            images.append({
                "path": fpath,
                "base64": f"data:{mime};base64,{b64}",
            })
    return images


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '...[truncated]' if needed."""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + '...[truncated]'
    return text


def entry_to_content(entry: dict, include_images: bool = True, max_chars: int = 0) -> str | list:
    """Convert a diary entry dict to Gemini message content.

    Args:
        entry: Diary entry dict from load_diary_entry.
        include_images: Whether to include images in the content.
        max_chars: Max characters for diary text. 0 means no limit.

    If include_images is False or there are no images, returns a plain string.
    Otherwise returns a list of content parts (text + inline_data) for multimodal input.
    """
    diary_text = _truncate(entry['text'], max_chars) if entry['text'] else ''
    text = f"(Datetime: {entry['timestamp']})\n\n{diary_text}"

    if not include_images or not entry['images']:
        return text

    parts = [text]
    for img in entry['images']:
        # Extract base64 data and mime type from data URL
        data_url = img['base64']
        header, b64_data = data_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        
        parts.append({
            "mime_type": mime_type,
            "data": base64.b64decode(b64_data)
        })
    return parts
