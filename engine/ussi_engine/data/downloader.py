"""Dataset downloader: fetch public domain training data.

Supports:
    - Project Gutenberg (public domain books)
    - Wikipedia dumps (plain text)
    - Sample datasets for quick testing

All data is public domain or permissively licensed. No copyrighted material.
Data stays local -- only gradients leave the peer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default data directory.
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".ussi", "data")

# Public domain sample texts for quick testing.
SAMPLE_TEXTS = {
    "alice": (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her "
        "sister was reading, but it had no pictures or conversations in it, and what "
        "is the use of a book, thought Alice without pictures or conversations? So "
        "she was considering in her own mind as well as she could, for the hot day "
        "made her feel very sleepy and stupid, whether the pleasure of making a "
        "daisy chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her.\n\n"
        "There was nothing so very remarkable in that; nor did Alice think it so very "
        "much out of the way to hear the Rabbit say to itself Oh dear Oh dear I "
        "shall be late when she thought it over afterwards it occurred to her that "
        "she ought to have wondered at this but at the time it all seemed quite "
        "natural but when the Rabbit actually took a watch out of its waistcoat "
        "pocket and looked at it and then hurried on Alice started to her feet "
        "for it flashed across her mind that she had never before seen a rabbit "
        "with either a waistcoat pocket or a watch to take out of it and burning "
        "with curiosity she ran across the field after it and fortunately was just "
        "in time to see it pop down a large rabbit hole under the hedge.\n\n"
    ),
    "shakespeare": (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles, "
        "And by opposing end them. To die: to sleep; "
        "No more; and by a sleep to say we end "
        "The heart-ache and the thousand natural shocks "
        "That flesh is heir to, 'tis a consummation "
        "Devoutly to be wish'd. To die, to sleep; "
        "To sleep: perchance to dream: ay, there's the rub; "
        "For in that sleep of death what dreams may come "
        "When we have shuffled off this mortal coil.\n\n"
        "All the world's a stage, and all the men and women merely players. "
        "They have their exits and their entrances, and one man in his time "
        "plays many parts, his acts being seven ages. At first the infant, "
        "mewling and puking in the nurse's arms. Then the whining schoolboy, "
        "with his satchel and shining morning face, creeping like snail "
        "unwillingly to school.\n\n"
    ),
    "philosophy": (
        "I think, therefore I am. The only thing I know is that I know nothing. "
        "The unexamined life is not worth living. We are what we repeatedly do. "
        "Excellence, then, is not an act, but a habit. Man is by nature a political "
        "animal. The life of money making is one undertaken under compulsion, and "
        "wealth is evidently not the good we are seeking, for it is merely useful "
        "and for the sake of something else. Happiness is the meaning and the "
        "purpose of life, the whole aim and end of human existence.\n\n"
        "Two things fill the mind with ever new and increasing admiration and awe, "
        "the more often and the more steadily we reflect on them: the starry heavens "
        "above me and the moral law within me. Dare to know! Have the courage to use "
        "your own understanding. Act only according to that maxim whereby you can at "
        "the same time will that it should become a universal law.\n\n"
    ),
    "science": (
        "If I have seen further it is by standing on the shoulders of giants. "
        "The most beautiful thing we can experience is the mysterious. It is the "
        "source of all true art and science. Imagination is more important than "
        "knowledge. Knowledge is limited. Imagination encircles the world. "
        "Nothing in life is to be feared, it is only to be understood. Now is "
        "the time to understand more, so that we may fear less.\n\n"
        "The important thing is not to stop questioning. Curiosity has its own "
        "reason for existing. One cannot help but be in awe when one contemplates "
        "the mysteries of eternity, of life, of the marvelous structure of reality. "
        "It is enough if one tries merely to comprehend a little of this mystery "
        "every day.\n\n"
    ),
}

# Project Gutenberg mirror URLs for popular public domain books.
GUTENBERG_BOOKS = {
    "alice_in_wonderland": {
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "title": "Alice's Adventures in Wonderland",
        "size_kb": 170,
    },
    "pride_and_prejudice": {
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "title": "Pride and Prejudice",
        "size_kb": 710,
    },
    "moby_dick": {
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "title": "Moby Dick",
        "size_kb": 1260,
    },
    "tale_of_two_cities": {
        "url": "https://www.gutenberg.org/files/98/98-0.txt",
        "title": "A Tale of Two Cities",
        "size_kb": 790,
    },
    "great_expectations": {
        "url": "https://www.gutenberg.org/files/1400/1400-0.txt",
        "title": "Great Expectations",
        "size_kb": 1030,
    },
    "frankenstein": {
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "title": "Frankenstein",
        "size_kb": 450,
    },
    "dracula": {
        "url": "https://www.gutenberg.org/files/345/345-0.txt",
        "title": "Dracula",
        "size_kb": 870,
    },
    "war_and_peace": {
        "url": "https://www.gutenberg.org/files/2600/2600-0.txt",
        "title": "War and Peace",
        "size_kb": 3290,
    },
    "sherlock_holmes": {
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "title": "The Adventures of Sherlock Holmes",
        "size_kb": 580,
    },
    "don_quixote": {
        "url": "https://www.gutenberg.org/files/996/996-0.txt",
        "title": "Don Quixote",
        "size_kb": 2380,
    },
}


def get_data_dir() -> str:
    """Get or create the default data directory."""
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    return DEFAULT_DATA_DIR


def get_sample_text(name: str = "all") -> str:
    """Get built-in sample text for quick testing.

    Args:
        name: One of 'alice', 'shakespeare', 'philosophy', 'science', 'all'.

    Returns:
        Text string suitable for training.
    """
    if name == "all":
        return "\n".join(SAMPLE_TEXTS.values()) * 5  # Repeat for more data.
    return SAMPLE_TEXTS.get(name, SAMPLE_TEXTS["alice"]) * 10


def download_gutenberg(
    books: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> List[str]:
    """Download public domain books from Project Gutenberg.

    Args:
        books: List of book keys (e.g. ['alice_in_wonderland', 'moby_dick']).
               If None, downloads all books.
        data_dir: Directory to save files to.
        progress_callback: Called with (book_key, status, bytes_downloaded).

    Returns:
        List of downloaded file paths.
    """
    if books is None:
        books = list(GUTENBERG_BOOKS.keys())

    data_dir = data_dir or os.path.join(get_data_dir(), "gutenberg")
    os.makedirs(data_dir, exist_ok=True)

    downloaded = []
    for key in books:
        if key not in GUTENBERG_BOOKS:
            logger.warning("Unknown book: %s", key)
            continue

        info = GUTENBERG_BOOKS[key]
        filepath = os.path.join(data_dir, f"{key}.txt")

        if os.path.exists(filepath):
            logger.info("Already downloaded: %s", info["title"])
            downloaded.append(filepath)
            if progress_callback:
                progress_callback(key, "cached", os.path.getsize(filepath))
            continue

        logger.info("Downloading: %s (~%d KB)", info["title"], info["size_kb"])
        if progress_callback:
            progress_callback(key, "downloading", 0)

        try:
            req = urllib.request.Request(
                info["url"],
                headers={"User-Agent": "USSI/1.0 (decentralized-llm-training)"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()

            with open(filepath, "wb") as f:
                f.write(content)

            downloaded.append(filepath)
            logger.info("Downloaded: %s (%d bytes)", info["title"], len(content))
            if progress_callback:
                progress_callback(key, "complete", len(content))

        except (urllib.error.URLError, OSError) as e:
            logger.warning("Failed to download %s: %s", info["title"], e)
            if progress_callback:
                progress_callback(key, "failed", 0)

    return downloaded


def list_gutenberg() -> List[dict]:
    """List available Gutenberg books with download status."""
    data_dir = os.path.join(get_data_dir(), "gutenberg")
    result = []
    for key, info in GUTENBERG_BOOKS.items():
        filepath = os.path.join(data_dir, f"{key}.txt")
        result.append({
            "key": key,
            "title": info["title"],
            "size_kb": info["size_kb"],
            "downloaded": os.path.exists(filepath),
            "path": filepath if os.path.exists(filepath) else None,
        })
    return result


def get_local_data_paths(data_dir: Optional[str] = None) -> List[str]:
    """Get all text files in the data directory."""
    data_dir = data_dir or get_data_dir()
    paths = []
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.endswith((".txt", ".md", ".py", ".json")):
                paths.append(os.path.join(root, fname))
    return paths
