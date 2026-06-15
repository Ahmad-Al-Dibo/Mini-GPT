from pathlib import Path
from urllib.parse import urlparse, unquote
import wikipediaapi


class WikipediaToMarkdown:
    def __init__(self, output_dir="dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wiki = wikipediaapi.Wikipedia(
            user_agent="NLP-V1 Dataset Builder (research project)",
            language="en"
        )

    def extract_title(self, wikipedia_url: str) -> str:
        path = urlparse(wikipedia_url).path

        if "/wiki/" not in path:
            raise ValueError("Invalid Wikipedia URL")

        title = path.split("/wiki/")[-1]
        return unquote(title)

    def clean_filename(self, filename: str) -> str:
        invalid_chars = '<>:"/\\|?*'

        for char in invalid_chars:
            filename = filename.replace(char, "_")

        return filename

    def download(self, wikipedia_url: str):
        title = self.extract_title(wikipedia_url)

        page = self.wiki.page(title)

        if not page.exists():
            raise ValueError(f"Wikipedia page not found: {title}")

        markdown = f"# {page.title}\n\n{page.text}"

        safe_name = self.clean_filename(page.title)
        output_file = self.output_dir / f"{safe_name}.md"

        if Path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            return output_file

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"Saved: {output_file}")
        print(f"Characters: {len(markdown):,}")

        return output_file


if __name__ == "__main__":
    l_link=[
        # PROGRAMMING LANGUAGES
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Java_(programming_language)",
        "https://en.wikipedia.org/wiki/JavaScript",
        "https://en.wikipedia.org/wiki/C_Sharp_(programming_language)",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        "https://en.wikipedia.org/wiki/C%2B%2B",
        "https://en.wikipedia.org/wiki/Go_(programming_language)",

        # FRAMEWORKS
        "https://en.wikipedia.org/wiki/React_(JavaScript_library)",
        "https://en.wikipedia.org/wiki/Angular_(web_framework)",
        "https://en.wikipedia.org/wiki/Vue.js",

        # DATABASES
        "https://en.wikipedia.org/wiki/MySQL",
        "https://en.wikipedia.org/wiki/PostgreSQL",
        "https://en.wikipedia.org/wiki/MongoDB",

        # CLOUD PLATFORMS
        "https://en.wikipedia.org/wiki/Amazon_Web_Services",
        "https://en.wikipedia.org/wiki/Microsoft_Azure",
        "https://en.wikipedia.org/wiki/Google_Cloud_Platform",


    ]
    scraper = WikipediaToMarkdown(output_dir="dataset")

    for link in l_link:
        try:
            scraper.download(link)
        except Exception as e:
            print(f"Error processing {link}: {e}")