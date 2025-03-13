import os
import logging
import wikipedia


def load_wikipedia_page(title: str, path: str) -> str:
    if os.path.exists(path):
        logging.info("load page content from file")

        with open(path, "r") as f:
            page_content = f.read()
    else:
        logging.info("load page content from wikipedia")

        page_content = wikipedia.page(title=title, auto_suggest=False).content

        with open(path, "w") as f:
            f.write(page_content)
    return page_content
