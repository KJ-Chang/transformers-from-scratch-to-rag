
import os
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter


def scrape_and_chunk_website(urls: list[str], chunk_size=200, chunk_overlap=50):
    if not urls:
        raise ValueError('No URLs provided for scraping')

    passages = []

    text_splitter = \
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
            )
    
    os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT )'

    for url in urls:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            text_split = text_splitter.split_text(text)
            if text_split:
                passages.extend(text_split)
    
    # print(passages)
    return passages

# 測試功能
if __name__ == "__main__":
    urls = [
        'https://www.royaltek.com/about/whoweare/'
    ]
    passages = scrape_and_chunk_website(urls, 100, 20)
    print(f'Length of passages: {len(passages)}')
    print(f'Passages: {passages[:3]}')