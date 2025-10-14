import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
import re
from typing import List, Dict, Tuple

# --- UTILITY FUNCTIONS ---

def is_in_domain(url: str, base_netloc: str) -> bool:
    """Checks if a URL belongs to the starting domain."""
    try:
        netloc = urlparse(url).netloc
        # Checks if domain is the same, or if it's a subdomain 
        return netloc == base_netloc or netloc.endswith('.' + base_netloc)
    except ValueError:
        return False

def extract_text(html_content: str) -> str:
    """Extracts clean text from HTML content and removes boilerplate."""
    # 3.1 Initialize BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 3.2 Remove common noise elements (scripts, styles, nav, headers, etc.)
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()
        
    # 3.3 Get text and normalize whitespace (removes extra newlines and spaces)
    text = soup.get_text(separator=' ', strip=True)
    
    # 3.4 Optional: Basic cleanup of non-essential characters
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- CORE CRAWLER FUNCTION ---

def crawl_site(start_url: str, max_pages: int, crawl_delay_ms: int) -> Tuple[List[Dict], List[str]]:
    """Crawls pages up to max_pages within the starting domain."""
    
    base_netloc = urlparse(start_url).netloc
    
    # Use a set to track visited URLs (for speed) and a list for the queue (for BFS order)
    queue = [(start_url, 0)] # (url, depth)
    visited = {start_url}
    
    crawled_data = [] # Stores the final cleaned page content and URL
    skipped_urls = []
    page_count = 0
    crawl_delay_s = crawl_delay_ms / 1000 # Convert ms to seconds

    # NOTE: Full robots.txt parsing is skipped for timebox constraints.
    
    while queue and page_count < max_pages:
        current_url, depth = queue.pop(0)
        time.sleep(crawl_delay_s) # Apply politeness delay
        
        try:
            # 4.3 Fetch page content
            # Set a timeout and check for status codes
            response = requests.get(current_url, timeout=15)
            response.raise_for_status() 
            
            # [cite_start]Ensure it's HTML text before proceeding (Konduit constraint [cite: 17])
            if 'text/html' not in response.headers.get('Content-Type', ''):
                skipped_urls.append(current_url); 
                continue
                
            # 4.4 Extract and store data
            cleaned_text = extract_text(response.text)
            crawled_data.append({
                "url": current_url,
                "text": cleaned_text
            })
            page_count += 1
            
            # 4.5 Find new links and add to queue
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                # Normalize URL: combines relative URL with base, and removes fragments (#section)
                new_url, _ = urldefrag(urljoin(current_url, link['href'])) 
                
                if new_url not in visited and is_in_domain(new_url, base_netloc):
                    visited.add(new_url)
                    queue.append((new_url, depth + 1))
        
        except requests.RequestException as e:
            # Catch network errors, timeouts, or 4xx/5xx responses
            skipped_urls.append(current_url)
            continue 

    return crawled_data, skipped_urls