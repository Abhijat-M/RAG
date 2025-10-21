import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Set
from urllib.parse import urljoin, urlparse
import time
import re
from logger import logger
from config import Config

class WebCrawler:
    """Web crawler for extracting information from websites"""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _fetch_and_parse_page(self, url: str, context: str = "") -> Dict[str, Any]:
        """Fetch and parse a single page. Returns processed data or raises an error."""
        response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '') if meta_description else ''
        
        if context:
            relevant_content = self.filter_by_context(text, context)
        else:
            relevant_content = text
        
        return {
            'content': relevant_content,
            'metadata': {
                'url': url,
                'title': title_text,
                'description': description,
                'source': 'web_crawl',
                'context': context
            }
        }
    
    def _start_crawl(self, start_url: str, context: str, max_pages: int, max_depth: int) -> List[Dict[str, Any]]:
        """Starts a new crawl session for a given URL."""
        self.visited_urls.clear()
        crawled_data: List[Dict[str, Any]] = []
        base_domain = urlparse(start_url).netloc
        
        self._crawl_recursive(start_url, base_domain, context, max_pages, max_depth, 0, crawled_data)
        
        return crawled_data

    def _crawl_recursive(self, url: str, base_domain: str, context: str, max_pages: int, max_depth: int, current_depth: int, crawled_data: List[Dict[str, Any]]):
        """Recursive helper function for crawling."""
        
        parsed_url = urlparse(url)
        if (url in self.visited_urls or 
            len(crawled_data) >= max_pages or 
            current_depth > max_depth or
            not self.is_valid_url(url) or 
            parsed_url.netloc != base_domain):
            return

        try:
            self.visited_urls.add(url)
            logger.info(f"Crawling (Depth {current_depth}): {url}")
            page_data = self._fetch_and_parse_page(url, context)
            
            if page_data and page_data['content']:
                crawled_data.append(page_data)
            
            if current_depth < max_depth and len(crawled_data) < max_pages:
                links = self._extract_links_from_page(url)
                
                for link in links:
                    self._crawl_recursive(link, base_domain, context, max_pages, max_depth, current_depth + 1, crawled_data)
                    if len(crawled_data) >= max_pages:
                        break
                                
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")

    def crawl_root_urls(self, urls: List[str], context: str, max_pages_per_url: int, max_depth: int) -> List[Dict[str, Any]]:
        """Crawl multiple root URLs, applying limits to each."""
        all_content = []
        
        for i, url in enumerate(urls):
            if not self.is_valid_url(url):
                logger.warning(f"Skipping invalid URL: {url}")
                continue
                
            logger.info(f"Starting crawl {i+1}/{len(urls)} for root URL: {url}")
            try:
                content = self._start_crawl(url, context, max_pages_per_url, max_depth)
                all_content.extend(content)
            except Exception as e:
                logger.error(f"Error during crawl for {url}: {e}")
            
            time.sleep(0.5) # Be respectful
        
        logger.info(f"Crawl complete. Fetched {len(all_content)} total pages.")
        return all_content
    
    def filter_by_context(self, text: str, context: str) -> str:
        """Filter content based on context keywords"""
        context_keywords = [k.strip() for k in context.lower().split(',') if k.strip()]
        if not context_keywords:
            return text

        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in context_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences)
        else:
            return ""
    
    def _extract_links_from_page(self, url: str) -> List[str]:
        """Extract all links from a webpage"""
        try:
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href']).split('#')[0]
                if self.is_valid_url(full_url):
                    links.append(full_url)
            return list(set(links))
        except Exception as e:
            logger.warning(f"Error extracting links from {url}: {e}")
            return []
    
    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def get_page_summary(self, url: str) -> Dict[str, Any]:
        """Get a quick summary of a webpage"""
        try:
            response = self.session.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            first_p = soup.find('p')
            first_paragraph = first_p.get_text().strip() if first_p else ''
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'preview': first_paragraph[:200] + "..."
            }
        except Exception as e:
            return {'url': url, 'title': 'Error loading page', 'description': str(e), 'preview': ''}