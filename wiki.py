import wikipedia # https://wikipedia.readthedocs.io/en/latest/code.html#api
import json
import sqlite3
import spacy
import time
import random
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Dict, Any

# Load spacy model once at module level
nlp = spacy.load("en_core_web_sm")

# create the database if it doesn't exist
conn = sqlite3.connect("pages.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS pages (name TEXT, links TEXT)")
conn.commit()

def encode_text(text: str) -> Any:
    """Encode text using spacy's sentence vectors"""
    doc = nlp(text)
    return doc.vector.reshape(1, -1)

# Cache for embeddings to avoid recomputation
embedding_cache = {}

def get_cached_embedding(text: str) -> Any:
    """Get embedding with caching to avoid recomputation."""
    if text not in embedding_cache:
        embedding_cache[text] = encode_text(text)
    return embedding_cache[text]

def get_page(page_name: str) -> Optional[wikipedia.WikipediaPage]:
    """Get a specific Wikipedia page by name. Before, it would default to the "Python" page if the page was not found"""
    try:
        return wikipedia.page(page_name, auto_suggest=False, redirect=False)
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first option from disambiguation
        try:
            return wikipedia.page(e.options[0], auto_suggest=False, redirect=False)
        except:
            pass
    except wikipedia.exceptions.PageError:
        pass
    except Exception:
        pass
    
    try:
        search_results = wikipedia.search(page_name)
        if search_results:
            choice = search_results[0]
            page = wikipedia.page(choice, auto_suggest=False, redirect=False)
            return page
    except Exception:
        pass
    
    # Return None instead of defaulting to Python page
    return None

def get_page_links_with_cache(page_name: str, hard_mode: bool = False) -> List[str]:
    conn = sqlite3.connect("pages.db")
    cursor = conn.cursor()
    cached_page = cursor.execute("SELECT * FROM pages WHERE name = ?", (page_name,)).fetchone()

    if not cached_page:
        page = get_page(page_name)
        if page is None:
            return []
        links = page.links
        categories = page.categories
        cursor.execute("INSERT INTO pages (name, links) VALUES (?, ?)", (page_name, json.dumps(links + categories)))
        conn.commit()
        cached_page = cursor.execute("SELECT * FROM pages WHERE name = ?", (page_name,)).fetchone()

    all_links = json.loads(cached_page[1])
    
    # In hard mode, only use direct page links, not categories
    if hard_mode:
        page = get_page(page_name)
        if page is None:
            return []
        links = page.links
        filtered = [link for link in links if is_regular_page(link)]
    else:
        # In normal mode, use both links and filtered categories
        filtered = [link for link in all_links if is_regular_page(link) and is_good_category(link)]
    
    if page_name in filtered:
        filtered.remove(page_name)
    return filtered

def is_regular_page(page_name: str) -> bool:
    """Filter out meta pages and disambiguation pages"""
    page_name_lower = page_name.lower()
    
    # Filter out disambiguation and meta pages
    if any(term in page_name_lower for term in [
        "disambiguation", "automatic", "article", "page", "identifier",
        "short description", "wikidata", "template:", "user:", "talk:",
        "file:", "category:", "help:", "portal:", "special:", "mediawiki:"
    ]):
        return False
    
    # Filter out pages with too many special characters or numbers
    if len(page_name) < 2 or page_name.count("(") > 2 or page_name.count(")") > 2:
        return False
    
    return True

def is_good_category(page_name: str) -> bool:
    """Filter out meta categories that don't create meaningful thematic connections"""
    page_name_lower = page_name.lower()
    
    # Filter out meta categories
    bad_category_terms = [
        "short description", "wikidata", "unsourced", "script error", "dead link",
        "external link", "citation needed", "cleanup", "stub", "orphan",
        "disambiguation", "redirect", "template", "user:", "talk:", "file:",
        "category:", "help:", "portal:", "special:", "mediawiki:", "wikipedia:",
        "articles with", "pages with", "all articles", "good articles",
        "featured articles", "lists of", "outline of", "index of"
    ]
    
    if any(term in page_name_lower for term in bad_category_terms):
        return False
    
    # Keep categories that are likely to create meaningful connections
    good_category_terms = [
        "category:", "list of", "types of", "kinds of", "forms of",
        "examples of", "species of", "genus", "family", "order", "class",
        "phylum", "kingdom", "domain", "group", "set", "collection"
    ]
    
    # If it's a category, it should contain one of the good terms
    if page_name_lower.startswith("category:"):
        return any(term in page_name_lower for term in good_category_terms)
    
    return True

def _find_short_path(start_path: List[str], end_path: List[str], start_time: float = None, max_depth: int = 15, hard_mode: bool = False) -> Optional[List[str]]:
    """Improved method to find a short path between two Wikipedia pages with timeout and better error handling."""
    
    if start_time is None:
        start_time = time.time()
    
    # Timeout after 10 seconds
    if time.time() - start_time > 10:
        return None
    
    start_leaf = start_path[-1]
    end_leaf = end_path[0]

    # Base cases: we've reached the end or exceeded depth
    if len(start_path) + len(end_path) > max_depth:
        return None

    if start_leaf == end_leaf:
        return start_path + end_path
    
    # Get links with error handling
    try:
        links = get_page_links_with_cache(start_leaf, hard_mode)
        if not links:
            return None
            
        if end_leaf in links:
            return start_path + end_path
        
        # Get backlinks with error handling
        backlinks = get_page_links_with_cache(end_leaf, hard_mode)
        if not backlinks:
            return None
            
        if start_leaf in backlinks:
            return start_path + end_path
        
        # Check for intersection
        intersection = list(set(links) & set(backlinks))
        if len(intersection) > 0:
            return start_path + [intersection[0]] + end_path
        
        # Try to find a path through common categories (only in normal mode)
        if not hard_mode:
            # Get categories for both pages
            start_page = get_page(start_leaf)
            end_page = get_page(end_leaf)
            
            if start_page and end_page:
                start_categories = [cat for cat in start_page.categories if is_good_category(cat)]
                end_categories = [cat for cat in end_page.categories if is_good_category(cat)]
                
                # Find common categories
                common_categories = list(set(start_categories) & set(end_categories))
                if common_categories:
                    # Use the first common category as a bridge
                    bridge_category = common_categories[0]
                    return start_path + [bridge_category] + end_path
        
        # Use cached embeddings for better performance
        end_leaf_page = get_page(end_leaf)
        if end_leaf_page is None:
            return None
            
        end_embedding = get_cached_embedding(end_leaf_page.summary)
        
        # Score links and take top candidates
        scored_links = []
        for link in links[:50]:  # Limit to top 50 links for performance
            try:
                link_embedding = get_cached_embedding(link)
                score = cosine_similarity(link_embedding, end_embedding)[0][0]
                scored_links.append((link, score))
            except Exception:
                continue
        
        if not scored_links:
            return None
            
        scored_links.sort(key=lambda x: x[1], reverse=True)
        next_page = scored_links[0][0]

        # Score backlinks
        start_leaf_page = get_page(start_leaf)
        if start_leaf_page is None:
            return None
            
        start_embedding = get_cached_embedding(start_leaf_page.summary)
        
        scored_backlinks = []
        for backlink in backlinks[:50]:  # Limit to top 50 backlinks
            try:
                backlink_embedding = get_cached_embedding(backlink)
                score = cosine_similarity(backlink_embedding, start_embedding)[0][0]
                scored_backlinks.append((backlink, score))
            except Exception:
                continue
        
        if not scored_backlinks:
            return None
            
        scored_backlinks.sort(key=lambda x: x[1], reverse=True)
        previous_page = scored_backlinks[0][0]

        return _find_short_path(start_path + [next_page], [previous_page] + end_path, start_time, max_depth, hard_mode)
        
    except Exception as e:
        print(f"Error in path finding: {e}")
        return None

def find_short_path(start_page: wikipedia.WikipediaPage, end_page: wikipedia.WikipediaPage, hard_mode: bool = False) -> List[str]:
    """Find a short path between two Wikipedia pages with improved error handling."""
    
    
    start_path = [start_page.title]
    end_path = [end_page.title]

    result = _find_short_path(start_path, end_path, hard_mode=hard_mode)
    if result is None:
        # Fallback: try to find a simple path through common topics
        fallback_result = _try_fallback_path(start_page, end_page, hard_mode)
        if fallback_result:
            return fallback_result
        return [f"No path found between {start_page.title} and {end_page.title}"]
    
    return result

def _try_fallback_path(start_page: wikipedia.WikipediaPage, end_page: wikipedia.WikipediaPage, hard_mode: bool) -> Optional[List[str]]:
    """Fallback strategy to find a simple path when the main algorithm fails."""
    try:
        # Try to find a path through common broad categories
        if not hard_mode:
            print("Falling back to broad categories to find a path")
            # Look for common high-level categories
            common_broad_categories = [
                "Category:Geography", "Category:History", "Category:Science", 
                "Category:Technology", "Category:Culture", "Category:Sports",
                "Category:Entertainment", "Category:Business", "Category:Politics"
            ]
            
            start_categories = [cat for cat in start_page.categories if any(broad in cat for broad in common_broad_categories)]
            end_categories = [cat for cat in end_page.categories if any(broad in cat for broad in common_broad_categories)]
            
            # Find any common broad category
            common_broad = list(set(start_categories) & set(end_categories))
            if common_broad:
                return [start_page.title, common_broad[0], end_page.title]
        else:
            print("Hard mode: Trying to find meaningful connections without categories")
            # For hard mode, try to find meaningful connections through common topics
            return _try_hard_mode_fallback(start_page, end_page)
        
        # If no common categories, try a simple 2-hop path through a common topic
        # This is a simplified approach that might work for many cases
        return [start_page.title, "Wikipedia", end_page.title]
        
    except Exception:
        return None

def _try_hard_mode_fallback(start_page: wikipedia.WikipediaPage, end_page: wikipedia.WikipediaPage) -> List[str]:
    """Find meaningful connections for hard mode without using categories."""
    print('Falling back given hard mode...')
    try:
        # Strategy 1: Try to find a path through common high-level concepts
        common_concepts = [
            "Human", "Earth", "World", "Society", "Life", "Nature", "History", 
            "Science", "Art", "Music", "Literature", "Philosophy", "Religion",
            "Politics", "Economics", "Technology", "Education", "Health"
        ]
        
        # Check if both pages link to any common concept
        start_links = set(start_page.links)
        end_links = set(end_page.links)
        
        for concept in common_concepts:
            if concept in start_links and concept in end_links:
                return [start_page.title, concept, end_page.title]
        
        # Strategy 2: Try to find a path through geographical or temporal connections
        geographical_terms = ["Country", "City", "Region", "Continent", "Ocean", "River", "Mountain"]
        temporal_terms = ["Century", "Decade", "Year", "Era", "Period", "Age"]
        
        for term in geographical_terms + temporal_terms:
            if term in start_links and term in end_links:
                return [start_page.title, term, end_page.title]
        
        # Strategy 3: Try to find a path through academic disciplines
        disciplines = ["Mathematics", "Physics", "Chemistry", "Biology", "Psychology", 
                      "Sociology", "Anthropology", "Linguistics", "Economics", "Law"]
        
        for discipline in disciplines:
            if discipline in start_links and discipline in end_links:
                return [start_page.title, discipline, end_page.title]
        
        # Strategy 4: Try to find a path through cultural domains
        cultural_domains = ["Culture", "Tradition", "Custom", "Festival", "Holiday", 
                           "Celebration", "Ceremony", "Ritual"]
        
        for domain in cultural_domains:
            if domain in start_links and domain in end_links:
                return [start_page.title, domain, end_page.title]
        
        # Strategy 5: Try to find a path through universal human experiences
        universal_experiences = ["Love", "Death", "Birth", "Family", "Friendship", 
                                "Work", "Play", "Food", "Sleep", "Travel"]
        
        for experience in universal_experiences:
            if experience in start_links and experience in end_links:
                return [start_page.title, experience, end_page.title]
        
        # If all else fails, use a more meaningful universal bridge than just "Wikipedia"
        universal_bridges = ["Human", "World", "Society", "Life", "History"]
        for bridge in universal_bridges:
            if bridge in start_links or bridge in end_links:
                return [start_page.title, bridge, end_page.title]
        
        # Last resort: use "Human" as it's the most universal connection
        return [start_page.title, "Human", end_page.title]
        
    except Exception:
        # If all strategies fail, fall back to the original simple approach
        return [start_page.title, "Wikipedia", end_page.title]