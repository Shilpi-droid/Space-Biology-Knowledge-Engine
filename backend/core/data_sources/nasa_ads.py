"""
NASA ADS API Integration for Space Biology Papers

This module handles data collection from NASA's Astrophysics Data System (ADS),
focusing on space biology and astrobiology research papers from the last 5 years.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a research paper from ADS."""
    bibcode: str
    title: str
    abstract: str
    authors: List[str]
    pub_date: str
    journal: str
    keywords: List[str]
    citation_count: int
    citations: List[str]  # bibcodes of citing papers
    topics: List[str]
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


class NASAADSClient:
    """Client for NASA ADS API with rate limiting and error handling."""

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.adsabs.harvard.edu/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0

        # Space biology keywords for targeted search
        self.space_biology_keywords = [
            "microgravity", "space biology", "astrobiology", "space medicine",
            "space physiology", "space radiation", "space environment",
            "microgravity effects", "space flight", "ISS", "international space station",
            "mars biology", "exobiology", "space life sciences", "gravitational biology",
            "space adaptation", "bone loss space", "muscle atrophy space",
            "cardiovascular space", "immune system space", "space psychology"
        ]

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _rate_limited_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited request to ADS API."""
        # Ensure rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    logger.warning("Rate limited, waiting 5 seconds...")
                    await asyncio.sleep(5)
                    return await self._rate_limited_request(url, params)
                else:
                    logger.error(f"ADS API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {}

    async def search_space_biology_papers(self,
                                        start_year: int = 2019,
                                        max_papers: int = 1000) -> AsyncIterator[Paper]:
        """
        Search for space biology papers from the last 5 years.

        Args:
            start_year: Starting year for search
            max_papers: Maximum number of papers to retrieve

        Yields:
            Paper objects with metadata and abstracts
        """
        current_year = datetime.now().year

        # Build search query for space biology topics
        keyword_query = " OR ".join([f'abs:"{kw}"' for kw in self.space_biology_keywords])

        # Construct query with year range and topic filters
        query = f"({keyword_query}) AND year:{start_year}-{current_year}"

        # Fields to retrieve
        fields = [
            "bibcode", "title", "abstract", "author", "pub", "keyword",
            "citation_count", "doi", "identifier", "pubdate"
        ]

        params = {
            "q": query,
            "fl": ",".join(fields),
            "rows": 50,  # ADS max per request
            "start": 0,
            "sort": "citation_count desc"  # Get most cited papers first
        }

        papers_retrieved = 0
        start_index = 0

        while papers_retrieved < max_papers:
            params["start"] = start_index

            url = f"{self.base_url}/search/query"
            response = await self._rate_limited_request(url, params)

            if not response or "response" not in response:
                break

            docs = response["response"].get("docs", [])
            if not docs:
                break

            for doc in docs:
                if papers_retrieved >= max_papers:
                    break

                paper = await self._parse_paper_doc(doc)
                if paper:
                    yield paper
                    papers_retrieved += 1

            start_index += len(docs)

            # If we got fewer docs than requested, we've reached the end
            if len(docs) < params["rows"]:
                break

        logger.info(f"Retrieved {papers_retrieved} space biology papers")

    async def _parse_paper_doc(self, doc: Dict[str, Any]) -> Optional[Paper]:
        """Parse a document from ADS API response into Paper object."""
        try:
            # Required fields
            bibcode = doc.get("bibcode", [""])[0] if isinstance(doc.get("bibcode"), list) else doc.get("bibcode", "")
            title = doc.get("title", [""])[0] if isinstance(doc.get("title"), list) else doc.get("title", "")

            if not bibcode or not title:
                return None

            # Handle abstract (can be missing)
            abstract = ""
            if "abstract" in doc and doc["abstract"]:
                abstract = doc["abstract"][0] if isinstance(doc["abstract"], list) else doc["abstract"]

            # Authors
            authors = doc.get("author", [])
            if isinstance(authors, str):
                authors = [authors]

            # Publication info
            pub_date = doc.get("pubdate", "")
            journal = doc.get("pub", "")
            if isinstance(journal, list):
                journal = journal[0] if journal else ""

            # Keywords and topics
            keywords = doc.get("keyword", [])
            if isinstance(keywords, str):
                keywords = [keywords]

            # Extract topics from keywords and title
            topics = self._extract_topics(title, abstract, keywords)

            # Citation count
            citation_count = doc.get("citation_count", 0)

            # DOI and identifiers
            doi = None
            arxiv_id = None
            identifiers = doc.get("identifier", [])
            for identifier in identifiers:
                if identifier.startswith("10."):
                    doi = identifier
                elif "arXiv:" in identifier:
                    arxiv_id = identifier

            # Get citations (separate API call if needed)
            citations = await self._get_paper_citations(bibcode)

            return Paper(
                bibcode=bibcode,
                title=title,
                abstract=abstract,
                authors=authors,
                pub_date=pub_date,
                journal=journal,
                keywords=keywords,
                citation_count=citation_count,
                citations=citations,
                topics=topics,
                doi=doi,
                arxiv_id=arxiv_id
            )

        except Exception as e:
            logger.error(f"Error parsing paper doc: {str(e)}")
            return None

    def _extract_topics(self, title: str, abstract: str, keywords: List[str]) -> List[str]:
        """Extract research topics from title, abstract, and keywords."""
        topics = set()

        # Combine text for analysis
        text = f"{title} {abstract} {' '.join(keywords)}".lower()

        # Topic mapping based on common space biology themes
        topic_patterns = {
            "bone_physiology": ["bone", "osteo", "calcium", "bone loss", "bone density"],
            "muscle_physiology": ["muscle", "atrophy", "sarcopenia", "muscle mass"],
            "cardiovascular": ["cardiovascular", "heart", "blood pressure", "circulation"],
            "immune_system": ["immune", "immunity", "antibody", "infection", "immune response"],
            "radiation": ["radiation", "cosmic ray", "solar particle", "radioprotection"],
            "microgravity": ["microgravity", "weightlessness", "zero gravity", "µg"],
            "psychology": ["psychology", "stress", "isolation", "mental health", "behavior"],
            "nutrition": ["nutrition", "diet", "food", "metabolism", "nutritional"],
            "cell_biology": ["cell", "cellular", "mitochondria", "DNA", "gene expression"],
            "plant_biology": ["plant", "botany", "crop", "agriculture", "photosynthesis"],
            "astrobiology": ["astrobiology", "extremophile", "mars", "exoplanet", "life detection"]
        }

        for topic, patterns in topic_patterns.items():
            if any(pattern in text for pattern in patterns):
                topics.add(topic)

        # Add mission-related topics
        if any(term in text for term in ["iss", "international space station"]):
            topics.add("iss_research")
        if any(term in text for term in ["mars", "martian"]):
            topics.add("mars_research")
        if any(term in text for term in ["moon", "lunar"]):
            topics.add("lunar_research")

        return list(topics)

    async def _get_paper_citations(self, bibcode: str, max_citations: int = 20) -> List[str]:
        """Get citations for a paper (bibcodes of papers that cite this one)."""
        try:
            params = {
                "q": f"citations(bibcode:{bibcode})",
                "fl": "bibcode",
                "rows": max_citations
            }

            url = f"{self.base_url}/search/query"
            response = await self._rate_limited_request(url, params)

            if "response" in response:
                docs = response["response"].get("docs", [])
                return [doc["bibcode"] for doc in docs if "bibcode" in doc]

            return []

        except Exception as e:
            logger.error(f"Error getting citations for {bibcode}: {str(e)}")
            return []

    async def get_paper_details(self, bibcode: str) -> Optional[Paper]:
        """Get detailed information for a specific paper."""
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "bibcode,title,abstract,author,pub,keyword,citation_count,doi,identifier,pubdate",
            "rows": 1
        }

        url = f"{self.base_url}/search/query"
        response = await self._rate_limited_request(url, params)

        if "response" in response:
            docs = response["response"].get("docs", [])
            if docs:
                return await self._parse_paper_doc(docs[0])

        return None


class SpaceBiologyDataCollector:
    """Main collector for space biology research data from NASA ADS."""

    def __init__(self, ads_token: str):
        self.ads_token = ads_token
        self.collected_papers = []
        self.author_network = {}
        self.citation_network = {}

    async def collect_five_year_dataset(self, max_papers: int = 1000) -> Dict[str, Any]:
        """
        Collect a comprehensive 5-year dataset of space biology research.

        Returns:
            Dictionary containing papers, authors, citations, and metadata
        """
        logger.info("Starting 5-year space biology data collection...")

        async with NASAADSClient(self.ads_token) as ads_client:
            # Collect papers from last 5 years
            current_year = datetime.now().year
            start_year = current_year - 5

            papers = []
            authors = set()
            citations = {}

            async for paper in ads_client.search_space_biology_papers(
                start_year=start_year,
                max_papers=max_papers
            ):
                papers.append(paper)
                authors.update(paper.authors)

                # Build citation network
                if paper.citations:
                    citations[paper.bibcode] = paper.citations

                if len(papers) % 50 == 0:
                    logger.info(f"Collected {len(papers)} papers...")

            # Build author collaboration network
            author_collaborations = self._build_author_network(papers)

            # Calculate additional metrics
            metrics = self._calculate_dataset_metrics(papers, authors, citations)

            dataset = {
                "papers": papers,
                "total_papers": len(papers),
                "unique_authors": list(authors),
                "total_authors": len(authors),
                "citation_network": citations,
                "author_collaborations": author_collaborations,
                "collection_date": datetime.now().isoformat(),
                "date_range": f"{start_year}-{current_year}",
                "metrics": metrics
            }

            logger.info(f"Collection complete: {len(papers)} papers, {len(authors)} authors")

            return dataset

    def _build_author_network(self, papers: List[Paper]) -> Dict[str, List[str]]:
        """Build author collaboration network from papers."""
        collaborations = {}

        for paper in papers:
            # Each author on a paper collaborates with all other authors
            for i, author1 in enumerate(paper.authors):
                if author1 not in collaborations:
                    collaborations[author1] = set()

                for j, author2 in enumerate(paper.authors):
                    if i != j:
                        collaborations[author1].add(author2)

        # Convert sets to lists for JSON serialization
        return {author: list(collabs) for author, collabs in collaborations.items()}

    def _calculate_dataset_metrics(self, papers: List[Paper], authors: set, citations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate useful metrics about the collected dataset."""
        if not papers:
            return {}

        # Topic distribution
        all_topics = []
        for paper in papers:
            all_topics.extend(paper.topics)

        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Journal distribution
        journal_counts = {}
        for paper in papers:
            if paper.journal:
                journal_counts[paper.journal] = journal_counts.get(paper.journal, 0) + 1

        # Citation statistics
        citation_counts = [paper.citation_count for paper in papers if paper.citation_count > 0]

        return {
            "topic_distribution": topic_counts,
            "top_journals": dict(sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "citation_stats": {
                "total_citations": sum(citation_counts),
                "avg_citations": sum(citation_counts) / len(citation_counts) if citation_counts else 0,
                "max_citations": max(citation_counts) if citation_counts else 0,
                "papers_with_citations": len(citation_counts)
            },
            "temporal_distribution": self._calculate_temporal_distribution(papers),
            "author_productivity": self._calculate_author_productivity(papers)
        }

    def _calculate_temporal_distribution(self, papers: List[Paper]) -> Dict[str, int]:
        """Calculate papers per year."""
        year_counts = {}
        for paper in papers:
            if paper.pub_date:
                try:
                    year = paper.pub_date[:4]  # Extract year
                    year_counts[year] = year_counts.get(year, 0) + 1
                except:
                    pass
        return year_counts

    def _calculate_author_productivity(self, papers: List[Paper]) -> Dict[str, int]:
        """Calculate number of papers per author."""
        author_counts = {}
        for paper in papers:
            for author in paper.authors:
                author_counts[author] = author_counts.get(author, 0) + 1

        # Return top 20 most productive authors
        return dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:20])


# Factory function for easy use
async def collect_space_biology_dataset(ads_token: str, max_papers: int = 1000) -> Dict[str, Any]:
    """
    Convenience function to collect space biology dataset.

    Args:
        ads_token: NASA ADS API token
        max_papers: Maximum number of papers to collect

    Returns:
        Complete dataset ready for storage and analysis
    """
    collector = SpaceBiologyDataCollector(ads_token)
    return await collector.collect_five_year_dataset(max_papers)