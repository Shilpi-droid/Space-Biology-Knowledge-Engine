"""
External data source integration services for the Space Biology Knowledge Engine.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
import requests
from xml.etree import ElementTree as ET

from config.settings import get_settings, get_data_paths
from utils.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


class BaseDataService:
    """Base class for data ingestion services."""

    def __init__(self):
        self.settings = get_settings()
        self.data_paths = get_data_paths()
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Space-Biology-Knowledge-Engine/1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _make_request(self, url: str, params: Optional[Dict] = None,
                          method: str = 'GET') -> Dict[str, Any]:
        """
        Make rate-limited API request.

        Args:
            url: Request URL
            params: Query parameters
            method: HTTP method

        Returns:
            JSON response data

        Raises:
            DataIngestionError: If request fails
        """
        if not self.session:
            raise DataIngestionError("Session not initialized. Use async context manager.")

        # Rate limiting
        await asyncio.sleep(self.settings.api_delay)

        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.upper() == 'POST':
                async with self.session.post(url, json=params) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise DataIngestionError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {url}, Error: {e}")
            raise DataIngestionError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise DataIngestionError(f"Unexpected error: {e}")

    def save_raw_data(self, data: Any, filename: str, subdir: str = "") -> Path:
        """
        Save raw data to file.

        Args:
            data: Data to save
            filename: Output filename
            subdir: Subdirectory within raw data directory

        Returns:
            Path to saved file
        """
        if subdir:
            save_dir = self.data_paths["raw"] / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.data_paths["raw"]

        filepath = save_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))

            logger.info(f"Saved raw data to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            raise DataIngestionError(f"Failed to save data: {e}")


class OSDrawDataService(BaseDataService):
    """
    Service for ingesting data from NASA's Open Science Data Repository (OSDR).
    """

    def __init__(self):
        super().__init__()
        self.base_url = self.settings.osdr_base_url

    async def fetch_studies_metadata(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get study metadata from OSDR API.

        Args:
            limit: Maximum number of studies to fetch

        Returns:
            List of study metadata dictionaries
        """
        logger.info(f"Fetching OSDR studies metadata (limit: {limit})")

        try:
            # OSDR API endpoint for studies
            url = f"{self.base_url}/studies"
            params = {
                "size": limit,
                "from": 0,
                "sort": "releaseDate:desc"
            }

            response_data = await self._make_request(url, params)

            # Extract studies from response
            studies = response_data.get('studies', [])
            logger.info(f"Retrieved {len(studies)} studies from OSDR")

            # Save raw response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_raw_data(
                response_data,
                f"osdr_studies_{timestamp}.json",
                "osdr"
            )

            return studies

        except Exception as e:
            logger.error(f"Failed to fetch OSDR studies: {e}")
            raise DataIngestionError(f"OSDR studies fetch failed: {e}")

    async def fetch_study_details(self, study_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific study.

        Args:
            study_id: OSDR study ID

        Returns:
            Detailed study information
        """
        logger.debug(f"Fetching details for OSDR study: {study_id}")

        try:
            url = f"{self.base_url}/studies/{study_id}"
            study_data = await self._make_request(url)

            # Add fetch metadata
            study_data['_fetch_metadata'] = {
                'source': 'osdr',
                'fetch_timestamp': datetime.now().isoformat(),
                'study_id': study_id
            }

            return study_data

        except Exception as e:
            logger.error(f"Failed to fetch OSDR study {study_id}: {e}")
            raise DataIngestionError(f"OSDR study fetch failed: {e}")

    async def download_datasets(self, study_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Download dataset metadata for a study.

        Args:
            study_id: OSDR study ID
            limit: Maximum number of datasets to process

        Returns:
            List of dataset metadata
        """
        logger.info(f"Downloading datasets for study: {study_id}")

        try:
            url = f"{self.base_url}/studies/{study_id}/datasets"
            response_data = await self._make_request(url)

            datasets = response_data.get('datasets', [])[:limit]

            # Save dataset metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_raw_data(
                datasets,
                f"osdr_datasets_{study_id}_{timestamp}.json",
                "osdr/datasets"
            )

            return datasets

        except Exception as e:
            logger.error(f"Failed to download datasets for {study_id}: {e}")
            raise DataIngestionError(f"Dataset download failed: {e}")


class PMCDataService(BaseDataService):
    """
    Service for ingesting data from PubMed Central (PMC).
    """

    def __init__(self):
        super().__init__()
        self.base_url = self.settings.pmc_base_url

    async def search_publications(self, search_terms: str = "space biology",
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search PubMed Central for space biology publications.

        Args:
            search_terms: Search query
            limit: Maximum number of publications to fetch

        Returns:
            List of publication metadata
        """
        logger.info(f"Searching PMC for: {search_terms} (limit: {limit})")

        try:
            # Build search query
            query = f"({search_terms}) AND (space biology OR microgravity OR spaceflight OR ISS OR astronaut)"

            # Search for PMC IDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                'db': 'pmc',
                'term': query,
                'retmode': 'json',
                'retmax': limit,
                'sort': 'relevance',
                'field': 'title,abstract'
            }

            search_results = await self._make_request(search_url, search_params)
            pmc_ids = search_results.get('esearchresult', {}).get('idlist', [])

            if not pmc_ids:
                logger.warning("No publications found in PMC search")
                return []

            logger.info(f"Found {len(pmc_ids)} publications, fetching details...")

            # Fetch detailed information for each publication
            publications = []
            for pmc_id in pmc_ids[:limit]:
                try:
                    pub_details = await self.fetch_publication_details(pmc_id)
                    if pub_details:
                        publications.append(pub_details)
                except Exception as e:
                    logger.warning(f"Failed to fetch details for PMC ID {pmc_id}: {e}")

            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_raw_data(
                {
                    'search_query': query,
                    'total_found': len(pmc_ids),
                    'publications': publications
                },
                f"pmc_publications_{timestamp}.json",
                "pmc"
            )

            return publications

        except Exception as e:
            logger.error(f"PMC search failed: {e}")
            raise DataIngestionError(f"PMC search failed: {e}")

    async def fetch_publication_details(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific publication.

        Args:
            pmc_id: PubMed Central ID

        Returns:
            Publication details or None if fetch fails
        """
        try:
            # Fetch publication summary
            summary_url = f"{self.base_url}/esummary.fcgi"
            summary_params = {
                'db': 'pmc',
                'id': pmc_id,
                'retmode': 'json'
            }

            summary_data = await self._make_request(summary_url, summary_params)
            pub_summary = summary_data.get('result', {}).get(pmc_id, {})

            if not pub_summary:
                return None

            # Extract relevant information
            publication = {
                'pmc_id': pmc_id,
                'title': pub_summary.get('title', ''),
                'authors': pub_summary.get('authors', []),
                'journal': pub_summary.get('source', ''),
                'publication_date': pub_summary.get('pubdate', ''),
                'doi': pub_summary.get('doi', ''),
                'pmid': pub_summary.get('pmid', ''),
                'source': 'pmc',
                'url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/",
                '_fetch_metadata': {
                    'fetch_timestamp': datetime.now().isoformat(),
                    'pmc_id': pmc_id
                }
            }

            return publication

        except Exception as e:
            logger.error(f"Failed to fetch PMC publication {pmc_id}: {e}")
            return None

    async def fetch_full_text(self, pmc_id: str) -> Optional[str]:
        """
        Fetch full text content for a publication.

        Args:
            pmc_id: PubMed Central ID

        Returns:
            Full text content or None if unavailable
        """
        try:
            # Fetch full text XML
            fulltext_url = f"{self.base_url}/efetch.fcgi"
            fulltext_params = {
                'db': 'pmc',
                'id': pmc_id,
                'retmode': 'xml'
            }

            if not self.session:
                raise DataIngestionError("Session not initialized")

            async with self.session.get(fulltext_url, params=fulltext_params) as response:
                response.raise_for_status()
                xml_content = await response.text()

            # Parse XML and extract text
            try:
                root = ET.fromstring(xml_content)
                # Extract text from various XML elements
                text_elements = root.findall(".//p") + root.findall(".//abstract")
                full_text = "\n".join([elem.text or "" for elem in text_elements if elem.text])

                return full_text

            except ET.ParseError as e:
                logger.error(f"Failed to parse XML for PMC {pmc_id}: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch full text for PMC {pmc_id}: {e}")
            return None


class NTRSDataService(BaseDataService):
    """
    Service for ingesting data from NASA Technical Reports Server (NTRS).
    """

    def __init__(self):
        super().__init__()
        self.base_url = self.settings.ntrs_base_url

    async def search_technical_reports(self, search_terms: str = "space biology",
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search NASA Technical Reports for space biology content.

        Args:
            search_terms: Search query
            limit: Maximum number of reports to fetch

        Returns:
            List of technical report metadata
        """
        logger.info(f"Searching NTRS for: {search_terms} (limit: {limit})")

        try:
            # Build search query for NTRS API
            search_url = f"{self.base_url}/search"
            search_params = {
                'q': f"{search_terms} space biology",
                'rows': limit,
                'sort': 'publishedDate desc',
                'fl': 'id,title,abstract,author,publishedDate,subjectCategory,keyword'
            }

            search_results = await self._make_request(search_url, search_params)
            reports = search_results.get('docs', [])

            logger.info(f"Found {len(reports)} technical reports")

            # Process and enrich report data
            processed_reports = []
            for report in reports:
                processed_report = {
                    'ntrs_id': report.get('id', ''),
                    'title': report.get('title', ''),
                    'abstract': report.get('abstract', ''),
                    'authors': report.get('author', []),
                    'publication_date': report.get('publishedDate', ''),
                    'subject_categories': report.get('subjectCategory', []),
                    'keywords': report.get('keyword', []),
                    'source': 'ntrs',
                    'url': f"https://ntrs.nasa.gov/citations/{report.get('id', '')}",
                    '_fetch_metadata': {
                        'fetch_timestamp': datetime.now().isoformat(),
                        'search_terms': search_terms
                    }
                }
                processed_reports.append(processed_report)

            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_raw_data(
                {
                    'search_query': search_terms,
                    'total_found': len(reports),
                    'reports': processed_reports
                },
                f"ntrs_reports_{timestamp}.json",
                "ntrs"
            )

            return processed_reports

        except Exception as e:
            logger.error(f"NTRS search failed: {e}")
            raise DataIngestionError(f"NTRS search failed: {e}")

    async def fetch_report_details(self, ntrs_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific technical report.

        Args:
            ntrs_id: NTRS document ID

        Returns:
            Detailed report information or None if fetch fails
        """
        try:
            url = f"{self.base_url}/citations/{ntrs_id}"
            report_data = await self._make_request(url)

            # Add fetch metadata
            report_data['_fetch_metadata'] = {
                'source': 'ntrs',
                'fetch_timestamp': datetime.now().isoformat(),
                'ntrs_id': ntrs_id
            }

            return report_data

        except Exception as e:
            logger.error(f"Failed to fetch NTRS report {ntrs_id}: {e}")
            return None


class DataIngestionOrchestrator:
    """
    Orchestrates data ingestion from multiple sources.
    """

    def __init__(self):
        self.settings = get_settings()
        self.data_paths = get_data_paths()

    async def ingest_all_sources(self, limits: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Ingest data from all configured sources.

        Args:
            limits: Per-source limits for data ingestion

        Returns:
            Summary of ingestion results
        """
        if limits is None:
            limits = {'osdr': 50, 'pmc': 100, 'ntrs': 50}

        results = {
            'started_at': datetime.now(),
            'sources': {},
            'total_items': 0,
            'errors': []
        }

        logger.info("Starting data ingestion from all sources")

        # Ingest from OSDR
        try:
            async with OSDrawDataService() as osdr_service:
                osdr_studies = await osdr_service.fetch_studies_metadata(limits.get('osdr', 50))
                results['sources']['osdr'] = {
                    'items_count': len(osdr_studies),
                    'status': 'success'
                }
                results['total_items'] += len(osdr_studies)
        except Exception as e:
            error_msg = f"OSDR ingestion failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['sources']['osdr'] = {'status': 'failed', 'error': str(e)}

        # Ingest from PMC
        try:
            async with PMCDataService() as pmc_service:
                pmc_publications = await pmc_service.search_publications(limit=limits.get('pmc', 100))
                results['sources']['pmc'] = {
                    'items_count': len(pmc_publications),
                    'status': 'success'
                }
                results['total_items'] += len(pmc_publications)
        except Exception as e:
            error_msg = f"PMC ingestion failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['sources']['pmc'] = {'status': 'failed', 'error': str(e)}

        # Ingest from NTRS
        try:
            async with NTRSDataService() as ntrs_service:
                ntrs_reports = await ntrs_service.search_technical_reports(limit=limits.get('ntrs', 50))
                results['sources']['ntrs'] = {
                    'items_count': len(ntrs_reports),
                    'status': 'success'
                }
                results['total_items'] += len(ntrs_reports)
        except Exception as e:
            error_msg = f"NTRS ingestion failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['sources']['ntrs'] = {'status': 'failed', 'error': str(e)}

        results['completed_at'] = datetime.now()
        results['duration'] = (results['completed_at'] - results['started_at']).total_seconds()

        logger.info(f"Data ingestion completed. Total items: {results['total_items']}")

        # Save ingestion summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.data_paths["logs"] / f"ingestion_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results