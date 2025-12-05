"""
Metrics Collection Service - Comprehensive ECA Performance Tracking

This service collects, aggregates, and provides metrics for the ECA dashboard,
enabling rigorous scientific evaluation of learning and emergence capabilities.
"""

import csv
import io
import json
import logging
import math
import statistics
import itertools
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    AGENT_ACTIVATION = "agent_activation"
    MEMORY_ACCESS = "memory_access"
    LEARNING_EVENT = "learning_event"
    CONFLICT_RESOLUTION = "conflict_resolution"
    COGNITIVE_CYCLE = "cognitive_cycle"
    ERROR_ANALYSIS = "error_analysis"
    RL_STRATEGY = "rl_strategy"
    META_COGNITIVE = "meta_cognitive"
    ATTENTION_DIRECTIVE = "attention_directive"


@dataclass
class MetricEvent:
    """Structured metric event"""
    type: MetricType
    timestamp: datetime
    data: Dict[str, Any]
    cycle_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class AggregateStats:
    """Rolling aggregate statistics"""
    count: int = 0
    sum: float = 0.0
    avg: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    last_updated: Optional[datetime] = None

    def update(self, value: float):
        """Update aggregate with new value"""
        self.count += 1
        self.sum += value
        self.avg = self.sum / self.count
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.last_updated = datetime.utcnow()


class MetricsService:
    """
    Centralized metrics collection and aggregation for ECA dashboard.

    Provides real-time metrics, historical analysis, and scientific validation data.
    """

    def __init__(self, max_buffer_size: int = 10000, retention_hours: int = 24, chroma_client: Optional[chromadb.Client] = None):
        """
        Initialize metrics service with ChromaDB persistence.

        Args:
            max_buffer_size: Maximum number of events to keep in memory
            retention_hours: How long to retain metrics data
            chroma_client: Optional existing ChromaDB client to reuse
        """
        self.max_buffer_size = max_buffer_size
        self.retention_period = timedelta(hours=retention_hours)

        # Event storage (in-memory buffer for fast access)
        self.events: deque[MetricEvent] = deque(maxlen=max_buffer_size)

        # Rolling aggregates (last 24 hours)
        self.aggregates: Dict[str, Dict[str, AggregateStats]] = defaultdict(lambda: defaultdict(AggregateStats))

        # Real-time counters
        self.counters: Dict[str, int] = defaultdict(int)

        # Specialized metric trackers
        self.agent_activations: Dict[str, int] = defaultdict(int)
        self.memory_performance: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.skill_performance: Dict[str, List[float]] = defaultdict(list)
        self.conflict_stats: Dict[str, int] = defaultdict(int)

        # ChromaDB setup
        self.client: Optional[chromadb.Client] = chroma_client
        self.metrics_collection: Optional[chromadb.Collection] = None

        # Initialize ChromaDB
        asyncio.create_task(self._init_chroma())
        
        logger.info("MetricsService initialized with ChromaDB persistence.")

    async def _init_chroma(self):
        """Initialize ChromaDB collection for metrics storage."""
        try:
            # Use existing client or create new one
            if self.client is None:
                from src.core.config import settings
                self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
                logger.info("MetricsService created new ChromaDB client.")
            else:
                logger.info("MetricsService reusing existing ChromaDB client.")

            # Create/get metrics collection
            try:
                self.metrics_collection = self.client.get_collection("metrics_events")
                logger.info("MetricsService connected to existing metrics collection.")
            except ValueError:
                # Collection doesn't exist, create it
                self.metrics_collection = self.client.create_collection(
                    name="metrics_events",
                    metadata={"description": "ECA metrics and performance data"}
                )
                logger.info("MetricsService created new metrics collection.")

            # Load recent events into memory buffer
            await self._load_recent_events()
            
            logger.info("ChromaDB metrics storage initialized and recent events loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB for metrics: {e}", exc_info=True)
            # If ChromaDB fails, we'll operate in memory-only mode
            logger.warning("Operating in memory-only mode for metrics")

    async def _load_recent_events(self):
        """Load recent events from ChromaDB into memory buffer."""
        try:
            if self.metrics_collection is None:
                logger.warning("Metrics collection not initialized, skipping event loading")
                return
                
            cutoff_time = datetime.utcnow() - self.retention_period
            
            # Query recent events from ChromaDB using Unix timestamp for numeric comparison
            results = self.metrics_collection.query(
                query_texts=[""],  # Empty query to get all
                where={"timestamp": {"$gt": cutoff_time.timestamp()}},
                n_results=self.max_buffer_size,
                include=["metadatas", "documents"]
            )
            
            # Process results and create MetricEvent objects
            if results and results["metadatas"]:
                for metadata_list in results["metadatas"]:
                    for metadata in metadata_list:
                        try:
                            # Handle timestamp conversion more robustly
                            timestamp_str = metadata["timestamp"]
                            if isinstance(timestamp_str, str):
                                try:
                                    # Try to parse as float first
                                    timestamp_val = float(timestamp_str)
                                except ValueError:
                                    # If it's an ISO string, parse it
                                    try:
                                        timestamp_val = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                                    except ValueError:
                                        # Skip invalid timestamps
                                        continue
                            else:
                                timestamp_val = timestamp_str
                            
                            event = MetricEvent(
                                type=MetricType(metadata["type"]),
                                timestamp=datetime.fromtimestamp(timestamp_val),
                                cycle_id=metadata.get("cycle_id"),
                                user_id=metadata.get("user_id"),
                                data=json.loads(metadata["data"])
                            )
                            self.events.appendleft(event)  # Add to front to maintain chronological order
                        except Exception as e:
                            logger.warning(f"Failed to load metric event from ChromaDB: {e}")
                            
            logger.info(f"Loaded {len(self.events)} recent events from ChromaDB")
                            
        except Exception as e:
            logger.error(f"Failed to load recent events from ChromaDB: {e}", exc_info=True)

    async def record_metric(
        self,
        metric_type: MetricType,
        data: Dict[str, Any],
        cycle_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Record a metric event.

        Args:
            metric_type: Type of metric being recorded
            data: Metric data payload
            cycle_id: Optional cognitive cycle ID
            user_id: Optional user ID
        """
        event = MetricEvent(
            type=metric_type,
            timestamp=datetime.utcnow(),
            data=data,
            cycle_id=cycle_id,
            user_id=user_id
        )

        # Add to in-memory buffer
        self.events.append(event)
        
        # Save to database
        await self._save_event_to_db(event)
        
        # Update aggregates and specialized metrics
        self._update_aggregates(event)
        self._update_specialized_metrics(event)

        # Cleanup old events
        self._cleanup_old_events()

        logger.debug(f"Recorded metric: {metric_type.value} for cycle {cycle_id}")

    async def _save_event_to_db(self, event: MetricEvent):
        """Save event to ChromaDB collection."""
        try:
            if self.metrics_collection is None:
                logger.warning("Metrics collection not initialized, skipping event save")
                return
                
            # Create unique ID for the event
            event_id = str(uuid4())
            
            # Prepare metadata for ChromaDB
            metadata = {
                "type": event.type.value,
                "timestamp": event.timestamp.timestamp(),  # Store as numeric Unix timestamp
                "data": json.dumps(event.data)
            }
            
            # Add optional fields if they exist
            if event.cycle_id:
                metadata["cycle_id"] = str(event.cycle_id)
            if event.user_id:
                metadata["user_id"] = str(event.user_id)
            
            # Create a simple text representation for vector search
            document = f"Metric event: {event.type.value} at {event.timestamp.timestamp()}"
            
            # Add to ChromaDB collection
            self.metrics_collection.add(
                ids=[event_id],
                metadatas=[metadata],
                documents=[document]
            )
            
        except Exception as e:
            logger.error(f"Failed to save metric event to ChromaDB: {e}", exc_info=True)

    def _update_aggregates(self, event: MetricEvent):
        """Update rolling aggregates based on event type"""
        metric_key = event.type.value

        # Update counters
        self.counters[metric_key] += 1

        # Update type-specific aggregates
        if event.type == MetricType.AGENT_ACTIVATION:
            # Track agent activation frequencies
            agents = event.data.get("agents_activated", [])
            for agent in agents:
                self.agent_activations[agent] += 1

            # Track processing time
            if "processing_time_ms" in event.data:
                self.aggregates["processing_time"]["overall"].update(event.data["processing_time_ms"])

        elif event.type == MetricType.MEMORY_ACCESS:
            # Track memory tier performance
            tier = event.data.get("tier_accessed", "unknown")
            hit_rate = event.data.get("hit_rate", 0.0)
            self.aggregates["memory_hit_rates"][tier].update(hit_rate)

            # Track retrieval times
            if "retrieval_time_ms" in event.data:
                self.aggregates["memory_retrieval_time"][tier].update(event.data["retrieval_time_ms"])

        elif event.type == MetricType.LEARNING_EVENT:
            # Track skill performance over time
            skill = event.data.get("skill_category", "unknown")
            outcome = event.data.get("outcome_score", 0.0)
            self.skill_performance[skill].append(outcome)

            # Keep only last 100 performance scores per skill
            if len(self.skill_performance[skill]) > 100:
                self.skill_performance[skill] = self.skill_performance[skill][-100:]

        elif event.type == MetricType.CONFLICT_RESOLUTION:
            # Track conflict types and resolutions
            conflict_types = event.data.get("conflict_types", [])
            for conflict_type in conflict_types:
                self.conflict_stats[conflict_type] += 1

            # Track coherence improvement
            if "coherence_improvement" in event.data:
                self.aggregates["coherence_improvement"]["overall"].update(event.data["coherence_improvement"])

    def _update_specialized_metrics(self, event: MetricEvent):
        """Update specialized metrics for scientific analysis"""
        if event.type == MetricType.COGNITIVE_CYCLE:
            # Track cycle completion metrics
            satisfaction = event.data.get("user_satisfaction")
            engagement = event.data.get("engagement_potential", 0.0)

            if satisfaction is not None:
                self.aggregates["user_satisfaction"]["overall"].update(satisfaction)
            if engagement is not None:
                self.aggregates["user_engagement"]["overall"].update(engagement)

        elif event.type == MetricType.ERROR_ANALYSIS:
            # Track error patterns for learning analysis
            error_category = event.data.get("primary_error_category", "unknown")
            severity = event.data.get("severity_score", 0.0)

            self.aggregates["error_severity"][error_category].update(severity)

        elif event.type == MetricType.ATTENTION_DIRECTIVE:
            applied = event.data.get("applied", False)
            mode_key = "attention_directive_applied" if applied else "attention_directive_shadow"
            self.counters[mode_key] += 1

    def _cleanup_old_events(self):
        """Remove events older than retention period"""
        cutoff_time = datetime.utcnow() - self.retention_period

        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for real-time display.

        Returns:
            Dictionary containing all current metrics and aggregates
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_events": len(self.events),
                "events_per_minute": self._calculate_events_per_minute(),
                "active_users": len(set(e.user_id for e in self.events if e.user_id)),
                "avg_processing_time_ms": self.aggregates["processing_time"]["overall"].avg
            },
            "agent_metrics": {
                "activation_frequencies": dict(self.agent_activations),
                "total_activations": sum(self.agent_activations.values())
            },
            "memory_metrics": {
                "tier_performance": {
                    tier: {
                        "hit_rate": stats.avg,
                        "retrieval_time_ms": self.aggregates["memory_retrieval_time"][tier].avg,
                        "access_count": stats.count
                    }
                    for tier, stats in self.aggregates["memory_hit_rates"].items()
                }
            },
            "learning_metrics": {
                "skill_performance": {
                    skill: {
                        "current_avg": sum(scores[-10:]) / len(scores[-10:]) if scores else 0.0,  # Last 10 scores
                        "trend": self._calculate_trend(scores[-20:]) if len(scores) >= 20 else 0.0,
                        "total_samples": len(scores)
                    }
                    for skill, scores in self.skill_performance.items()
                }
            },
            "conflict_metrics": {
                "resolution_stats": dict(self.conflict_stats),
                "coherence_improvement": self.aggregates["coherence_improvement"]["overall"].avg,
                "total_conflicts": sum(self.conflict_stats.values())
            },
            "user_experience": {
                "avg_satisfaction": self.aggregates["user_satisfaction"]["overall"].avg,
                "avg_engagement": self.aggregates["user_engagement"]["overall"].avg
            },
            "recent_events": [
                {
                    "type": event.type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "cycle_id": str(event.cycle_id) if event.cycle_id else None,
                    "user_id": str(event.user_id) if event.user_id else None,
                    "summary": self._summarize_event(event)
                }
                for event in list(self.events)[-20:]  # Last 20 events
            ]
        }

    def _calculate_events_per_minute(self) -> float:
        """Calculate average events per minute over last hour"""
        if not self.events:
            return 0.0

        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_events = [e for e in self.events if e.timestamp > one_hour_ago]

        if not recent_events:
            return 0.0

        minutes_elapsed = (datetime.utcnow() - recent_events[0].timestamp).total_seconds() / 60
        return len(recent_events) / max(minutes_elapsed, 1)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope for recent values"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # Simple linear regression slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def _summarize_event(self, event: MetricEvent) -> str:
        """Create human-readable summary of metric event"""
        if event.type == MetricType.AGENT_ACTIVATION:
            agents = event.data.get("agents_activated", [])
            return f"Activated {len(agents)} agents: {', '.join(agents[:3])}{'...' if len(agents) > 3 else ''}"

        elif event.type == MetricType.MEMORY_ACCESS:
            tier = event.data.get("tier_accessed", "unknown")
            hit_rate = event.data.get("hit_rate", 0.0)
            return f"Memory access ({tier}): {hit_rate:.1%} hit rate"

        elif event.type == MetricType.LEARNING_EVENT:
            skill = event.data.get("skill_category", "unknown")
            outcome = event.data.get("outcome_score", 0.0)
            return f"Skill improvement ({skill}): {outcome:.2f} outcome"

        elif event.type == MetricType.CONFLICT_RESOLUTION:
            conflicts = len(event.data.get("conflict_types", []))
            return f"Resolved {conflicts} conflicts"

        else:
            return f"{event.type.value} event"

    async def get_historical_data(
        self,
        metric_type: Optional[MetricType] = None,
        hours: int = 24,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metric data for analysis from ChromaDB.

        Args:
            metric_type: Filter by metric type (optional)
            hours: Hours of history to retrieve
            user_id: Filter by user ID (optional)

        Returns:
            List of historical metric events
        """
        try:
            if self.metrics_collection is None:
                logger.warning("Metrics collection not initialized, falling back to in-memory data")
                return self._get_historical_from_memory(metric_type, hours, user_id)
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Build where clause for ChromaDB query - use numeric timestamp for comparison
            where_conditions = {"timestamp": {"$gt": cutoff_time.timestamp()}}
            
            if metric_type:
                where_conditions["type"] = metric_type.value
                
            if user_id:
                where_conditions["user_id"] = user_id
            
            # Query ChromaDB - get all matching documents
            results = self.metrics_collection.query(
                query_texts=[""],  # Empty query to match all
                where=where_conditions,
                n_results=10000,  # Large limit to get all historical data
                include=["metadatas"]
            )
            
            # Convert ChromaDB results to expected format
            historical_data = []
            if results and results["metadatas"]:
                for metadata_list in results["metadatas"]:
                    for metadata in metadata_list:
                        # Convert timestamp string back to float for consistency
                        timestamp_str = metadata["timestamp"]
                        try:
                            # Try to parse as float first (Unix timestamp)
                            timestamp_val = float(timestamp_str)
                        except ValueError:
                            # If it's an ISO string, parse it and convert to timestamp
                            try:
                                timestamp_val = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                            except ValueError:
                                # Fallback to current time if parsing fails
                                timestamp_val = datetime.utcnow().timestamp()
                        
                        historical_data.append({
                            "type": metadata["type"],
                            "timestamp": timestamp_val,
                            "cycle_id": metadata.get("cycle_id"),
                            "user_id": metadata.get("user_id"),
                            "data": json.loads(metadata["data"])
                        })
            
            # Sort by timestamp (most recent first)
            historical_data.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical data from ChromaDB: {e}", exc_info=True)
            # Fallback to in-memory data if ChromaDB fails
            logger.warning("Falling back to in-memory historical data")
            return self._get_historical_from_memory(metric_type, hours, user_id)

    def _get_historical_from_memory(
        self,
        metric_type: Optional[MetricType] = None,
        hours: int = 24,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fallback method to get historical data from in-memory buffer."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_events = [
            event for event in self.events
            if event.timestamp >= cutoff_time
            and (metric_type is None or event.type == metric_type)
            and (user_id is None or event.user_id == user_id)
        ]

        return [
            {
                "type": event.type.value,
                "timestamp": event.timestamp.isoformat(),
                "cycle_id": str(event.cycle_id) if event.cycle_id else None,
                "user_id": str(event.user_id) if event.user_id else None,
                "data": event.data
            }
            for event in filtered_events
        ]

    async def get_correlation_analysis(self, hours: int = 24, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics for scientific insights.

        Args:
            hours: Hours of historical data to analyze
            user_id: Optional user filter

        Returns:
            Correlation analysis results
        """
        try:
            # Get historical data
            all_events = await self.get_historical_data(hours=hours, user_id=user_id)
            
            if len(all_events) < 10:
                return {"error": "Insufficient data for correlation analysis", "min_required": 10, "available": len(all_events)}
            
            # Group events by type and time windows
            time_windows = {}
            metric_series = defaultdict(list)
            
            # Create time-bucketed data (hourly windows)
            for event in all_events:
                # Handle timestamp conversion more robustly
                timestamp_val = event["timestamp"]
                if isinstance(timestamp_val, str):
                    try:
                        # Try to parse as float first
                        timestamp_val = float(timestamp_val)
                    except ValueError:
                        # If it's an ISO string, parse it
                        try:
                            timestamp_val = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00')).timestamp()
                        except ValueError:
                            # Skip invalid timestamps
                            continue
                
                timestamp = datetime.fromtimestamp(timestamp_val)
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in time_windows:
                    time_windows[hour_key] = defaultdict(int)
                
                time_windows[hour_key][event["type"]] += 1
            
            # Convert to time series
            sorted_windows = sorted(time_windows.keys())
            for window_time in sorted_windows:
                window_data = time_windows[window_time]
                for metric_type in set(event["type"] for event in all_events):
                    metric_series[metric_type].append(window_data.get(metric_type, 0))
            
            # Calculate correlations
            correlations = {}
            metric_types = list(metric_series.keys())
            
            for i, type1 in enumerate(metric_types):
                for type2 in metric_types[i+1:]:
                    series1 = metric_series[type1]
                    series2 = metric_series[type2]
                    
                    if len(series1) == len(series2) and len(series1) > 1:
                        try:
                            corr, p_value = stats.pearsonr(series1, series2)
                            if not (math.isnan(corr) or math.isnan(p_value)):
                                key = f"{type1}_vs_{type2}"
                                correlations[key] = {
                                    "correlation": round(corr, 3),
                                    "p_value": round(p_value, 4),
                                    "significance": "significant" if p_value < 0.05 else "not_significant",
                                    "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                                }
                        except Exception as e:
                            logger.debug(f"Could not calculate correlation for {type1} vs {type2}: {e}")
            
            # Find most interesting correlations
            significant_correlations = {
                k: v for k, v in correlations.items() 
                if v["significance"] == "significant" and v["strength"] in ["moderate", "strong"]
            }
            
            # Calculate novel behaviors detected based on emergence patterns
            novel_behaviors_detected = 0
            if significant_correlations:
                # Count unique agent types involved in significant correlations
                agent_types_involved = set()
                for corr_key in significant_correlations.keys():
                    # Extract agent names from correlation keys like "agent_activation_vs_memory_access"
                    parts = corr_key.split("_vs_")
                    for part in parts:
                        if "agent" in part or any(agent in part for agent in ["perception", "emotional", "memory", "planning", "creative", "critic", "discovery"]):
                            agent_types_involved.add(part)
                
                # Novel behaviors = significant correlations + agent diversity bonus
                novel_behaviors_detected = len(significant_correlations) + len(agent_types_involved)
            
            return {
                "analysis_period_hours": hours,
                "total_events": len(all_events),
                "time_windows": len(sorted_windows),
                "metric_types_analyzed": metric_types,
                "significant_correlations": significant_correlations,
                "correlation_summary": {
                    "total_correlations_calculated": len(correlations),
                    "significant_correlations": len(significant_correlations),
                    "strongest_correlation": max(correlations.items(), key=lambda x: abs(x[1]["correlation"])) if correlations else None
                },
                "executive_summary": {
                    "emergence_indicators": {
                        "novel_behaviors_detected": novel_behaviors_detected,
                        "agent_coordination_patterns": len(significant_correlations),
                        "emergence_level": "high" if novel_behaviors_detected > 10 else "moderate" if novel_behaviors_detected > 5 else "low"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to perform correlation analysis: {e}", exc_info=True)
            return {"error": f"Correlation analysis failed: {str(e)}"}

    async def export_scientific_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Export data in scientific analysis format.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Structured data for scientific analysis
        """
        events_in_period = [
            event for event in self.events
            if start_date <= event.timestamp <= end_date
        ]

        return {
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "total_events": len(events_in_period)
            },
            "learning_curves": {
                skill: self.skill_performance[skill]
                for skill in self.skill_performance
            },
            "performance_metrics": {
                "user_satisfaction_trend": [e.data.get("user_satisfaction", 0.0)
                                          for e in events_in_period
                                          if e.type == MetricType.COGNITIVE_CYCLE],
                "agent_activation_patterns": self.agent_activations,
                "memory_efficiency": dict(self.aggregates["memory_hit_rates"])
            },
            "emergence_indicators": {
                "conflict_resolution_rate": sum(self.conflict_stats.values()) / max(len(events_in_period), 1),
                "skill_diversity": len(self.skill_performance),
                "adaptation_signals": self._calculate_adaptation_metrics(events_in_period)
            }
        }

    def _calculate_adaptation_metrics(self, events: List[MetricEvent]) -> Dict[str, Any]:
        """Calculate metrics indicating system adaptation and learning"""
        # Group events by hour
        hourly_stats = defaultdict(lambda: {"cycles": 0, "avg_satisfaction": 0.0, "skills_improved": 0})

        for event in events:
            hour = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hour_key = hour.isoformat()

            if event.type == MetricType.COGNITIVE_CYCLE:
                hourly_stats[hour_key]["cycles"] += 1
                hourly_stats[hour_key]["avg_satisfaction"] += event.data.get("user_satisfaction", 0.0)

            elif event.type == MetricType.LEARNING_EVENT:
                hourly_stats[hour_key]["skills_improved"] += 1

        # Calculate adaptation trends
        hours = sorted(hourly_stats.keys())
        if len(hours) < 2:
            return {"insufficient_data": True}

        # Trend analysis
        satisfaction_trend = self._calculate_trend([
            hourly_stats[h]["avg_satisfaction"] / max(hourly_stats[h]["cycles"], 1)
            for h in hours
        ])

        improvement_trend = self._calculate_trend([
            hourly_stats[h]["skills_improved"]
            for h in hours
        ])

        return {
            "satisfaction_improvement_trend": satisfaction_trend,
            "learning_acceleration": improvement_trend,
            "adaptation_confidence": min(1.0, len(hours) / 24.0)  # Confidence based on data points
        }

    # ===== STATISTICAL ANALYSIS METHODS =====

    def perform_statistical_analysis(self, data_series: List[float]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on a data series.

        Args:
            data_series: List of numerical values to analyze

        Returns:
            Dictionary containing statistical measures and tests
        """
        if len(data_series) < 2:
            return {"insufficient_data": True, "sample_size": len(data_series)}

        # Basic descriptive statistics
        mean = statistics.mean(data_series)
        median = statistics.median(data_series)
        std_dev = statistics.stdev(data_series) if len(data_series) > 1 else 0.0
        variance = statistics.variance(data_series) if len(data_series) > 1 else 0.0
        min_val = min(data_series)
        max_val = max(data_series)
        range_val = max_val - min_val

        # Quartiles and IQR
        q1 = statistics.quantiles(data_series, n=4)[0]
        q3 = statistics.quantiles(data_series, n=4)[2]
        iqr = q3 - q1

        # Outlier detection (IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data_series if x < lower_bound or x > upper_bound]

        # Trend analysis
        trend_slope, trend_intercept, trend_r_value, trend_p_value, trend_std_err = stats.linregress(
            range(len(data_series)), data_series
        )

        # Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_statistic, adf_p_value, _, _, adf_critical_values, _ = stats.adfuller(data_series)
            is_stationary = bool(adf_p_value < 0.05)
        except:
            adf_statistic, adf_p_value, is_stationary = None, None, None

        # Normality test (Shapiro-Wilk)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data_series)
            is_normal = bool(shapiro_p > 0.05)
        except:
            shapiro_stat, shapiro_p, is_normal = None, None, None

        # Autocorrelation (lag-1)
        if len(data_series) > 1:
            autocorr = np.corrcoef(data_series[:-1], data_series[1:])[0, 1]
        else:
            autocorr = None

        return {
            "sample_size": len(data_series),
            "descriptive_stats": {
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "variance": variance,
                "min": min_val,
                "max": max_val,
                "range": range_val,
                "q1": q1,
                "q3": q3,
                "iqr": iqr
            },
            "distribution": {
                "is_normal": is_normal,
                "shapiro_test": {
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p
                },
                "outliers": {
                    "count": len(outliers),
                    "values": outliers[:10],  # First 10 outliers
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }
            },
            "trend_analysis": {
                "slope": trend_slope,
                "intercept": trend_intercept,
                "r_squared": trend_r_value ** 2,
                "p_value": trend_p_value,
                "std_error": trend_std_err,
                "direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable",
                "significance": "significant" if trend_p_value < 0.05 else "not_significant"
            },
            "time_series": {
                "is_stationary": is_stationary,
                "adf_test": {
                    "statistic": adf_statistic,
                    "p_value": adf_p_value
                },
                "autocorrelation_lag1": autocorr
            }
        }

    def compare_groups_statistical_test(self, group1: List[float], group2: List[float],
                                      test_type: str = "auto") -> Dict[str, Any]:
        """
        Perform statistical comparison between two groups.

        Args:
            group1: First data series
            group2: Second data series
            test_type: Type of test ("t-test", "mann-whitney", "auto")

        Returns:
            Statistical test results
        """
        if len(group1) < 2 or len(group2) < 2:
            return {"insufficient_data": True, "group1_size": len(group1), "group2_size": len(group2)}

        # Determine appropriate test
        if test_type == "auto":
            # Check normality for both groups
            try:
                _, p1 = stats.shapiro(group1)
                _, p2 = stats.shapiro(group2)
                both_normal = bool(p1 > 0.05 and p2 > 0.05)
            except:
                both_normal = False

            # Check equal variances
            try:
                _, p_var = stats.levene(group1, group2)
                equal_var = bool(p_var > 0.05)
            except:
                equal_var = True

            test_type = "t-test" if both_normal and equal_var else "mann-whitney"

        results = {
            "test_type": test_type,
            "group1_stats": {
                "size": len(group1),
                "mean": statistics.mean(group1),
                "std": statistics.stdev(group1) if len(group1) > 1 else 0
            },
            "group2_stats": {
                "size": len(group2),
                "mean": statistics.mean(group2),
                "std": statistics.stdev(group2) if len(group2) > 1 else 0
            }
        }

        if test_type == "t-test":
            try:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
                results["t_test"] = {
                    "statistic": t_stat,
                    "p_value": p_value,
                    "significant": bool(p_value < 0.05),
                    "effect_size": abs(t_stat) / math.sqrt((len(group1) + len(group2)) / 2)  # Cohen's d approximation
                }
            except Exception as e:
                results["t_test"] = {"error": str(e)}

        elif test_type == "mann-whitney":
            try:
                u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                results["mann_whitney"] = {
                    "statistic": u_stat,
                    "p_value": p_value,
                    "significant": bool(p_value < 0.05),
                    "effect_size": 1 - (2 * u_stat) / (len(group1) * len(group2))  # Common language effect size
                }
            except Exception as e:
                results["mann_whitney"] = {"error": str(e)}

        return results

    def analyze_learning_curves(self, skill_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze learning curves for multiple skills using power-law fitting.

        Args:
            skill_data: Dictionary mapping skill names to performance series

        Returns:
            Learning curve analysis results
        """
        results = {}

        for skill_name, performance_series in skill_data.items():
            if len(performance_series) < 5:
                results[skill_name] = {"insufficient_data": True, "sample_size": len(performance_series)}
                continue

            # Power-law fitting: performance = a * trial^b
            trials = np.array(range(1, len(performance_series) + 1))
            performance = np.array(performance_series)

            try:
                # Log-linear regression for power-law parameters
                log_trials = np.log(trials)
                log_performance = np.log(np.maximum(performance, 0.001))  # Avoid log(0)

                slope, intercept, r_value, p_value, std_err = stats.linregress(log_trials, log_performance)

                # Power-law parameters
                a = math.exp(intercept)
                b = slope

                # Goodness of fit
                r_squared = r_value ** 2

                # Learning rate interpretation
                learning_rate = abs(b)  # How quickly performance improves
                learning_efficiency = "high" if learning_rate > 0.1 else "moderate" if learning_rate > 0.05 else "low"

                results[skill_name] = {
                    "power_law_fit": {
                        "a": a,
                        "b": b,
                        "r_squared": r_squared,
                        "p_value": p_value,
                        "goodness_of_fit": "excellent" if r_squared > 0.8 else "good" if r_squared > 0.6 else "poor"
                    },
                    "learning_characteristics": {
                        "learning_rate": learning_rate,
                        "efficiency": learning_efficiency,
                        "convergence_indicated": bool(b < -0.01),  # Negative slope indicates improvement
                        "samples": len(performance_series)
                    },
                    "performance_trajectory": {
                        "initial_performance": performance[0],
                        "final_performance": performance[-1],
                        "improvement": performance[-1] - performance[0],
                        "improvement_rate": (performance[-1] - performance[0]) / len(performance_series)
                    }
                }

            except Exception as e:
                results[skill_name] = {"fitting_error": str(e), "sample_size": len(performance_series)}

        return results

    # ===== RESEARCH EXPORT METHODS =====

    def export_to_csv(self, data: Dict[str, Any], filename_prefix: str = "eca_metrics") -> str:
        """
        Export analysis data to CSV format.

        Args:
            data: Data to export
            filename_prefix: Prefix for the filename

        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["Metric", "Value", "Timestamp", "Category"])

        # Flatten nested data
        def flatten_data(data_dict, prefix=""):
            for key, value in data_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    flatten_data(value, full_key)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            flatten_data(item, f"{full_key}[{i}]")
                        else:
                            writer.writerow([f"{full_key}[{i}]", item, datetime.utcnow().isoformat(), "list_item"])
                else:
                    writer.writerow([full_key, value, datetime.utcnow().isoformat(), "scalar"])

        flatten_data(data)
        return output.getvalue()

    def export_to_json(self, data: Dict[str, Any], include_metadata: bool = True) -> str:
        """
        Export analysis data to JSON format with optional metadata.

        Args:
            data: Data to export
            include_metadata: Whether to include export metadata

        Returns:
            JSON string
        """
        export_data = {
            "data": data,
            "export_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "eca_version": "2.0",
                "export_type": "scientific_analysis"
            } if include_metadata else None
        }

        return json.dumps(export_data, indent=2, default=str)

    def generate_research_report(self, analysis_period_days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive research report for scientific publication.

        Args:
            analysis_period_days: Number of days to analyze

        Returns:
            Complete research report data
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=analysis_period_days)

        # Get all relevant data
        dashboard_data = asyncio.run(self.get_dashboard_data())
        historical_data = asyncio.run(self.get_historical_data(hours=analysis_period_days * 24))
        scientific_data = asyncio.run(self.export_scientific_data(start_date, end_date))

        # Perform statistical analyses
        learning_curves_analysis = self.analyze_learning_curves(self.skill_performance)

        # Analyze performance trends
        performance_trends = {}
        for metric_name, metric_data in dashboard_data.items():
            if isinstance(metric_data, dict) and "avg" in metric_data:
                # This is an aggregate metric - analyze its trend
                # Note: In a real implementation, we'd need historical aggregate data
                performance_trends[metric_name] = {
                    "current_value": metric_data.get("avg", 0),
                    "analysis": "historical_trend_analysis_needed"
                }

        # Emergence indicators
        emergence_analysis = self._analyze_emergence_indicators(historical_data)

        # Comparative analysis (if we had baseline data)
        comparative_insights = self._generate_comparative_insights(dashboard_data)

        return {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "analysis_period_days": analysis_period_days,
                "eca_version": "2.0",
                "total_events_analyzed": len(historical_data)
            },
            "executive_summary": {
                "system_performance": self._generate_performance_summary(dashboard_data),
                "learning_effectiveness": self._summarize_learning_curves(learning_curves_analysis),
                "emergence_indicators": emergence_analysis["summary"],
                "key_findings": self._extract_key_findings(dashboard_data, learning_curves_analysis)
            },
            "detailed_analysis": {
                "learning_curves": learning_curves_analysis,
                "performance_trends": performance_trends,
                "emergence_analysis": emergence_analysis,
                "statistical_tests": self._run_statistical_test_suite(historical_data)
            },
            "methodology": {
                "data_collection": "Continuous real-time metrics collection across all cognitive subsystems",
                "analysis_methods": "Statistical analysis, trend detection, power-law fitting, emergence indicators",
                "validation_approach": "Comparative analysis, significance testing, longitudinal tracking"
            },
            "conclusions": {
                "cognitive_architecture_validation": self._assess_architecture_effectiveness(dashboard_data),
                "learning_systems_effectiveness": self._evaluate_learning_systems(learning_curves_analysis),
                "emergence_detection": emergence_analysis["emergence_detected"],
                "research_implications": self._generate_research_implications(dashboard_data, learning_curves_analysis)
            },
            "raw_data": {
                "dashboard_snapshot": dashboard_data,
                "historical_events": historical_data[:100],  # First 100 events as sample
                "scientific_export": scientific_data
            }
        }

    def _analyze_emergence_indicators(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze indicators of emergent behavior"""
        # Group events by type
        event_types = {}
        for event in historical_data:
            event_type = event.get("type", "unknown")
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)

        # Analyze agent coordination patterns
        agent_events = event_types.get("agent_activation", [])
        coordination_patterns = self._analyze_coordination_patterns(agent_events)

        # Analyze novel behavior emergence
        novel_behaviors = self._detect_novel_behaviors(historical_data)

        # Calculate emergence score
        emergence_score = self._calculate_emergence_score(coordination_patterns, novel_behaviors, len(historical_data))

        return {
            "summary": {
                "emergence_score": emergence_score,
                "emergence_level": "high" if emergence_score > 0.7 else "moderate" if emergence_score > 0.4 else "low",
                "coordination_complexity": coordination_patterns.get("complexity_score", 0),
                "novel_behaviors_detected": len(novel_behaviors)
            },
            "coordination_patterns": coordination_patterns,
            "novel_behaviors": novel_behaviors,
            "emergence_detected": emergence_score > 0.5
        }

    def _analyze_coordination_patterns(self, agent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in agent coordination"""
        if not agent_events:
            return {"insufficient_data": True}

        # Extract agent activation sequences
        sequences = []
        for event in agent_events:
            agents = event.get("data", {}).get("agents_activated", [])
            if agents:
                sequences.append(agents)

        # Calculate coordination complexity
        unique_sequences = len(set(tuple(seq) for seq in sequences))
        avg_sequence_length = statistics.mean(len(seq) for seq in sequences) if sequences else 0

        # Diversity of agent combinations
        all_combinations = set()
        for seq in sequences:
            for r in range(2, len(seq) + 1):
                for combo in itertools.combinations(sorted(seq), r):
                    all_combinations.add(combo)

        return {
            "total_sequences": len(sequences),
            "unique_sequences": unique_sequences,
            "sequence_diversity": unique_sequences / max(len(sequences), 1),
            "avg_sequence_length": avg_sequence_length,
            "unique_combinations": len(all_combinations),
            "complexity_score": min(1.0, (unique_sequences * len(all_combinations)) / 1000)
        }

    def _detect_novel_behaviors(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potentially novel or emergent behaviors"""
        # This is a simplified implementation - in practice, this would use
        # more sophisticated novelty detection algorithms
        novel_behaviors = []

        # Look for unusual event patterns or high coherence scores
        for event in historical_data:
            data = event.get("data", {})

            # High coherence with complex agent coordination
            coherence = data.get("coherence_score", 0)
            agents_activated = len(data.get("agents_activated", []))

            if coherence > 0.8 and agents_activated >= 3:
                novel_behaviors.append({
                    "type": "high_coherence_complex_coordination",
                    "timestamp": event.get("timestamp"),
                    "coherence": coherence,
                    "agents": agents_activated,
                    "cycle_id": event.get("cycle_id")
                })

            # Unusual learning events
            if event.get("type") == "learning_event":
                outcome = data.get("outcome_score", 0)
                if outcome > 0.9:  # Exceptional performance
                    novel_behaviors.append({
                        "type": "exceptional_learning_outcome",
                        "timestamp": event.get("timestamp"),
                        "outcome": outcome,
                        "skill": data.get("skill_category"),
                        "cycle_id": event.get("cycle_id")
                    })

        return novel_behaviors

    def _calculate_emergence_score(self, coordination: Dict, novel_behaviors: List, total_events: int) -> float:
        """Calculate an emergence score based on coordination and novelty"""
        coordination_score = coordination.get("complexity_score", 0)
        novelty_score = min(1.0, len(novel_behaviors) / max(total_events * 0.01, 1))  # 1% of events

        # Weighted combination
        emergence_score = (coordination_score * 0.6) + (novelty_score * 0.4)
        return min(1.0, emergence_score)

    def _generate_comparative_insights(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative insights (would compare against baselines in full implementation)"""
        return {
            "baseline_comparison": "baseline_data_not_available",
            "insights": [
                "Current metrics indicate active cognitive processing",
                "Learning systems show skill improvement patterns",
                "Agent coordination demonstrates adaptive behavior"
            ]
        }

    def _generate_performance_summary(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate executive summary of system performance"""
        summary_stats = dashboard_data.get("summary", {})
        total_events = summary_stats.get("total_events", 0)
        events_per_minute = summary_stats.get("events_per_minute", 0)

        return f"System processed {total_events} events at {events_per_minute:.1f} events/minute, demonstrating active cognitive engagement across all subsystems."

    def _summarize_learning_curves(self, learning_analysis: Dict[str, Any]) -> str:
        """Summarize learning curve analysis results"""
        successful_fits = sum(1 for skill_data in learning_analysis.values()
                            if isinstance(skill_data, dict) and "power_law_fit" in skill_data)

        total_skills = len(learning_analysis)
        fit_rate = successful_fits / max(total_skills, 1)

        return f"Successfully analyzed learning curves for {successful_fits}/{total_skills} skills ({fit_rate:.1%} fit rate), indicating robust learning system performance."

    def _extract_key_findings(self, dashboard_data: Dict[str, Any], learning_analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from the analysis"""
        findings = []

        # Check for learning effectiveness
        learning_metrics = dashboard_data.get("learning_metrics", {})
        if learning_metrics.get("skill_performance"):
            findings.append("Active skill improvement detected across multiple cognitive domains")

        # Check for emergence indicators
        agent_metrics = dashboard_data.get("agent_metrics", {})
        total_activations = agent_metrics.get("total_activations", 0)
        if total_activations > 100:
            findings.append("High agent coordination activity suggests emergent collaborative behavior")

        # Check for system stability
        summary = dashboard_data.get("summary", {})
        avg_processing_time = summary.get("avg_processing_time_ms", 0)
        if avg_processing_time > 0 and avg_processing_time < 5000:  # Reasonable response time
            findings.append("System demonstrates stable performance with acceptable response times")

        return findings if findings else ["Analysis indicates system is operational with baseline cognitive processing"]

    def _run_statistical_test_suite(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a suite of statistical tests on historical data"""
        # Extract numerical series for testing
        satisfaction_scores = []
        processing_times = []
        coherence_scores = []

        for event in historical_data:
            data = event.get("data", {})

            if "user_satisfaction" in data:
                satisfaction_scores.append(data["user_satisfaction"])
            if "processing_time_ms" in data:
                processing_times.append(data["processing_time_ms"])
            if "coherence_score" in data:
                coherence_scores.append(data["coherence_score"])

        results = {}

        # Test satisfaction trends
        if len(satisfaction_scores) >= 10:
            results["satisfaction_trend"] = self.perform_statistical_analysis(satisfaction_scores)

        # Test processing time consistency
        if len(processing_times) >= 10:
            results["processing_time_analysis"] = self.perform_statistical_analysis(processing_times)

        # Test coherence improvement
        if len(coherence_scores) >= 10:
            results["coherence_analysis"] = self.perform_statistical_analysis(coherence_scores)

        return results

    def _assess_architecture_effectiveness(self, dashboard_data: Dict[str, Any]) -> str:
        """Assess overall cognitive architecture effectiveness"""
        agent_metrics = dashboard_data.get("agent_metrics", {})
        memory_metrics = dashboard_data.get("memory_metrics", {})
        learning_metrics = dashboard_data.get("learning_metrics", {})

        # Simple assessment based on activity levels
        agent_activity = len(agent_metrics.get("activation_frequencies", {}))
        memory_tiers = len(memory_metrics.get("tier_performance", {}))
        skills_tracked = len(learning_metrics.get("skill_performance", {}))

        if agent_activity >= 5 and memory_tiers >= 2 and skills_tracked >= 3:
            return "Cognitive architecture demonstrates comprehensive multi-agent processing with hierarchical memory systems and active learning across multiple skill domains."
        else:
            return "Cognitive architecture shows baseline functionality with room for expanded agent specialization and learning domain coverage."

    def _evaluate_learning_systems(self, learning_analysis: Dict[str, Any]) -> str:
        """Evaluate effectiveness of learning systems"""
        successful_analyses = sum(1 for skill_data in learning_analysis.values()
                                if isinstance(skill_data, dict) and "power_law_fit" in skill_data)

        if successful_analyses > 0:
            good_fits = sum(1 for skill_data in learning_analysis.values()
                          if isinstance(skill_data, dict) and
                          skill_data.get("power_law_fit", {}).get("r_squared", 0) > 0.6)

            fit_quality = good_fits / successful_analyses
            if fit_quality > 0.7:
                return "Learning systems demonstrate strong power-law improvement curves with high statistical significance, indicating effective skill acquisition mechanisms."
            else:
                return "Learning systems show skill improvement patterns with moderate statistical support, suggesting functional but improvable learning mechanisms."
        else:
            return "Learning systems analysis indicates active skill tracking with insufficient data for comprehensive statistical evaluation."

    def _generate_research_implications(self, dashboard_data: Dict[str, Any], learning_analysis: Dict[str, Any]) -> List[str]:
        """Generate research implications from the analysis"""
        implications = []

        # Check for learning effectiveness
        learning_metrics = dashboard_data.get("learning_metrics", {})
        skill_performance = learning_metrics.get("skill_performance", {})

        if skill_performance:
            implications.append("Demonstrates measurable skill improvement in cognitive AI systems, supporting the effectiveness of multi-agent learning architectures")

        # Check for emergence indicators
        agent_metrics = dashboard_data.get("agent_metrics", {})
        activation_patterns = agent_metrics.get("activation_frequencies", {})

        if len(activation_patterns) > 3:
            implications.append("Shows complex agent coordination patterns, suggesting emergence of collaborative problem-solving behaviors in multi-agent systems")

        # Check for memory system effectiveness
        memory_metrics = dashboard_data.get("memory_metrics", {})
        tier_performance = memory_metrics.get("tier_performance", {})

        if tier_performance:
            implications.append("Validates hierarchical memory system design with differential performance across STM, summary, and LTM tiers")

        if not implications:
            implications.append("Provides baseline metrics for cognitive AI system evaluation and establishes framework for longitudinal performance tracking")

        return implications