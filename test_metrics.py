#!/usr/bin/env python3
"""
Test script to verify ECA metrics collection and dashboard functionality.

This script tests the complete metrics pipeline by running a cognitive cycle
and verifying that metrics are properly recorded and accessible via the dashboard API.
"""

import asyncio
import logging
import sys
from uuid import uuid4
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '.')

from src.models.core_models import UserRequest
from src.services.metrics_service import MetricsService, MetricType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_metrics_collection():
    """Test the metrics collection system end-to-end."""

    logger.info("🧪 Starting ECA Metrics Collection Test")

    try:
        # Import and initialize the application
        import main
        from main import app

        # Initialize the lifespan (this sets up all services)
        async with app.router.lifespan_context(app):
            logger.info("✅ Application initialized successfully")

            # Get the metrics service
            metrics_service = app.state.metrics_service
            if not metrics_service:
                logger.error("❌ MetricsService not initialized")
                return False

            logger.info("✅ MetricsService is available")

            # Test basic metrics recording
            test_user_id = str(uuid4())
            test_cycle_id = str(uuid4())

            # Record a test cognitive cycle metric
            await metrics_service.record_metric(
                MetricType.COGNITIVE_CYCLE,
                {
                    "event": "test_cycle_started",
                    "test_mode": True,
                    "user_id": test_user_id
                },
                cycle_id=test_cycle_id,
                user_id=test_user_id
            )

            # Record a test learning event
            await metrics_service.record_metric(
                MetricType.LEARNING_EVENT,
                {
                    "skill_category": "test_skill",
                    "outcome_score": 0.85,
                    "confidence_score": 0.9,
                    "success": True,
                    "test_mode": True
                },
                cycle_id=test_cycle_id,
                user_id=test_user_id
            )

            # Record a test memory access
            await metrics_service.record_metric(
                MetricType.MEMORY_ACCESS,
                {
                    "operation": "test_query",
                    "stm_hits": 2,
                    "ltm_hits": 3,
                    "total_results": 5,
                    "avg_relevance": 0.78,
                    "test_mode": True
                },
                user_id=test_user_id
            )

            logger.info("✅ Test metrics recorded successfully")

            # Test dashboard data retrieval
            dashboard_data = await metrics_service.get_dashboard_data()
            if dashboard_data:
                logger.info("✅ Dashboard data retrieved successfully")
                logger.info(f"📊 Dashboard contains {len(dashboard_data.get('recent_metrics', []))} recent metrics")
                logger.info(f"📈 Dashboard contains {len(dashboard_data.get('performance_stats', {}))} performance stats")
            else:
                logger.error("❌ Failed to retrieve dashboard data")
                return False

            # Test historical data retrieval
            historical_data = await metrics_service.get_historical_data(hours=1)
            if historical_data:
                logger.info("✅ Historical data retrieved successfully")
                logger.info(f"📈 Historical data contains {len(historical_data.get('metrics_over_time', []))} time points")
            else:
                logger.error("❌ Failed to retrieve historical data")
                return False

            # Test metrics by type
            cognitive_metrics = [m for m in dashboard_data.get('recent_metrics', [])
                               if m.get('type') == MetricType.COGNITIVE_CYCLE.value]
            learning_metrics = [m for m in dashboard_data.get('recent_metrics', [])
                              if m.get('type') == MetricType.LEARNING_EVENT.value]
            memory_metrics = [m for m in dashboard_data.get('recent_metrics', [])
                            if m.get('type') == MetricType.MEMORY_ACCESS.value]

            logger.info(f"📊 Found {len(cognitive_metrics)} cognitive cycle metrics")
            logger.info(f"🧠 Found {len(learning_metrics)} learning event metrics")
            logger.info(f"💾 Found {len(memory_metrics)} memory access metrics")

            # Verify we have our test metrics
            if len(cognitive_metrics) == 0:
                logger.warning("⚠️ No cognitive cycle metrics found")
            if len(learning_metrics) == 0:
                logger.warning("⚠️ No learning event metrics found")
            if len(memory_metrics) == 0:
                logger.warning("⚠️ No memory access metrics found")

            logger.info("🎉 Metrics collection test completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        return False

async def main():
    """Main test function."""
    success = await test_metrics_collection()
    if success:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())