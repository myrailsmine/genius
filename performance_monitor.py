from typing import Dict, List
from utils.logger import AsyncLogger
import time
from collections import defaultdict

class PerformanceMonitor:
    """
    A utility class to monitor performance metrics for the agentic framework.
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.task_latencies = {}
        self.agent_success_rates = defaultdict(lambda: {"success": 0, "total": 0})

    async def record_task_start(self, task_id: str):
        """
        Record the start time of a task.
        """
        self.task_latencies[task_id] = time.time()
        await AsyncLogger.info(f"Task {task_id} started")

    async def record_task_end(self, task_id: str, success: bool = True):
        """
        Record the end time of a task and calculate latency.
        """
        if task_id in self.task_latencies:
            latency = time.time() - self.task_latencies[task_id]
            self.metrics["task_latency"].append(latency)
            agent = task_id.split("_")[0]  # Assume task_id format: agent_subtask
            self.agent_success_rates[agent]["total"] += 1
            if success:
                self.agent_success_rates[agent]["success"] += 1
            await AsyncLogger.info(f"Task {task_id} completed in {latency:.2f} seconds (success: {success})")
            del self.task_latencies[task_id]

    async def record_metric(self, metric_name: str, value: float):
        """
        Record a custom metric.
        """
        self.metrics[metric_name].append(value)
        await AsyncLogger.info(f"Recorded metric {metric_name}: {value}")

    async def get_metrics(self) -> Dict:
        """
        Get aggregated metrics.

        Returns:
            Dict: Metrics summary.
        """
        summary = {
            "average_task_latency": sum(self.metrics["task_latency"]) / len(self.metrics["task_latency"]) if self.metrics["task_latency"] else 0,
            "total_tasks": len(self.metrics["task_latency"]),
            "agent_success_rates": {
                agent: success_data["success"] / success_data["total"] if success_data["total"] > 0 else 0
                for agent, success_data in self.agent_success_rates.items()
            }
        }
        return summary
