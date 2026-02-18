# -*- coding: utf-8 -*-
"""Location: ./mcp-servers/python/mcp_eval_server/mcp_eval_server/server.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

MCP Evaluation Server - Main entry point using FastMCP.
"""

# Standard
import asyncio
import logging
import os
from typing import Any, Dict

import orjson
# Load .env file if it exists
try:
    # Third-Party
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not available, skip
    pass

# Third-Party
from fastmcp import FastMCP

# Local
from .health import mark_judge_tools_ready, mark_ready, mark_storage_ready, start_health_server, stop_health_server
from .storage.cache import BenchmarkCache, EvaluationCache, JudgeResponseCache
from .storage.results_store import ResultsStore
from .tools.agent_tools import AgentTools
from .tools.bias_tools import BiasTools
from .tools.calibration_tools import CalibrationTools
from .tools.judge_tools import JudgeTools
from .tools.multilingual_tools import MultilingualTools
from .tools.performance_tools import PerformanceTools
from .tools.privacy_tools import PrivacyTools
from .tools.prompt_tools import PromptTools
from .tools.quality_tools import QualityTools
from .tools.rag_tools import RAGTools
from .tools.robustness_tools import RobustnessTools
from .tools.safety_tools import SafetyTools
from .tools.workflow_tools import WorkflowTools

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("mcp-eval-server")

# Global variables for tools (initialized in lifespan)
JUDGE_TOOLS = None  # pylint: disable=invalid-name
PROMPT_TOOLS = None  # pylint: disable=invalid-name
AGENT_TOOLS = None  # pylint: disable=invalid-name
QUALITY_TOOLS = None  # pylint: disable=invalid-name
RAG_TOOLS = None  # pylint: disable=invalid-name
BIAS_TOOLS = None  # pylint: disable=invalid-name
ROBUSTNESS_TOOLS = None  # pylint: disable=invalid-name
SAFETY_TOOLS = None  # pylint: disable=invalid-name
MULTILINGUAL_TOOLS = None  # pylint: disable=invalid-name
PERFORMANCE_TOOLS = None  # pylint: disable=invalid-name
PRIVACY_TOOLS = None  # pylint: disable=invalid-name
WORKFLOW_TOOLS = None  # pylint: disable=invalid-name
CALIBRATION_TOOLS = None  # pylint: disable=invalid-name
EVALUATION_CACHE = None  # pylint: disable=invalid-name
JUDGE_CACHE = None  # pylint: disable=invalid-name
BENCHMARK_CACHE = None  # pylint: disable=invalid-name
RESULTS_STORE = None  # pylint: disable=invalid-name
HEALTH_SERVER = None  # pylint: disable=invalid-name


# Judge tools
@mcp.tool()
async def judge_evaluate_response(
    response: str,
    criteria: list[dict[str, Any]],
    rubric: dict[str, Any],
    judge_model: str = "gpt-4o-mini",
    context: str = "",
    use_cot: bool = True,
) -> dict[str, Any]:
    """Evaluate a single response using LLM-as-a-judge with customizable criteria and rubrics.
    
    Args:
        response: Text response to evaluate
        criteria: List of evaluation criteria
        rubric: Scoring rubric
        judge_model: Judge model to use (default: gpt-4o-mini)
        context: Optional context
        use_cot: Use chain-of-thought reasoning (default: True)
    
    Returns:
        Evaluation results with scores and reasoning
    """
    try:
        result = await JUDGE_TOOLS.evaluate_response(
            response=response,
            criteria=criteria,
            rubric=rubric,
            judge_model=judge_model,
            context=context,
            use_cot=use_cot,
        )
        return result
    except Exception as e:
        logger.error(f"Error in judge_evaluate_response: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def judge_pairwise_comparison(
    response_a: str,
    response_b: str,
    criteria: list[dict[str, Any]],
    judge_model: str = "gpt-4o-mini",
    context: str = "",
    position_bias_mitigation: bool = True,
) -> dict[str, Any]:
    """Compare two responses and determine which is better using LLM-as-a-judge.
    
    Args:
        response_a: First response
        response_b: Second response
        criteria: Comparison criteria
        judge_model: Judge model to use (default: gpt-4o-mini)
        context: Optional context
        position_bias_mitigation: Mitigate position bias (default: True)
    
    Returns:
        Comparison results with winner and reasoning
    """
    try:
        result = await JUDGE_TOOLS.pairwise_comparison(
            response_a=response_a,
            response_b=response_b,
            criteria=criteria,
            judge_model=judge_model,
            context=context,
            position_bias_mitigation=position_bias_mitigation,
        )
        return result
    except Exception as e:
        logger.error(f"Error in judge_pairwise_comparison: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def judge_rank_responses(
    responses: list[str],
    criteria: list[dict[str, Any]],
    judge_model: str = "gpt-4o-mini",
    context: str = "",
    ranking_method: str = "tournament",
) -> dict[str, Any]:
    """Rank multiple responses from best to worst using LLM-as-a-judge.
    
    Args:
        responses: List of responses to rank
        criteria: Ranking criteria
        judge_model: Judge model to use (default: gpt-4o-mini)
        context: Optional context
        ranking_method: Method to use (tournament, round_robin, or scoring)
    
    Returns:
        Ranked responses with scores
    """
    try:
        result = await JUDGE_TOOLS.rank_responses(
            responses=responses,
            criteria=criteria,
            judge_model=judge_model,
            context=context,
            ranking_method=ranking_method,
        )
        return result
    except Exception as e:
        logger.error(f"Error in judge_rank_responses: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def judge_evaluate_with_reference(
    response: str,
    reference: str,
    judge_model: str = "gpt-4o-mini",
    evaluation_type: str = "factuality",
    tolerance: str = "moderate",
) -> dict[str, Any]:
    """Evaluate response against a gold standard reference using LLM-as-a-judge.
    
    Args:
        response: Generated response
        reference: Gold standard reference
        judge_model: Judge model to use (default: gpt-4o-mini)
        evaluation_type: Type of evaluation (factuality, completeness, or style_match)
        tolerance: Tolerance level (strict, moderate, or loose)
    
    Returns:
        Evaluation results comparing response to reference
    """
    try:
        result = await JUDGE_TOOLS.evaluate_with_reference(
            response=response,
            reference=reference,
            judge_model=judge_model,
            evaluation_type=evaluation_type,
            tolerance=tolerance,
        )
        return result
    except Exception as e:
        logger.error(f"Error in judge_evaluate_with_reference: {str(e)}")
        return {"error": str(e)}


# Prompt evaluation tools
@mcp.tool()
async def prompt_evaluate_clarity(
    prompt_text: str,
    target_model: str = "general",
    domain_context: str = "",
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Assess prompt clarity using multiple rule-based and LLM-based metrics.
    
    Args:
        prompt_text: The prompt to evaluate
        target_model: Model the prompt is designed for (default: general)
        domain_context: Optional domain-specific requirements
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Clarity assessment with scores and recommendations
    """
    try:
        result = await PROMPT_TOOLS.evaluate_clarity(
            prompt_text=prompt_text,
            target_model=target_model,
            domain_context=domain_context,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in prompt_evaluate_clarity: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def prompt_test_consistency(
    prompt: str,
    test_inputs: list[str],
    num_runs: int = 3,
    temperature_range: list[float] = [0.1, 0.5, 0.9],
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Test prompt consistency across multiple runs and temperature settings.
    
    Args:
        prompt: Prompt template
        test_inputs: List of input variations
        num_runs: Repetitions per input (default: 3)
        temperature_range: Test different temperatures (default: [0.1, 0.5, 0.9])
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Consistency analysis across runs and temperatures
    """
    try:
        result = await PROMPT_TOOLS.test_consistency(
            prompt=prompt,
            test_inputs=test_inputs,
            num_runs=num_runs,
            temperature_range=temperature_range,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in prompt_test_consistency: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def prompt_measure_completeness(
    prompt: str,
    expected_components: list[str],
    test_samples: list[str] = [],
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Evaluate if prompt generates complete responses covering expected components.
    
    Args:
        prompt: The prompt text
        expected_components: List of required elements
        test_samples: Sample outputs to analyze (default: [])
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Completeness analysis with coverage metrics
    """
    try:
        result = await PROMPT_TOOLS.measure_completeness(
            prompt=prompt,
            expected_components=expected_components,
            test_samples=test_samples,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in prompt_measure_completeness: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def prompt_assess_relevance(
    prompt: str,
    outputs: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    relevance_threshold: float = 0.7,
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Measure semantic alignment between prompt and outputs using embeddings.
    
    Args:
        prompt: Input prompt
        outputs: Generated responses
        embedding_model: Model for semantic similarity (default: all-MiniLM-L6-v2)
        relevance_threshold: Minimum acceptable score (default: 0.7)
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Relevance scores and semantic alignment metrics
    """
    try:
        result = await PROMPT_TOOLS.assess_relevance(
            prompt=prompt,
            outputs=outputs,
            embedding_model=embedding_model,
            relevance_threshold=relevance_threshold,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in prompt_assess_relevance: {str(e)}")
        return {"error": str(e)}


# Agent evaluation tools
@mcp.tool()
async def agent_evaluate_tool_use(
    agent_trace: dict[str, Any],
    expected_tools: list[str],
    tool_sequence_matters: bool = False,
    allow_extra_tools: bool = True,
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Assess agent's tool selection and usage effectiveness.
    
    Args:
        agent_trace: Complete execution trace with tool calls
        expected_tools: Tools that should be used
        tool_sequence_matters: Whether order is important (default: False)
        allow_extra_tools: Permit additional tool calls (default: True)
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Tool usage analysis with effectiveness scores
    """
    try:
        result = await AGENT_TOOLS.evaluate_tool_use(
            agent_trace=agent_trace,
            expected_tools=expected_tools,
            tool_sequence_matters=tool_sequence_matters,
            allow_extra_tools=allow_extra_tools,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in agent_evaluate_tool_use: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def agent_measure_task_completion(
    task_description: str,
    success_criteria: list[dict[str, Any]],
    agent_trace: dict[str, Any],
    final_state: dict[str, Any] = {},
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Evaluate end-to-end task success against measurable criteria.
    
    Args:
        task_description: What the agent should accomplish
        success_criteria: Measurable outcomes
        agent_trace: Execution history
        final_state: System state after execution (default: {})
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Task completion analysis with success metrics
    """
    try:
        result = await AGENT_TOOLS.measure_task_completion(
            task_description=task_description,
            success_criteria=success_criteria,
            agent_trace=agent_trace,
            final_state=final_state,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in agent_measure_task_completion: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def agent_analyze_reasoning(
    reasoning_trace: list[dict[str, Any]],
    decision_points: list[dict[str, Any]],
    context: dict[str, Any],
    optimal_path: list[str] = [],
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Evaluate agent's decision-making process and reasoning quality.
    
    Args:
        reasoning_trace: Agent's thought process
        decision_points: Key choices made
        context: Available information
        optimal_path: Best possible approach (default: [])
        judge_model: Judge model to use (default: gpt-4o-mini)
    
    Returns:
        Reasoning quality analysis with decision evaluation
    """
    try:
        result = await AGENT_TOOLS.analyze_reasoning(
            reasoning_trace=reasoning_trace,
            decision_points=decision_points,
            context=context,
            optimal_path=optimal_path,
            judge_model=judge_model,
        )
        return result
    except Exception as e:
        logger.error(f"Error in agent_analyze_reasoning: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def agent_benchmark_performance(
    benchmark_suite: str,
    agent_config: dict[str, Any],
    baseline_comparison: dict[str, Any] = {},
    metrics_focus: list[str] = ["accuracy", "efficiency", "reliability"],
) -> dict[str, Any]:
    """Run comprehensive agent benchmarks comparing against baselines.
    
    Args:
        benchmark_suite: Which tests to run
        agent_config: Agent setup
        baseline_comparison: Compare to other agents (default: {})
        metrics_focus: Priority metrics (default: ["accuracy", "efficiency", "reliability"])
    
    Returns:
        Benchmark results with comparative analysis
    """
    try:
        result = await AGENT_TOOLS.benchmark_performance(
            benchmark_suite=benchmark_suite,
            agent_config=agent_config,
            baseline_comparison=baseline_comparison,
            metrics_focus=metrics_focus,
        )
        return result
    except Exception as e:
        logger.error(f"Error in agent_benchmark_performance: {str(e)}")
        return {"error": str(e)}


async def initialize_tools():
    """Initialize all tools and storage."""
    global JUDGE_TOOLS, PROMPT_TOOLS, AGENT_TOOLS, QUALITY_TOOLS, RAG_TOOLS, BIAS_TOOLS, ROBUSTNESS_TOOLS, SAFETY_TOOLS, MULTILINGUAL_TOOLS, PERFORMANCE_TOOLS, PRIVACY_TOOLS, WORKFLOW_TOOLS, CALIBRATION_TOOLS  # pylint: disable=global-statement
    global EVALUATION_CACHE, JUDGE_CACHE, BENCHMARK_CACHE, RESULTS_STORE, HEALTH_SERVER  # pylint: disable=global-statement

    logger.info("🚀 Starting MCP Evaluation Server...")
    logger.info("📡 Protocol: Model Context Protocol (MCP) via FastMCP")
    logger.info("📋 Server: mcp-eval-server v0.1.0")

    # Initialize tools and storage after environment variables are loaded
    logger.info("🔧 Initializing tools and storage...")

    # Support custom configuration paths
    models_config_path = os.getenv("MCP_EVAL_MODELS_CONFIG")
    if models_config_path:
        logger.info(f"📄 Using custom models config: {models_config_path}")

    JUDGE_TOOLS = JudgeTools(config_path=models_config_path)
    PROMPT_TOOLS = PromptTools(JUDGE_TOOLS)
    AGENT_TOOLS = AgentTools(JUDGE_TOOLS)
    QUALITY_TOOLS = QualityTools(JUDGE_TOOLS)
    RAG_TOOLS = RAGTools(JUDGE_TOOLS)
    BIAS_TOOLS = BiasTools(JUDGE_TOOLS)
    ROBUSTNESS_TOOLS = RobustnessTools(JUDGE_TOOLS)
    SAFETY_TOOLS = SafetyTools(JUDGE_TOOLS)
    MULTILINGUAL_TOOLS = MultilingualTools(JUDGE_TOOLS)
    PERFORMANCE_TOOLS = PerformanceTools(JUDGE_TOOLS)
    PRIVACY_TOOLS = PrivacyTools(JUDGE_TOOLS)
    WORKFLOW_TOOLS = WorkflowTools(JUDGE_TOOLS, PROMPT_TOOLS, AGENT_TOOLS, QUALITY_TOOLS)
    CALIBRATION_TOOLS = CalibrationTools(JUDGE_TOOLS)

    # Initialize caching and storage
    EVALUATION_CACHE = EvaluationCache()
    JUDGE_CACHE = JudgeResponseCache()
    BENCHMARK_CACHE = BenchmarkCache()
    RESULTS_STORE = ResultsStore()

    # Mark storage as ready
    mark_storage_ready()

    # Log environment configuration
    logger.info("🔧 Environment Configuration:")
    env_vars = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "AZURE_OPENAI_API_KEY": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "not set"),
        "AZURE_DEPLOYMENT_NAME": os.getenv("AZURE_DEPLOYMENT_NAME", "not set"),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "AWS_ACCESS_KEY_ID": bool(os.getenv("AWS_ACCESS_KEY_ID")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "WATSONX_API_KEY": bool(os.getenv("WATSONX_API_KEY")),
        "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID", "not set"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "not set"),
        "DEFAULT_JUDGE_MODEL": os.getenv("DEFAULT_JUDGE_MODEL", "not set"),
    }
    for var, value in env_vars.items():
        if var in ["AZURE_OPENAI_ENDPOINT", "AZURE_DEPLOYMENT_NAME", "WATSONX_PROJECT_ID", "OLLAMA_BASE_URL", "DEFAULT_JUDGE_MODEL"]:
            logger.info(f"   📊 {var}: {value}")
        else:
            status = "✅" if value else "❌"
            logger.info(f"   {status} {var}: {'configured' if value else 'not set'}")

    # Log judge initialization and test connectivity
    available_judges = JUDGE_TOOLS.get_available_judges()
    logger.info(f"⚖️  Loaded {len(available_judges)} judge models: {available_judges}")

    # Test judge connectivity and log detailed status with endpoints
    for judge_name in available_judges:
        info = JUDGE_TOOLS.get_judge_info(judge_name)
        provider = info.get("provider", "unknown")
        model_name = info.get("model_name", "N/A")

        # Get detailed configuration for each judge
        judge_instance = JUDGE_TOOLS.judges.get(judge_name)
        endpoint_info = ""

        if provider == "openai" and hasattr(judge_instance, "client"):
            base_url = str(judge_instance.client.base_url) if judge_instance.client.base_url else "https://api.openai.com/v1"
            endpoint_info = f" → {base_url}"
        elif provider == "azure":
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "not configured")
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "not configured")
            endpoint_info = f" → {endpoint} (deployment: {deployment})"
        elif provider == "anthropic":
            endpoint_info = " → https://api.anthropic.com"
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Test OLLAMA connectivity for status display
            try:
                # Third-Party
                import aiohttp  # pylint: disable=import-outside-toplevel

                async def test_ollama(test_url, aiohttp_module):  # pylint: disable=redefined-outer-name
                    try:
                        timeout = aiohttp_module.ClientTimeout(total=2)
                        async with aiohttp_module.ClientSession(timeout=timeout) as session:
                            async with session.get(f"{test_url}/api/tags") as response:
                                return response.status == 200
                    except Exception:
                        return False

                is_connected = await test_ollama(base_url, aiohttp)
                status = "🟢 connected" if is_connected else "🔴 not reachable"
                endpoint_info = f" → {base_url} ({status})"
            except Exception:
                endpoint_info = f" → {base_url} (🔴 not reachable)"
        elif provider == "bedrock":
            region = os.getenv("AWS_REGION", "us-east-1")
            endpoint_info = f" → AWS Bedrock ({region})"
        elif provider == "gemini":
            endpoint_info = " → Google AI Studio"
        elif provider == "watsonx":
            watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            project_id = os.getenv("WATSONX_PROJECT_ID", "not configured")
            endpoint_info = f" → {watsonx_url} (project: {project_id})"

        logger.info(f"   📊 {judge_name} ({provider}): {model_name}{endpoint_info}")

    # Log tool categories
    logger.info("🛠️  Tool categories:")
    logger.info("   • 4 Judge tools (evaluate, compare, rank, reference)")
    logger.info("   • 4 Prompt tools (clarity, consistency, completeness, relevance)")
    logger.info("   • 4 Agent tools (tool usage, task completion, reasoning, benchmarks)")
    logger.info("   • 3 Quality tools (factuality, coherence, toxicity)")
    logger.info("   • 8 RAG tools (retrieval, context, grounding, hallucination, coverage, citations, chunks, benchmarks)")
    logger.info("   • 6 Bias & Fairness tools (demographic, representation, equity, cultural, linguistic, intersectional)")
    logger.info("   • 5 Robustness tools (adversarial, sensitivity, injection, distribution, consistency)")
    logger.info("   • 4 Safety & Alignment tools (harmful content, instruction following, refusal, value alignment)")
    logger.info("   • 4 Multilingual tools (translation quality, cross-lingual consistency, cultural adaptation, language mixing)")
    logger.info("   • 4 Performance tools (latency, efficiency, throughput, memory)")
    logger.info("   • 8 Privacy tools (PII detection, data minimization, consent compliance, anonymization, leakage detection)")
    logger.info("   • 3 Workflow tools (suites, execution, comparison)")
    logger.info("   • 2 Calibration tools (agreement, optimization)")
    logger.info("   • 4 Server tools (management, statistics, health)")

    # Test primary judge with a simple evaluation if available
    primary_judge = os.getenv("DEFAULT_JUDGE_MODEL", "gpt-4o-mini")
    logger.info(f"🎯 Primary judge selection: {primary_judge}")

    if primary_judge in available_judges:
        try:
            logger.info(f"🧪 Testing primary judge: {primary_judge}")

            # Perform actual inference test
            criteria = [{"name": "helpfulness", "description": "Response helpfulness", "scale": "1-5", "weight": 1.0}]
            rubric = {"criteria": [], "scale_description": {"1": "Poor", "5": "Excellent"}}

            result = await JUDGE_TOOLS.evaluate_response(response="Hi, tell me about this model in one sentence.", criteria=criteria, rubric=rubric, judge_model=primary_judge)

            logger.info(f"✅ Primary judge {primary_judge} inference test successful - Score: {result['overall_score']:.2f}")

            # Log the model's actual response reasoning (truncated)
            if "reasoning" in result and result["reasoning"]:
                for criterion, reasoning in result["reasoning"].items():
                    truncated = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                    logger.info(f"   💬 Model reasoning ({criterion}): {truncated}")

            # Mark judge tools as ready after successful primary judge test
            mark_judge_tools_ready()
        except Exception as e:
            logger.warning(f"⚠️  Primary judge {primary_judge} test failed: {e}")
            # Still mark as ready - server can function with fallback or rule-based judges
            mark_judge_tools_ready()
    elif available_judges:
        fallback = available_judges[0]
        logger.info(f"💡 Primary judge not available, using fallback: {fallback}")

        # Test fallback judge
        try:
            criteria = [{"name": "helpfulness", "description": "Response helpfulness", "scale": "1-5", "weight": 1.0}]
            rubric = {"criteria": [], "scale_description": {"1": "Poor", "5": "Excellent"}}

            result = await JUDGE_TOOLS.evaluate_response(response="Hi, tell me about this model in one sentence.", criteria=criteria, rubric=rubric, judge_model=fallback)

            logger.info(f"✅ Fallback judge {fallback} test successful - Score: {result['overall_score']:.2f}")

            # Log the model's actual response reasoning (truncated)
            if "reasoning" in result and result["reasoning"]:
                for criterion, reasoning in result["reasoning"].items():
                    truncated = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                    logger.info(f"   💬 Model reasoning ({criterion}): {truncated}")

            # Mark judge tools as ready after successful fallback judge test
            mark_judge_tools_ready()
        except Exception as e:
            logger.warning(f"⚠️  Fallback judge {fallback} test failed: {e}")
            # Still mark as ready - server can function with rule-based judges
            mark_judge_tools_ready()
    else:
        logger.warning("⚠️  No judges available, but server can still function for non-LLM evaluations")
        # Mark judge tools as ready (even if no LLM judges available, rule-based judges can work)
        mark_judge_tools_ready()

    # Start health check server
    try:
        HEALTH_SERVER = await start_health_server()
    except Exception as e:
        logger.warning(f"⚠️  Could not start health check server: {e}")
        HEALTH_SERVER = None

    # Mark server as fully ready
    mark_ready()

    logger.info("🎯 Server ready for MCP client connections")
    logger.info("💡 Connect via: python -m mcp_eval_server.server")


async def cleanup():
    """Cleanup resources on shutdown."""
    global HEALTH_SERVER  # pylint: disable=global-statement
    
    logger.info("🛑 Shutting down MCP Evaluation Server...")
    
    # Cleanup health server
    if HEALTH_SERVER:
        await stop_health_server()
    
    logger.info("✅ Cleanup complete")


# Register lifespan handlers
@mcp.lifespan()
async def lifespan():
    """Lifespan context manager for startup and shutdown."""
    await initialize_tools()
    yield
    await cleanup()


def main():
    """Main server entry point."""
    # Run the FastMCP server
    mcp.run()


if __name__ == "__main__":
    main()

# Made with Bob
