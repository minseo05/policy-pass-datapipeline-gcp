"""Reference-free RAGAS metrics for live service logs.

The existing src.evaluation.ragas_metrics module is QA-oriented because it needs
reference/ground_truth for context recall. Live service logs usually do not have
references, so this module only runs metrics that can work without a reference:
faithfulness and answer_relevancy.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_ragas_llm = None
_ragas_embeddings = None


def _make_llm():
    global _ragas_llm  # noqa: PLW0603
    if _ragas_llm is not None:
        return _ragas_llm
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
    _ragas_llm = llm_factory("gpt-4o-mini", client=client)
    return _ragas_llm


def _make_embeddings():
    global _ragas_embeddings  # noqa: PLW0603
    if _ragas_embeddings is not None:
        return _ragas_embeddings
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)
    _ragas_embeddings = OpenAIEmbeddings(client=client)
    return _ragas_embeddings


def evaluate_ragas_service(question: str, contexts: list[str], answer: str) -> dict[str, float | None]:
    """Run service-safe RAGAS metrics.

    Returns keys compatible with the normal RAGAS block. context_precision and
    context_recall are intentionally None because they need a reference answer.
    """
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    async def _run_all() -> dict[str, float | None]:
        llm = _make_llm()
        embeddings = _make_embeddings()
        scores: dict[str, float | None] = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        try:
            result = await Faithfulness(llm=llm).ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            scores["faithfulness"] = float(result.value)
        except Exception:
            logger.exception("Service RAGAS Faithfulness failed")

        try:
            result = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
                user_input=question,
                response=answer,
            )
            scores["answer_relevancy"] = float(result.value)
        except Exception:
            logger.exception("Service RAGAS AnswerRelevancy failed")

        return scores

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run_all()).result()
    return asyncio.run(_run_all())
