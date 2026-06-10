"""Realistic customer-operations workflow backed by a remote LLM URL.

This example is intentionally richer than ``foo2.py``: it includes service
pre-processing, four-way routing, optional risk-lead escalation, QA review,
and looped batch handling.

**Pattern**: ``CustomerOpsWorkflow`` subclasses ``Workflow`` so it can be
converted to LangGraph, CrewAI, or AutoGen via the CLI::

    choreo_mini -f examples/customer_ops_url.py -b langgraph \\
                -o output/customer_ops_langgraph.py

To run directly against a real OpenAI-compatible endpoint::

    python examples/customer_ops_url.py

Set ``CHOREO_LLM_URL``, ``CHOREO_API_TOKEN``, and ``CHOREO_LLM_MODEL`` to
skip the interactive prompts.  Leave ``CHOREO_LLM_URL`` empty to use the
built-in demo handlers (no network calls required).
"""

import os
import sys
import time
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

# Ensure local package imports work when run from examples/.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from choreo_mini.core.llm import CustomLLM, Message
from choreo_mini.core.nodes import AgentNode, ServiceNode
from choreo_mini.core.workflow import Workflow


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 45.0
EXAMPLE_BATCH = (
    "channel=email|customer_tier=vip|issue=duplicate charge and refund request|region=us; "
    "channel=phone|customer_tier=standard|issue=mobile app crash during checkout|region=eu; "
    "channel=chat|customer_tier=standard|issue=where is my delayed shipment|region=apac; "
    "channel=email|customer_tier=standard|issue=thinking about canceling annual plan|region=us"
)


SYSTEM_PROMPTS = {
    "Router": (
        "You are a customer-operations router. Reply with exactly one label: "
        "billing, technical, logistics, or retention."
    ),
    "BillingDesk": "You resolve payment disputes, refunds, invoices, and charge problems.",
    "TechnicalDesk": "You resolve product defects, outages, crashes, and incident symptoms.",
    "LogisticsDesk": "You resolve shipping delays, fulfillment issues, and delivery exceptions.",
    "RetentionDesk": "You handle churn risk, cancellations, renewals, and save offers.",
    "RiskLead": "You review high-risk cases and produce a concise escalation decision.",
    "QAReviewer": "You rewrite internal notes into a clear customer-ready response.",
}


# ---------------------------------------------------------------------------
# Remote LLM config
# ---------------------------------------------------------------------------

@dataclass
class RemoteLLMConfig:
    endpoint: str
    api_token: str
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    api_style: str = "responses"
    max_retries: int = 2
    retry_base_delay: float = 1.0


def _normalize_endpoint(raw_url: str):
    url = raw_url.strip().strip('"').strip("'").rstrip("/")
    if not url:
        raise ValueError("LLM URL cannot be empty")

    if "://" not in url:
        url = f"https://{url}"

    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(
            "LLM URL must start with http:// or https:// "
            "(for example https://api.openai.com/v1/responses)"
        )

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(
            "LLM URL is missing a hostname "
            "(for example https://api.openai.com/v1/responses)"
        )

    if url.endswith("/responses"):
        return url, "responses"
    if url.endswith("/chat/completions"):
        return url, "chat_completions"
    if url.endswith("/v1"):
        return f"{url}/responses", "responses"
    return f"{url}/v1/responses", "responses"


# ---------------------------------------------------------------------------
# Remote LLM helpers
# ---------------------------------------------------------------------------

def _serialize_messages(
    system_prompt: str,
    prompt: str,
    context: Optional[List[Message]] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context:
        for message in context:
            if message.role in {"user", "assistant", "system"} and message.content:
                messages.append({"role": message.role, "content": message.content})
        return messages
    messages.append({"role": "user", "content": prompt})
    return messages


def _serialize_responses_input(
    system_prompt: str,
    prompt: str,
    context: Optional[List[Message]] = None,
) -> str:
    lines: List[str] = [f"system: {system_prompt}"]
    if context:
        for message in context:
            if message.role in {"user", "assistant", "system"} and message.content:
                lines.append(f"{message.role}: {message.content}")
    else:
        lines.append(f"user: {prompt}")
    return "\n".join(lines)


def _extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    chunks.append(str(item["text"]))
            if chunks:
                return "\n".join(chunks).strip()

        text = choices[0].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    raise ValueError(f"Unexpected LLM response payload: {payload}")


def _call_remote_llm(
    config: RemoteLLMConfig,
    system_prompt: str,
    prompt: str,
    context: Optional[List[Message]] = None,
    **kwargs: Any,
) -> str:
    model = kwargs.get("model", config.model)
    temperature = kwargs.get("temperature", 0.2)
    if config.api_style == "chat_completions":
        payload = {
            "model": model,
            "messages": _serialize_messages(system_prompt, prompt, context=context),
            "temperature": temperature,
        }
    else:
        payload = {
            "model": model,
            "input": _serialize_responses_input(system_prompt, prompt, context=context),
            "temperature": temperature,
        }
    headers = {
        "Authorization": f"Bearer {config.api_token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=kwargs.get("timeout", config.timeout)) as client:
        attempts = max(0, config.max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = client.post(config.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return _extract_response_text(response.json())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt >= attempts:
                    if exc.response.status_code == 429:
                        raise RuntimeError(
                            "LLM API rate limit hit (429 Too Many Requests). "
                            "Wait a bit, reduce request frequency, or switch to a key/model "
                            "with higher limits."
                        ) from exc
                    raise

                retry_after_raw = exc.response.headers.get("retry-after")
                if retry_after_raw is not None:
                    try:
                        delay = max(float(retry_after_raw), 0.0)
                    except ValueError:
                        delay = config.retry_base_delay * (2 ** (attempt - 1))
                else:
                    delay = config.retry_base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    "Could not connect to LLM endpoint. "
                    f"Tried URL: {config.endpoint}. "
                    "Check hostname, internet/VPN, and proxy/firewall settings."
                ) from exc


def _latest_user_text(prompt: str, context: Optional[List[Message]] = None) -> str:
    if context:
        user_messages = [
            message.content
            for message in context
            if message.role == "user" and message.content
        ]
        if user_messages:
            return user_messages[-1]
    return prompt.splitlines()[-1] if prompt else ""


# ---------------------------------------------------------------------------
# Demo handlers (no network calls — used when demo_mode=True)
# ---------------------------------------------------------------------------

def _demo_router(prompt: str, context=None, **kwargs) -> str:
    text = _latest_user_text(prompt, context=context).lower()
    if any(word in text for word in ("refund", "invoice", "charge", "billing", "payment")):
        return "billing"
    if any(word in text for word in ("crash", "error", "bug", "timeout", "checkout")):
        return "technical"
    if any(word in text for word in ("shipment", "delivery", "tracking", "fulfillment", "shipping")):
        return "logistics"
    return "retention"


def _demo_billing(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    return f"Billing desk plan: confirm charges, explain refund path, and log finance notes for {latest}."


def _demo_technical(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    return (
        f"Technical desk plan: gather repro details, reference incident playbook, "
        f"and suggest a stable workaround for {latest}."
    )


def _demo_logistics(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    return (
        f"Logistics desk plan: inspect carrier status, set a delivery expectation, "
        f"and offer shipment recovery steps for {latest}."
    )


def _demo_retention(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    return (
        f"Retention desk plan: acknowledge churn risk, confirm account goals, "
        f"and present a right-sized save offer for {latest}."
    )


def _demo_risk(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    return (
        f"Risk escalation approved: verify identity, document exposure, "
        f"and release the guarded action for {latest}."
    )


def _demo_qa(prompt: str, context=None, **kwargs) -> str:
    latest = _latest_user_text(prompt, context=context)
    if "channel=chat" in latest.lower():
        return f"Customer-ready response: {latest}. Follow-up required within 30 minutes."
    return f"Customer-ready response: {latest}."


_DEMO_HANDLERS = {
    "Router": _demo_router,
    "BillingDesk": _demo_billing,
    "TechnicalDesk": _demo_technical,
    "LogisticsDesk": _demo_logistics,
    "RetentionDesk": _demo_retention,
    "RiskLead": _demo_risk,
    "QAReviewer": _demo_qa,
}


def build_agent_llm(
    agent_name: str,
    system_prompt: str,
    client_config: Optional[RemoteLLMConfig] = None,
    demo_mode: bool = False,
) -> CustomLLM:
    if demo_mode:
        return CustomLLM(_DEMO_HANDLERS[agent_name])
    if client_config is None:
        raise ValueError("client_config is required when demo_mode is False")

    def _generate(prompt: str, context=None, **kwargs) -> str:
        return _call_remote_llm(client_config, system_prompt, prompt, context=context, **kwargs)

    return CustomLLM(_generate)


# ---------------------------------------------------------------------------
# Service functions (pure functions — no Workflow state)
# ---------------------------------------------------------------------------

def parse_cases(raw_batch: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, raw_case in enumerate(raw_batch.split(";"), start=1):
        text = raw_case.strip()
        if not text:
            continue

        fields: Dict[str, str] = {}
        for segment in text.split("|"):
            piece = segment.strip()
            if not piece:
                continue
            if "=" in piece:
                key, value = piece.split("=", 1)
                fields[key.strip().lower()] = value.strip()
            elif "issue" not in fields:
                fields["issue"] = piece

        issue = fields.get("issue", text)
        customer_tier = fields.get("customer_tier", "standard").lower()
        cases.append(
            {
                "case_id": fields.get("case_id", f"CASE-{index:03d}"),
                "channel": fields.get("channel", "email").lower(),
                "customer_tier": customer_tier,
                "issue": issue,
                "region": fields.get("region", "global").lower(),
                "needs_risk_review": customer_tier == "vip"
                or any(word in issue.lower() for word in ("chargeback", "fraud", "abuse")),
            }
        )
    return cases


def prioritize_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked_cases: List[Dict[str, Any]] = []
    for case in cases:
        issue = case["issue"].lower()
        if case["customer_tier"] == "vip" or any(
            word in issue for word in ("chargeback", "fraud", "outage")
        ):
            priority = "urgent"
        elif any(word in issue for word in ("crash", "refund", "delayed", "cancel")):
            priority = "high"
        else:
            priority = "normal"

        ranked_case = dict(case)
        ranked_case["priority"] = priority
        ranked_cases.append(ranked_case)

    priority_order = {"urgent": 0, "high": 1, "normal": 2}
    return sorted(ranked_cases, key=lambda item: (priority_order[item["priority"]], item["case_id"]))


def format_case(case: Dict[str, Any]) -> str:
    return (
        f"case_id={case['case_id']} channel={case['channel']} "
        f"customer_tier={case['customer_tier']} priority={case['priority']} "
        f"region={case['region']} issue={case['issue']}"
    )


# ---------------------------------------------------------------------------
# Workflow subclass (the conversion target)
# ---------------------------------------------------------------------------

class CustomerOpsWorkflow(Workflow):
    """End-to-end customer-operations control tower.

    Agents and service nodes are wired in ``__init__``; the multi-case
    batch-processing loop lives in ``process_batch``.

    This is the *subclass pattern* — it can be run directly **and** compiled
    to LangGraph, CrewAI, or AutoGen with the CLI.
    """

    def __init__(
        self,
        client_config: Optional[RemoteLLMConfig] = None,
        demo_mode: bool = False,
        observability=None,
    ):
        super().__init__("customer_ops_control_tower", enable_profiling=True, observability=observability)

        self.state["batch_number"] = 0
        self.state["last_batch"] = []
        self.state["escalations"] = []
        self.state["follow_ups"] = []

        # Agent nodes
        self.router = AgentNode(
            self,
            "Router",
            role="customer operations triage router",
            llm=build_agent_llm("Router", SYSTEM_PROMPTS["Router"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.billing_desk = AgentNode(
            self,
            "BillingDesk",
            role="billing operations specialist",
            llm=build_agent_llm("BillingDesk", SYSTEM_PROMPTS["BillingDesk"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.technical_desk = AgentNode(
            self,
            "TechnicalDesk",
            role="technical operations specialist",
            llm=build_agent_llm("TechnicalDesk", SYSTEM_PROMPTS["TechnicalDesk"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.logistics_desk = AgentNode(
            self,
            "LogisticsDesk",
            role="logistics operations specialist",
            llm=build_agent_llm("LogisticsDesk", SYSTEM_PROMPTS["LogisticsDesk"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.retention_desk = AgentNode(
            self,
            "RetentionDesk",
            role="retention specialist",
            llm=build_agent_llm("RetentionDesk", SYSTEM_PROMPTS["RetentionDesk"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.risk_lead = AgentNode(
            self,
            "RiskLead",
            role="risk escalation lead",
            llm=build_agent_llm("RiskLead", SYSTEM_PROMPTS["RiskLead"],
                                client_config=client_config, demo_mode=demo_mode),
        )
        self.qa_reviewer = AgentNode(
            self,
            "QAReviewer",
            role="quality reviewer",
            llm=build_agent_llm("QAReviewer", SYSTEM_PROMPTS["QAReviewer"],
                                client_config=client_config, demo_mode=demo_mode),
        )

        # Service nodes (pure-function transforms, no LLM)
        self.case_loader = ServiceNode(self, "CaseLoader", parse_cases)
        self.priority_tagger = ServiceNode(self, "PriorityTagger", prioritize_cases)
        self.case_formatter = ServiceNode(self, "CaseFormatter", format_case)

    def process_batch(self, raw_batch: str) -> List[str]:
        """Parse, prioritise, route, optionally escalate, then QA-review each case.

        Returns a list of customer-ready responses, one per case.
        """
        cases = self.case_loader.execute(self, raw_batch)
        cases = self.priority_tagger.execute(self, cases)
        self.state["last_batch"] = cases
        self.state["batch_number"] = self.state["batch_number"] + 1

        results = []
        for index, case in enumerate(cases, start=1):
            case_summary = self.case_formatter.execute(self, case)
            route = self.send("Router", case_summary).content.strip().lower()

            if route == "billing":
                owner = "BillingDesk"
            elif route == "technical":
                owner = "TechnicalDesk"
            elif route == "logistics":
                owner = "LogisticsDesk"
            else:
                owner = "RetentionDesk"

            draft = self.send(owner, f"case#{index} {case_summary}").content

            if case["needs_risk_review"]:
                draft = self.send(
                    "RiskLead",
                    f"risk review for {owner}: {draft}",
                ).content
                self.state["escalations"] = self.state["escalations"] + [case["case_id"]]

            final = self.send(
                "QAReviewer",
                f"channel={case['channel']} customer-ready response for {owner}: {draft}",
            ).content

            if case["channel"] == "chat" or "follow-up" in final.lower():
                self.state["follow_ups"] = self.state["follow_ups"] + [case["case_id"]]

            results.append(final)

        return results


# ---------------------------------------------------------------------------
# Interactive CLI helper
# ---------------------------------------------------------------------------

def prompt_for_remote_config() -> RemoteLLMConfig:
    endpoint = os.getenv("CHOREO_LLM_URL", "").strip()
    if not endpoint:
        endpoint = input("LLM URL (base URL, /responses, or /chat/completions URL)> ").strip()

    api_token = os.getenv("CHOREO_API_TOKEN", "").strip()
    if not api_token:
        api_token = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_token:
        api_token = getpass("API token> ").strip()

    model = os.getenv("CHOREO_LLM_MODEL", "").strip()
    if not model:
        model = input(f"Model [{DEFAULT_MODEL}]> ").strip() or DEFAULT_MODEL

    retry_count_raw = os.getenv("CHOREO_LLM_MAX_RETRIES", "2").strip()
    retry_delay_raw = os.getenv("CHOREO_LLM_RETRY_BASE_DELAY", "1.0").strip()
    try:
        max_retries = max(0, int(retry_count_raw))
    except ValueError:
        max_retries = 2
    try:
        retry_base_delay = max(0.0, float(retry_delay_raw))
    except ValueError:
        retry_base_delay = 1.0

    try:
        endpoint, api_style = _normalize_endpoint(endpoint)
    except ValueError as exc:
        raise ValueError(f"Invalid LLM URL: {exc}") from exc

    return RemoteLLMConfig(
        endpoint=endpoint,
        api_token=api_token,
        model=model,
        api_style=api_style,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    demo_mode = not os.getenv("CHOREO_LLM_URL", "").strip()
    if demo_mode:
        print("No CHOREO_LLM_URL set — running in demo mode (no network calls).")
        wf = CustomerOpsWorkflow(demo_mode=True)
    else:
        client_config = prompt_for_remote_config()
        wf = CustomerOpsWorkflow(client_config=client_config)

    print("Enter semicolon-separated cases. Example:")
    print(EXAMPLE_BATCH)
    print("Press Enter on an empty line or type 'quit' to stop.\n")

    while True:
        try:
            raw_batch = input("Cases> ")
        except (EOFError, KeyboardInterrupt):
            break

        raw_batch = raw_batch.strip()
        if not raw_batch or raw_batch.lower() == "quit":
            break

        results = wf.process_batch(raw_batch)

        batch_num = wf.state["batch_number"]
        cases = wf.state["last_batch"]
        print(f"\nBatch {batch_num} — {len(cases)} case(s) processed.")
        for case, response in zip(cases, results):
            print(f"  [{case['case_id']}] {response}")
        print(f"  Escalations : {wf.state['escalations']}")
        print(f"  Follow-ups  : {wf.state['follow_ups']}\n")

    if wf.enable_profiling:
        print("Profiling summary:")
        for agent_name, stats in wf.profile_data.items():
            print(
                f"  {agent_name}: calls={stats['calls']} "
                f"latency={stats['total_latency']:.4f}s "
                f"memory={stats['total_memory']}"
            )


if __name__ == "__main__":
    main()
