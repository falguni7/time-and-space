"""The Orchestrator: one Agent Framework workflow that routes every request through the agents.

    request ─► Router ─┬─► Facts ─► Stats ─► Explainer ⇄ Verifier ⇄ Critic ─┬─► Personas (fan-out) ─► Collector
                       │                         (retry once on either)   └─► Template ─┘
                       ├─► Recap   (match summary + top moments, verified, per profile)
                       └─► Analyst (MCP tools over the metrics cube, verified)

    async with Orchestrator() as orch:
        story = await orch.explain("R01_EAS-NOR-M008", DEFAULT_PROFILES)
        recap = await orch.recap("R01_EAS-NOR", profile)
        answer = await orch.ask("Which team presses highest up the pitch?")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Self

from agent_framework import Case, Default, Executor, WorkflowBuilder, WorkflowContext, handler

from ..engine import moments as engine_moments
from .facts import build_fact_sheet, template_version
from .llm import LLM, chat_client
from .personas import DEFAULT_PROFILES, Profile
from .schemas import Explanation, FactSheet, Feed, FeedItem, MomentStory, Version
from .selector import select_moments
from .team import MAX_ATTEMPTS, Team, check_explanation
from .verify import Check

# --------------------------------------------------------------------------- messages


@dataclass
class ExplainRequest:
    moment_id: str


@dataclass
class RecapRequest:
    match_id: str
    top_moments: int = 5


@dataclass
class AskRequest:
    question: str


@dataclass
class Draft:
    sheet: FactSheet
    attempt: int = 0
    explanation: Explanation | None = None
    feedback: str | None = None
    issues: list[str] = field(default_factory=list)
    ok: bool = False
    rejected: list[str] = field(default_factory=list)


@dataclass
class Neutral:
    sheet: FactSheet
    version: Version


@dataclass
class Personalised:
    sheet: FactSheet
    neutral: Version
    version: Version


# --------------------------------------------------------------------------- executors


class Router(Executor):
    @handler
    async def explain(self, req: ExplainRequest, ctx: WorkflowContext[ExplainRequest]) -> None:
        await ctx.send_message(req)

    @handler
    async def recap(self, req: RecapRequest, ctx: WorkflowContext[RecapRequest]) -> None:
        await ctx.send_message(req)

    @handler
    async def ask(self, req: AskRequest, ctx: WorkflowContext[AskRequest]) -> None:
        await ctx.send_message(req)


class Facts(Executor):
    """Code, not a model: the trusted fact sheet from the MCP tool functions."""

    @handler
    async def run(self, req: ExplainRequest, ctx: WorkflowContext[FactSheet]) -> None:
        await ctx.send_message(build_fact_sheet(req.moment_id))


class Stats(Executor):
    def __init__(self, team: Team):
        super().__init__(id="stats")
        self.team = team

    @handler
    async def run(self, sheet: FactSheet, ctx: WorkflowContext[Draft]) -> None:
        sheet = sheet.model_copy(update={"context_facts": await self.team.context(sheet)})
        await ctx.send_message(Draft(sheet=sheet))


class Explainer(Executor):
    def __init__(self, team: Team):
        super().__init__(id="explainer")
        self.team = team

    @handler
    async def run(self, draft: Draft, ctx: WorkflowContext[Draft]) -> None:
        exp, _ = await self.team.explain(draft.sheet, draft.feedback)
        await ctx.send_message(Draft(sheet=draft.sheet, attempt=draft.attempt + 1, explanation=exp,
                                     issues=draft.issues, rejected=draft.rejected))


def _checked(draft: Draft, check: Check) -> Draft:
    return Draft(sheet=draft.sheet, attempt=draft.attempt, explanation=draft.explanation,
                 feedback=None if check.ok else check.feedback(), issues=check.issues, ok=check.ok,
                 rejected=draft.rejected + ([] if check.ok else check.issues))


class Verifier(Executor):
    """Rule-based: numbers and names must trace back to the fact sheet."""

    @handler
    async def run(self, draft: Draft, ctx: WorkflowContext[Draft]) -> None:
        if draft.explanation is None:
            await ctx.send_message(Draft(sheet=draft.sheet, attempt=MAX_ATTEMPTS, issues=["no model answer"],
                                         rejected=draft.rejected))
            return
        await ctx.send_message(_checked(draft, check_explanation(draft.sheet, draft.explanation)))


class Critic(Executor):
    """A model that reads the text against the facts for invented causes and wrong directions.
    Only drafts that passed the rule Verifier get here; it can reject, never approve numbers."""

    def __init__(self, team: Team):
        super().__init__(id="critic")
        self.team = team

    @handler
    async def run(self, draft: Draft, ctx: WorkflowContext[Draft]) -> None:
        e = draft.explanation
        problems = await self.team.critique(draft.sheet.to_prompt(), f"{e.headline}\n{e.body}")
        await ctx.send_message(_checked(draft, Check(not problems, problems)))


class Approve(Executor):
    def __init__(self, label: str):
        super().__init__(id="approve")
        self.label = label

    @handler
    async def run(self, draft: Draft, ctx: WorkflowContext[Neutral]) -> None:
        e = draft.explanation
        v = Version(profile_id="neutral", headline=e.headline, body=e.body, source=self.label, verified=True,
                    attempts=draft.attempt, rejected=draft.rejected)
        await ctx.send_message(Neutral(draft.sheet, v))


class Template(Executor):
    @handler
    async def run(self, draft: Draft, ctx: WorkflowContext[Neutral]) -> None:
        v = template_version(draft.sheet, issues=draft.issues)
        await ctx.send_message(Neutral(draft.sheet, v.model_copy(update={"rejected": draft.rejected})))


class PersonaHub(Executor):
    @handler
    async def run(self, neutral: Neutral, ctx: WorkflowContext[Neutral, MomentStory]) -> None:
        await ctx.send_message(neutral)


class Persona(Executor):
    def __init__(self, team: Team, profile: Profile):
        super().__init__(id=f"persona:{profile.id}")
        self.team = team
        self.profile = profile

    @handler
    async def run(self, neutral: Neutral, ctx: WorkflowContext[Personalised]) -> None:
        v = await self.team.personalise(neutral.sheet, neutral.version, self.profile)
        await ctx.send_message(Personalised(neutral.sheet, neutral.version, v))


class Collector(Executor):
    @handler
    async def one(self, part: Personalised, ctx: WorkflowContext[None, MomentStory]) -> None:
        await self.run([part], ctx)

    @handler
    async def run(self, parts: list[Personalised], ctx: WorkflowContext[None, MomentStory]) -> None:
        first = parts[0]
        await ctx.yield_output(MomentStory(facts=first.sheet, neutral=first.neutral,
                                           versions=[p.version for p in parts]))


class NeutralOnly(Executor):
    @handler
    async def run(self, neutral: Neutral, ctx: WorkflowContext[None, MomentStory]) -> None:
        await ctx.yield_output(MomentStory(facts=neutral.sheet, neutral=neutral.version))


class RecapWriter(Executor):
    def __init__(self, team: Team, profiles: list[Profile]):
        super().__init__(id="recap")
        self.team = team
        self.profiles = profiles

    @handler
    async def run(self, req: RecapRequest, ctx: WorkflowContext[None, list]) -> None:
        cards = engine_moments.load_moments(req.match_id)
        top = sorted(cards, key=lambda c: -c["importance"])[:req.top_moments]
        sheets = [build_fact_sheet(c["moment_id"]) for c in sorted(top, key=lambda c: c["frame"])]
        await ctx.yield_output([await self.team.recap(req.match_id, sheets, p) for p in self.profiles])


class Analyst(Executor):
    def __init__(self, team: Team):
        super().__init__(id="analyst")
        self.team = team

    @handler
    async def run(self, req: AskRequest, ctx: WorkflowContext[None, Version]) -> None:
        await ctx.yield_output(await self.team.ask(req.question))


# --------------------------------------------------------------------------- graph


def build_workflow(team: Team, profiles: list[Profile]):
    router, facts, stats = Router(id="router"), Facts(id="facts"), Stats(team)
    explainer, verifier, critic = Explainer(team), Verifier(id="verifier"), Critic(team)
    approve, template, hub = Approve(team.llm.label), Template(id="template"), PersonaHub(id="personas")
    recap, analyst = RecapWriter(team, profiles or [Profile(id="neutral", kind="analyst")]), Analyst(team)

    b = WorkflowBuilder(name="presslens", description="PressLens orchestrator", start_executor=router)
    b.add_switch_case_edge_group(router, [
        Case(lambda m: isinstance(m, ExplainRequest), facts),
        Case(lambda m: isinstance(m, RecapRequest), recap),
        Default(analyst),
    ])
    b.add_chain([facts, stats, explainer, verifier])
    b.add_switch_case_edge_group(verifier, [
        Case(lambda d: d.ok, critic),
        Case(lambda d: d.attempt < MAX_ATTEMPTS, explainer),
        Default(template),
    ])
    b.add_switch_case_edge_group(critic, [
        Case(lambda d: d.ok, approve),
        Case(lambda d: d.attempt < MAX_ATTEMPTS, explainer),
        Default(template),
    ])
    b.add_edge(approve, hub)
    b.add_edge(template, hub)
    if len(profiles) == 1:  # fan-out/fan-in groups need at least two branches
        b.add_chain([hub, Persona(team, profiles[0]), Collector(id="collector")])
    elif profiles:
        personas = [Persona(team, p) for p in profiles]
        b.add_fan_out_edges(hub, personas)
        b.add_fan_in_edges(personas, Collector(id="collector"))
    else:
        b.add_edge(hub, NeutralOnly(id="neutral_only"))
    return b.build()


class Orchestrator:
    """Owns the chat client, the MCP connection and the agent team for a series of requests."""

    def __init__(self, llm: LLM | None = None, use_mcp: bool = True):
        self.llm = llm or chat_client()
        self.use_mcp = use_mcp
        self._mcp = None
        self.team: Team | None = None

    async def __aenter__(self) -> Self:
        if self.use_mcp:
            from ..mcp_server.connect import metrics_tool
            self._mcp = metrics_tool()
            await self._mcp.__aenter__()
        self.team = Team(self.llm, self._mcp)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(*exc)

    async def _run(self, request, profiles: list[Profile]):
        t0 = time.time()
        result = await build_workflow(self.team, profiles).run(request)
        outputs = result.get_outputs()
        if not outputs:
            raise RuntimeError(f"workflow produced no output for {request}")
        self.last_seconds = time.time() - t0
        return outputs[0]

    async def explain(self, moment_id: str, profiles: list[Profile] | None = None) -> MomentStory:
        return await self._run(ExplainRequest(moment_id), DEFAULT_PROFILES if profiles is None else profiles)

    async def feeds(self, match_id: str, profiles: list[Profile], parallel: int = 4) -> list[Feed]:
        """Each viewer's personal feed. A moment picked by several viewers is explained once, with
        a persona version for each of them that knows why they got it."""
        picks = {p.id: select_moments(match_id, p) for p in profiles}
        wanted: dict[str, list[Profile]] = {}
        for p in profiles:
            for pick in picks[p.id]:
                wanted.setdefault(pick.moment_id, []).append(p.model_copy(update={"note": "; ".join(pick.reasons)}))
        gate = asyncio.Semaphore(parallel)

        async def one(moment_id: str) -> MomentStory:
            async with gate:
                return await self.explain(moment_id, wanted[moment_id])
        stories = dict(zip(wanted, await asyncio.gather(*(one(m) for m in wanted))))
        feeds = []
        for p in profiles:
            items = []
            for pick in picks[p.id]:
                story = stories[pick.moment_id]
                version = next(v for v in story.versions if v.profile_id == p.id)
                items.append(FeedItem(moment_id=pick.moment_id, type=pick.type, clock=story.facts.clock,
                                      reasons=pick.reasons, version=version, neutral=story.neutral))
            feeds.append(Feed(profile_id=p.id, match_id=match_id, items=items))
        return feeds

    async def recap(self, match_id: str, profiles: list[Profile] | None = None, top_moments: int = 5) -> list[Version]:
        return await self._run(RecapRequest(match_id, top_moments), profiles or [Profile(id="neutral", kind="analyst")])

    async def ask(self, question: str) -> Version:
        return await self._run(AskRequest(question), [])
