from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional


# =========================
# Models (Gegevensstructuren)
# =========================

@dataclass
class RoutedMatch:
    kind: str  # 'command' | 'tool'
    name: str
    source_hint: str
    score: int


@dataclass
class PortingModule:
    name: str
    source_hint: str
    responsibility: str


@dataclass
class TurnResult:
    matched_commands: Tuple[str, ...]
    matched_tools: Tuple[str, ...]
    permission_denials: Tuple[str, ...]
    stop_reason: str = "completed"


@dataclass
class RuntimeSession:
    prompt: str
    routed_matches: List[RoutedMatch]
    turn_result: TurnResult
    command_execution_messages: Tuple[str, ...]
    tool_execution_messages: Tuple[str, ...]


# =========================
# Dummy Registry (simuleren van echt systeem)
# =========================

class Command:
    def __init__(self, name: str):
        self.name = name

    def execute(self, prompt: str) -> str:
        return f"[COMMAND:{self.name}] executed with prompt='{prompt}'"


class Tool:
    def __init__(self, name: str):
        self.name = name

    def execute(self, prompt: str) -> str:
        return f"[TOOL:{self.name}] executed with prompt='{prompt}'"


class ExecutionRegistry:
    def __init__(self):
        """Initialiseert lege storage dictionaries."""
        self._commands = {}  
        self._tools = {}     

    def register_command(self, name: str):
        self._commands[name] = Command(name)

    def register_tool(self, name: str):
        self._tools[name] = Tool(name)

    def command(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def build_execution_registry() -> ExecutionRegistry:
    reg = ExecutionRegistry()

    # Voorbeeldcommands registreren
    reg.register_command("search")   
    reg.register_command("analyze")  

    # Voorbeeldtools registreren
    reg.register_tool("python")      
    reg.register_tool("bash")        

    return reg


# =========================
# Query Engine (mock/simulatie)
# =========================

class QueryEnginePort:
    @classmethod
    def from_workspace(cls):
        return cls()

    def submit_message(
        self,
        prompt: str,
        matched_commands: Tuple[str, ...],
        matched_tools: Tuple[str, ...],
        denied_tools: Tuple[str, ...],
    ) -> TurnResult:
        return TurnResult(
            matched_commands=matched_commands,
            matched_tools=matched_tools,
            permission_denials=denied_tools,
            stop_reason="completed"
        )


# =========================
# Router (Routeerder)
# =========================

class PortRouter:
    def route(self, prompt: str, limit: int = 5) -> List[RoutedMatch]:
        tokens = self._tokenize(prompt)

        # Verzamel matches van commands en tools
        matches = (
            self._collect(tokens, PORTED_COMMANDS, "command") +
            self._collect(tokens, PORTED_TOOLS, "tool")
        )

        # Sorteer: eerst op score (aflopend), daarna op type en naam
        matches.sort(key=lambda m: (-m.score, m.kind, m.name))
        return matches[:limit]  # Retourneer alleen top N

    def _tokenize(self, prompt: str) -> set[str]:
        return {
            t.lower()
            for t in prompt.replace("/", " ").replace("-", " ").split()
            if t  # Negeer lege strings
        }

    def _collect(self, tokens, modules, kind):
        results = []
        for m in modules:
            score = self._score(tokens, m)
            if score > 0:  # Alleen opnemen als er minstens 1 overeenkomst is
                results.append(RoutedMatch(kind, m.name, m.source_hint, score))
        return results

    def _score(self, tokens, module: PortingModule) -> int:
        # Combineer alle beschrijvende tekst van module
        text = f"{module.name} {module.source_hint} {module.responsibility}".lower()
        # Tel hoeveel tokens in deze tekst voorkomen
        return sum(1 for t in tokens if t in text)


# =========================
# Executor (Uitvoeringsmotor)
# =========================

class ExecutionEngine:
    def __init__(self, registry: ExecutionRegistry):
        self.registry = registry

    def execute(
        self,
        prompt: str,
        matches: List[RoutedMatch]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:

        command_execs = []  # Storage voor command outputs
        tool_execs = []     # Storage voor tool outputs
        denials = []        # Storage voor geweigerde tools

        for m in matches:
            if m.kind == "command":
                # Zoek het command in registry en voer uit
                cmd = self.registry.command(m.name)
                if cmd:
                    command_execs.append(cmd.execute(prompt))

            elif m.kind == "tool":
                # Simpele permission check: bash tools zijn geweigerd
                if "bash" in m.name.lower():
                    denials.append(m.name)
                    continue  # Skip dit tool (niet uitvoeren)

                # Zoek de tool in registry en voer uit
                tool = self.registry.tool(m.name)
                if tool:
                    tool_execs.append(tool.execute(prompt))

        # Retourneer alles als tuples
        return tuple(command_execs), tuple(tool_execs), tuple(denials)


# =========================
# Runtime (Schone runtime omgeving)
# =========================

class CleanRuntime:
    def __init__(self):
        self.router = PortRouter()                          # Routeert prompts naar matches
        self.engine = QueryEnginePort.from_workspace()      # Verwerkt berichten
        self.registry = build_execution_registry()          # Bevat commands/tools
        self.executor = ExecutionEngine(self.registry)      # Voert matches uit

    def run(self, prompt: str, limit: int = 5) -> RuntimeSession:
        """
        Voert een compleet request uit van begin tot eind.
        
        Wat: Verwerkt een user prompt compleet: routing, execution, en resultaat-compilatie.
        Hoe: 1. Route prompt naar relevante commands/tools
             2. Voer de gevonden matches uit
             3. Verzamel resultaten via QueryEngine
             4. Compileer alles in RuntimeSession en return
        
        Args:
            prompt: De user input/vraag
            limit: Max aantal matches om te routeren (default 5)
            
        Returns:
            RuntimeSession object met compleet resultaat van processing
        """
        # STAP 1: Route de prompt naar relevante matches
        matches = self.router.route(prompt, limit)

        # STAP 2: Voer gevonden matches uit
        command_execs, tool_execs, denials = self.executor.execute(prompt, matches)

        # STAP 3: Verwerk via QueryEngine
        result = self.engine.submit_message(
            prompt,
            matched_commands=tuple(m.name for m in matches if m.kind == "command"),
            matched_tools=tuple(m.name for m in matches if m.kind == "tool"),
            denied_tools=denials
        )

        # STAP 4: Compileer alles in RuntimeSession en retourneer
        return RuntimeSession(
            prompt=prompt,
            routed_matches=matches,
            turn_result=result,
            command_execution_messages=command_execs,
            tool_execution_messages=tool_execs,
        )


# =========================
# Mock Data (modules) - Testgegevens
# =========================


PORTED_COMMANDS = (
    # Command voor zoekopdrachten: zoekt en filtert data
    PortingModule("search", "query engine", "search data"),
    # Command voor data-analyse: analyzeert en verwerkt resultaten
    PortingModule("analyze", "data processing", "analyze results"),
)

PORTED_TOOLS = (
    # Tool voor Python: voert Python-code uit in een interpreter
    PortingModule("python", "execution environment", "run python code"),
    # Tool voor Bash: voert shell/bash commands uit (wordt normaal geweigerd vanwege veiligheid)
    PortingModule("bash", "shell access", "execute shell commands"),
)


# =========================
# Example usage (Voorbeeldgebruik)
# =========================

if __name__ == "__main__":
    # Stap 1: Maak een CleanRuntime instance aan (initialize alle componenten)
    runtime = CleanRuntime()

    # Stap 2: Run met een voorbeeld prompt
    # Deze vraag bevat woorden die matchen met 'search', 'python', 'data processing'
    session = runtime.run("Search how data procssing using python and does python do powervol job?")

    # Stap 3: Print alle informatie uit resultaat
    print("\n--- ROUTED MATCHES ---")
    print("Welke commands/tools zijn gevonden als relevant:")
    for m in session.routed_matches:
        print(f"{m.kind}: {m.name} (relevantie-score={m.score})")

    print("\n--- COMMAND EXECUTION ---")
    print("Output van uitgevoerde commands:")
    for msg in session.command_execution_messages:
        print(msg)

    print("\n--- TOOL EXECUTION ---")
    print("Output van uitgevoerde tools:")
    for msg in session.tool_execution_messages:
        print(msg)

    print("\n--- TURN RESULT ---")
    print("Samenvatting van alles wat gebeurde:")
    print(session.turn_result)
