"""
CIPHER Phase 9 — Prompt Evolution Learning Loop.

PromptEvolver reads rewards_log.csv after every EVOLVE_EVERY_N episodes,
extracts data-driven heuristics, and appends them to RED and BLUE prompt
files.  All file writes are atomic (.tmp → rename).  The class has no
side-effects at import time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class PromptEvolver:
    """
    Closes the training loop by converting reward statistics into prompt
    heuristics that the LLM agents can act on in future episodes.

    Design constraints
    ------------------
    * No I/O at import or __init__ time.
    * should_evolve() returns False if rewards_log.csv has < EVOLVE_EVERY_N rows.
    * evolve() is idempotent: running twice on the same data produces no
      duplicate rules in prompt files.
    * All file writes use .tmp -> rename for atomicity.
    * Heuristic extraction is deterministic given the same DataFrame.
    * Each rule *type* appears at most once per prompt file (keyword-fingerprint
      deduplication).  Newer data replaces the old rule in-place.
    """

    PROMPTS_DIR: Path = Path("cipher/agents/prompts")
    EVOLUTION_LOG: Path = Path("prompt_evolution_log.jsonl")
    EVOLVE_EVERY_N: int = 5

    # Prompt files updated for each team (5b: added red_analyst and blue_deception_architect)
    _RED_PROMPTS: tuple[str, ...] = (
        "red_planner.txt",
        "red_operative.txt",
        "red_exfiltrator.txt",
        "red_analyst.txt",
    )
    _BLUE_PROMPTS: tuple[str, ...] = (
        "blue_surveillance.txt",
        "blue_threat_hunter.txt",
        "blue_forensics.txt",
        "blue_deception_architect.txt",
    )

    _SECTION_HEADER: str = "\n\n## LEARNED HEURISTICS (Auto-evolved)\n"

    # One canonical keyword per rule type.  Each keyword identifies a unique
    # rule "slot" in the prompt file — at most one rule containing this keyword
    # may exist at a time.  When a newer rule arrives it replaces the old line.
    _RULE_TYPE_KEYWORDS: tuple[str, ...] = (
        "aborting",
        "exfiltrat",
        "stealth score",
        "complexity multiplier",
        "dead drop",
        "false positive penalty",
        "honeypot trigger",
        "response speed",
        "emergent actions",  # 5a: tracks emergent action effectiveness slot
        "strategy is failing",  # 5c: stagnation detector slot
    )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def should_evolve(self, episode_number: int) -> bool:
        """
        Return True iff *episode_number* is a multiple of EVOLVE_EVERY_N
        AND rewards_log.csv contains at least EVOLVE_EVERY_N rows.
        """
        if episode_number % self.EVOLVE_EVERY_N != 0:
            return False
        df = self._load_recent_episodes(self.EVOLVE_EVERY_N)
        return len(df) >= self.EVOLVE_EVERY_N

    def evolve(self, episode_number: int) -> dict:
        """
        Main entry point.

        Reads rewards_log.csv, extracts heuristics, updates prompt files,
        and appends one line to prompt_evolution_log.jsonl.

        Returns
        -------
        dict with keys:
            evolution_number, red_rules_added, blue_rules_added,
            red_rules, blue_rules
        """
        df = self._load_recent_episodes(30)

        red_rules = self._extract_red_heuristics(df)
        blue_rules = self._extract_blue_heuristics(df)

        red_added = 0
        for filename in self._RED_PROMPTS:
            if self._update_prompt(filename, red_rules):
                red_added += len(red_rules)
        # De-duplicate count: count unique new rules written (not per-file)
        # The spec says red_rules_added = number of distinct rules added.
        # We track per-filename; a rule is "added" if it was new to ≥1 file.
        # Simplification: count rules that were new to the first file written.
        red_added = self._count_new_rules(self._RED_PROMPTS[0], red_rules)

        blue_added = self._count_new_rules(self._BLUE_PROMPTS[0], blue_rules)
        for filename in self._BLUE_PROMPTS:
            self._update_prompt(filename, blue_rules)

        self._log_evolution(episode_number, red_rules, blue_rules)

        # Re-read evolution log to get the number just written
        summary = self.get_evolution_summary()
        evo_number = summary["total_evolutions"]

        return {
            "evolution_number": evo_number,
            "red_rules_added": red_added,
            "blue_rules_added": blue_added,
            "red_rules": red_rules,
            "blue_rules": blue_rules,
        }

    def get_evolution_summary(self) -> dict:
        """
        Parse prompt_evolution_log.jsonl and return aggregate statistics.

        Returns
        -------
        {
            'total_evolutions': int,
            'total_red_rules': int,
            'total_blue_rules': int,
            'log': list[dict],
        }
        """
        if not self.EVOLUTION_LOG.exists():
            return {
                "total_evolutions": 0,
                "total_red_rules": 0,
                "total_blue_rules": 0,
                "log": [],
            }

        entries: list[dict] = []
        for raw in self.EVOLUTION_LOG.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass

        return {
            "total_evolutions": len(entries),
            "total_red_rules": sum(e.get("red_rules_count", 0) for e in entries),
            "total_blue_rules": sum(e.get("blue_rules_count", 0) for e in entries),
            "log": entries,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_recent_episodes(self, n: int = 30) -> pd.DataFrame:
        """
        Load the last *n* rows from rewards_log.csv.

        Returns an empty DataFrame (not raising) if the file is absent or
        unreadable.
        """
        csv_path = Path("rewards_log.csv")
        if not csv_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(csv_path)
            for col in (
                "red_total", "blue_total", "red_exfil", "red_stealth",
                "red_complexity", "red_abort_penalty", "red_honeypot_penalty",
                "blue_detection", "blue_speed", "blue_fp_penalty",
                "blue_honeypot_rate", "blue_graph_reconstruction",
                "red_complexity_multiplier", "red_unique_nodes",
                "red_drops_written", "red_traps_placed", "red_context_resets",
                "oversight_red_adj", "oversight_blue_adj",
                "red_emergent_bonus", "blue_emergent_bonus",
            ):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.tail(n).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _extract_red_heuristics(self, df: pd.DataFrame) -> list[str]:
        """
        Analyse *df* and return a list of ``LEARNED: …`` rule strings for
        RED agents.

        Returns ``[]`` if *df* has fewer than 5 rows.
        """
        if len(df) < 5:
            return []

        rules: list[str] = []

        top_q = df[df["red_total"] >= df["red_total"].quantile(0.75)]
        bottom_q = df[df["red_total"] <= df["red_total"].quantile(0.25)]

        # Rule 1 — Suspicion / stealth management
        if "red_stealth" in df.columns and len(top_q) >= 2:
            mean_stealth = top_q["red_stealth"].mean()
            safe_threshold = round(1.0 - mean_stealth + 0.05, 2)
            safe_threshold = min(0.80, max(0.45, safe_threshold))
            rules.append(
                f"LEARNED: Top-quartile episodes maintained stealth score "
                f"{mean_stealth:.2f}. "
                f"WAIT or use false_trail trap when suspicion exceeds "
                f"{safe_threshold}."
            )

        # Rule 2 — Abort rate
        abort_rate = (df["terminal_reason"] == "aborted").mean()
        if abort_rate > 0.35:
            rules.append(
                f"LEARNED: You are aborting in {abort_rate * 100:.0f}% of episodes. "
                f"Continue the operation unless suspicion exceeds 0.75. "
                f"Aborting costs -0.30."
            )
        elif abort_rate < 0.10 and len(df) >= 10:
            rules.append(
                "LEARNED: Low abort rate detected. Remember: aborting early is "
                "better than triggering honeypots. Abort if suspicion exceeds "
                "0.80 and no exfil yet."
            )

        # Rule 3 — Exfiltration success rate
        exfil_rate = (df["red_exfil"] > 0).mean()
        if exfil_rate < 0.25:
            rules.append(
                f"LEARNED: Exfiltration succeeds in only "
                f"{exfil_rate * 100:.0f}% of episodes. "
                f"Upon reaching zone 3, EXFILTRATE immediately rather than "
                f"reading files first."
            )

        # Rule 4 — Complexity multiplier
        if "red_complexity_multiplier" in df.columns and len(top_q) >= 2:
            top_complexity = top_q["red_complexity_multiplier"].mean()
            if top_complexity > 1.5:
                rules.append(
                    f"LEARNED: Top-quartile episodes achieve complexity "
                    f"multiplier {top_complexity:.2f}x by traversing multiple "
                    f"zones. Prioritize crossing zone boundaries before "
                    f"exfiltrating."
                )

        # Rule 5 — Dead drop spam in bottom quartile
        if "red_drops_written" in df.columns and len(bottom_q) >= 2:
            bad_drops = bottom_q["red_drops_written"].mean()
            if bad_drops > 2.0:
                rules.append(
                    f"LEARNED: Bottom-quartile episodes write "
                    f"{bad_drops:.1f} dead drops on average. Write at most "
                    f"1-2 focused dead drops. Excessive drops hurt "
                    f"memory_efficiency_score."
                )

        # 5a — Emergent action effectiveness
        if "red_emergent_bonus" in df.columns:
            emergent_eps = df[df["red_emergent_bonus"] > 0]
            non_emergent = df[df["red_emergent_bonus"] == 0]
            if len(emergent_eps) >= 2 and len(non_emergent) >= 2:
                em_mean = emergent_eps["red_total"].mean()
                no_em_mean = non_emergent["red_total"].mean()
                if em_mean > no_em_mean:
                    rules.append(
                        f"LEARNED: Episodes using emergent actions scored "
                        f"{em_mean:.2f} avg vs {no_em_mean:.2f} without. "
                        "Use emergent actions when stuck or under pressure — "
                        "they significantly improve outcomes."
                    )
                elif em_mean < no_em_mean - 0.2:
                    rules.append(
                        "LEARNED: Emergent actions have underperformed standard "
                        "actions recently. Use them only when truly stuck, not "
                        "as a first resort."
                    )

        # 5c — Stagnation detector: same terminal reason 3+ times in a row
        if "terminal_reason" in df.columns and len(df) >= 3:
            last_3_reasons = df["terminal_reason"].tail(3).tolist()
            if len(set(last_3_reasons)) == 1:
                repeated_reason = last_3_reasons[0]
                rules.append(
                    f"LEARNED: Your last 3 episodes ended the same way: "
                    f"'{repeated_reason}'. Your current strategy is failing — "
                    "you MUST try a different approach. Consider emergent "
                    "actions, different zone crossing sequences, or alternative "
                    "paths you have not tried."
                )

        return rules

    def _extract_blue_heuristics(self, df: pd.DataFrame) -> list[str]:
        """
        Analyse *df* and return a list of ``LEARNED: …`` rule strings for
        BLUE agents.

        Returns ``[]`` if *df* has fewer than 5 rows.
        """
        if len(df) < 5:
            return []

        rules: list[str] = []

        # Rule 1 — False positive discipline
        if "blue_fp_penalty" in df.columns:
            fp_mean = df["blue_fp_penalty"].mean()
            if fp_mean > 0.5:
                rules.append(
                    f"LEARNED: Mean false positive penalty is {fp_mean:.2f}. "
                    f"Investigate only nodes with anomaly scores above 0.6. "
                    f"Precision over recall."
                )

        # Rule 2 — Honeypot placement
        if "blue_honeypot_rate" in df.columns:
            hp_mean = df["blue_honeypot_rate"].mean()
            if hp_mean < 0.2:
                rules.append(
                    f"LEARNED: Honeypot trigger rate is only "
                    f"{hp_mean * 100:.0f}%. Place honeypots on zone 1→2 and "
                    f"zone 2→3 boundary nodes for 3x higher trigger probability."
                )

        # Rule 3 — Early anomaly analysis
        if "blue_speed" in df.columns:
            speed_mean = df["blue_speed"].mean()
            if speed_mean < 0.10:
                rules.append(
                    f"LEARNED: Response speed bonus averages {speed_mean:.3f}. "
                    f"Issue ANALYZE_ANOMALY within the first 3 steps to "
                    f"capture early detection bonuses."
                )

        # 5a — Emergent action effectiveness for BLUE
        if "blue_emergent_bonus" in df.columns:
            emergent_eps = df[df["blue_emergent_bonus"] > 0]
            non_emergent = df[df["blue_emergent_bonus"] == 0]
            if len(emergent_eps) >= 2 and len(non_emergent) >= 2:
                em_mean = emergent_eps["blue_total"].mean()
                no_em_mean = non_emergent["blue_total"].mean()
                if em_mean > no_em_mean:
                    rules.append(
                        f"LEARNED: Episodes using emergent actions scored "
                        f"{em_mean:.2f} avg vs {no_em_mean:.2f} without. "
                        "Use emergent detection/investigation actions when "
                        "standard methods are failing to find RED."
                    )

        # 5c — Stagnation detector for BLUE
        if "terminal_reason" in df.columns and len(df) >= 3:
            last_3_reasons = df["terminal_reason"].tail(3).tolist()
            # BLUE stagnates when RED keeps winning (exfiltration_complete)
            if all(r == "exfiltration_complete" for r in last_3_reasons):
                rules.append(
                    "LEARNED: RED has completed exfiltration in the last 3 "
                    "episodes. Your current detection strategy is failing — "
                    "try placing honeypots at different zone boundaries, "
                    "using emergent surveillance techniques, or coordinating "
                    "INVESTIGATE_NODE on high-value path nodes earlier."
                )

        return rules

    def _update_prompt(self, filename: str, new_rules: list[str]) -> bool:
        """
        Update ``cipher/agents/prompts/{filename}`` with *new_rules*.

        Strategy (keyword-fingerprint deduplication)
        --------------------------------------------
        For each incoming rule, we look for a canonical keyword that identifies
        its *type* (e.g. "aborting", "stealth score").  If the file already
        contains a rule line with that keyword, the old line is replaced with
        the new one.  If no existing rule shares the keyword, the new rule is
        appended.  This ensures each rule *type* appears at most once per file
        and values stay current as more episodes accumulate.

        Writes atomically via .tmp -> rename.

        Returns True if the file was modified, False otherwise.
        """
        path = self.PROMPTS_DIR / filename
        if not path.exists():
            return False

        current = path.read_text(encoding="utf-8")

        # Ensure section header exists
        if self._SECTION_HEADER.strip() not in current:
            current += self._SECTION_HEADER

        lines = current.splitlines(keepends=True)
        modified = False

        for rule in new_rules:
            rule_lower = rule.lower()

            # Find which keyword fingerprint this rule carries (if any)
            matched_keyword: str | None = None
            for kw in self._RULE_TYPE_KEYWORDS:
                if kw in rule_lower:
                    matched_keyword = kw
                    break

            if matched_keyword is not None:
                # Look for an existing rule line with the same keyword
                replaced = False
                for i, line in enumerate(lines):
                    if line.startswith("- LEARNED:") and matched_keyword in line.lower():
                        if line.strip() != f"- {rule}":
                            lines[i] = f"- {rule}\n"
                            modified = True
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"- {rule}\n")
                    modified = True
            else:
                # No fingerprint match: fall back to exact-string dedup
                rule_line = f"- {rule}\n"
                if rule_line not in lines and f"- {rule}" not in "".join(lines):
                    lines.append(rule_line)
                    modified = True

        if modified:
            new_content = "".join(lines)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(new_content, encoding="utf-8")
            tmp.replace(path)

        return modified

    def _count_new_rules(self, filename: str, rules: list[str]) -> int:
        """
        Count how many of *rules* would cause a modification in *filename*.

        A rule causes a modification if either:
        - its keyword fingerprint is not yet present in the file, OR
        - its keyword fingerprint is present but the rule text differs
          (i.e. it would trigger a replacement).

        Used to report accurate ``*_rules_added`` counts without a second
        write pass.
        """
        path = self.PROMPTS_DIR / filename
        if not path.exists():
            return len(rules)
        current = path.read_text(encoding="utf-8")
        current_lines = current.splitlines()

        count = 0
        for rule in rules:
            rule_lower = rule.lower()
            matched_keyword: str | None = None
            for kw in self._RULE_TYPE_KEYWORDS:
                if kw in rule_lower:
                    matched_keyword = kw
                    break

            if matched_keyword is not None:
                existing = next(
                    (ln for ln in current_lines
                     if ln.startswith("- LEARNED:") and matched_keyword in ln.lower()),
                    None,
                )
                # New rule OR replacement of a different value => counts as modified
                if existing is None or existing.strip() != f"- {rule}":
                    count += 1
            else:
                if f"- {rule}" not in current:
                    count += 1
        return count

    def _log_evolution(
        self,
        episode: int,
        red_rules: list[str],
        blue_rules: list[str],
    ) -> None:
        """
        Append one JSONL line to *EVOLUTION_LOG*.

        The evolution_number field auto-increments based on how many lines
        already exist in the log.
        """
        # Determine next evolution number
        current_count = 0
        if self.EVOLUTION_LOG.exists():
            for raw in self.EVOLUTION_LOG.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    current_count += 1

        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "episode": episode,
            "evolution_number": current_count + 1,
            "red_rules_count": len(red_rules),
            "blue_rules_count": len(blue_rules),
            "red_rules": red_rules,
            "blue_rules": blue_rules,
        }

        with open(self.EVOLUTION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
