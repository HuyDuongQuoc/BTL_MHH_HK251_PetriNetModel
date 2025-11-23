#!/usr/bin/env python3
"""
petri_analyzer.py

Implements:
  - PNML parsing for 1-safe P/T nets (Task 1)
  - Explicit (BFS) reachability (Task 2)
  - BDD-based symbolic reachability using dd.autoref.BDD (Task 3)
  - ILP + BDD deadlock detection using PuLP (Task 4)
  - Linear optimization over reachable markings (Task 5, ILP + BDD filter)

This is educational reference code for your assignment.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Iterable, Optional

import xml.etree.ElementTree as ET

# Optional dependencies
try:
    # pip install dd
    from dd.autoref import BDD  # type: ignore
except ImportError:  # pragma: no cover
    BDD = None  # type: ignore

try:
    # pip install pulp
    import pulp  # type: ignore
except ImportError:  # pragma: no cover
    pulp = None  # type: ignore


# ---------- Basic Petri net data structures ----------
@dataclass
class Place:
    pid: str          # PNML id
    name: str         # human-readable name
    index: int        # index in places list (0..n-1)


@dataclass
class Transition:
    tid: str          # PNML id
    name: str
    pre: Set[int] = field(default_factory=set)   # preset: indices of input places
    post: Set[int] = field(default_factory=set)  # postset: indices of output places

    def __repr__(self) -> str:
        return f"Transition({self.tid}, pre={sorted(self.pre)}, post={sorted(self.post)})"


@dataclass
class PetriNet:
    places: List[Place]
    transitions: List[Transition]
    initial: Tuple[int, ...]          # initial marking, 0/1 per place
    place_index: Dict[str, int]       # PNML place id -> index
    trans_index: Dict[str, int]       # PNML transition id -> index

    @property
    def num_places(self) -> int:
        return len(self.places)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    # ----- marking utilities -----

    def initial_bits(self) -> int:
        """Encode initial marking as an integer bitset (1 bit per place)."""
        bits = 0
        for i, v in enumerate(self.initial):
            if v:
                bits |= (1 << i)
        return bits

    def bits_to_marking(self, bits: int) -> Tuple[int, ...]:
        """Convert bitset -> marking vector."""
        return tuple(1 if (bits >> i) & 1 else 0 for i in range(self.num_places))

    def marking_to_bits(self, marking: Iterable[int]) -> int:
        """Convert marking vector -> bitset."""
        bits = 0
        for i, v in enumerate(marking):
            if v:
                bits |= (1 << i)
        return bits

    # ----- transition firing semantics (1-safe P/T net) -----

    def is_enabled_bits(self, marking_bits: int, t: Transition) -> bool:
        """
        Check if transition t is enabled at marking represented by bitset.

        For 1-safe nets:
          - each pre place must have a token
          - we forbid putting a second token in a place
        """
        # All pre places must contain a token.
        for p in t.pre:
            if ((marking_bits >> p) & 1) == 0:
                return False

        # For 1-safe nets, do not allow putting a second token in a place.
        for p in t.post:
            if p not in t.pre and ((marking_bits >> p) & 1) == 1:
                return False
        return True

    def fire_bits(self, marking_bits: int, t: Transition) -> int:
        """Fire t from marking_bits and return successor marking_bits."""
        new_bits = marking_bits
        # consume tokens from pre \ post
        for p in t.pre:
            if p not in t.post:
                new_bits &= ~(1 << p)
        # produce tokens in post \ pre
        for p in t.post:
            if p not in t.pre:
                new_bits |= (1 << p)
        # pre ∩ post keep their tokens
        return new_bits


# ---------- PNML parsing (Task 1) ----------

def _local_tag(tag: str) -> str:
    """Strip XML namespace from a tag name if present."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, local_name: str) -> Optional[ET.Element]:
    """Find direct child with given local (namespace-free) name."""
    for c in elem:
        if _local_tag(c.tag) == local_name:
            return c
    return None


def parse_pnml(filename: str) -> PetriNet:
    """
    Parse a 1-safe P/T net from a PNML file.
    Supports:
      - <place id=...> with <name><text>..</text></name> and optional <initialMarking>
      - <transition id=...> with optional <name><text>..</text></name>
      - <arc id=... source=... target=...> between places and transitions
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # 1) collect places
    places: List[Place] = []
    place_index: Dict[str, int] = {}

    for place_el in root.iter():
        if _local_tag(place_el.tag) != "place":
            continue
        pid = place_el.attrib.get("id")
        if pid is None:
            raise ValueError("Place without id in PNML")
        # name label
        name_el = _find_child(place_el, "name")
        text = None
        if name_el is not None:
            text_el = _find_child(name_el, "text")
            if text_el is not None and text_el.text is not None:
                text = text_el.text.strip()
        name = text if text else pid
        idx = len(places)
        places.append(Place(pid=pid, name=name, index=idx))
        place_index[pid] = idx

    # 2) collect transitions
    transitions: List[Transition] = []
    trans_index: Dict[str, int] = {}
    for trans_el in root.iter():
        if _local_tag(trans_el.tag) != "transition":
            continue
        tid = trans_el.attrib.get("id")
        if tid is None:
            raise ValueError("Transition without id in PNML")
        name_el = _find_child(trans_el, "name")
        text = None
        if name_el is not None:
            text_el = _find_child(name_el, "text")
            if text_el is not None and text_el.text is not None:
                text = text_el.text.strip()
        name = text if text else tid
        idx = len(transitions)
        transitions.append(Transition(tid=tid, name=name))
        trans_index[tid] = idx

    if not places or not transitions:
        raise ValueError("PNML file seems to contain no places or no transitions")

    # 3) initial marking for each place (assume 0 if missing)
    initial = [0] * len(places)
    for place_el in root.iter():
        if _local_tag(place_el.tag) != "place":
            continue
        pid = place_el.attrib.get("id")
        if pid is None or pid not in place_index:
            continue
        idx = place_index[pid]
        im_el = _find_child(place_el, "initialMarking")
        if im_el is None:
            continue
        text_el = _find_child(im_el, "text")
        if text_el is None or text_el.text is None:
            continue
        try:
            val = int(text_el.text.strip())
        except ValueError:
            val = 0
        # Enforce 1-safety: any positive number is treated as 1
        initial[idx] = 1 if val > 0 else 0

    # 4) arcs: fill pre/post sets
    for arc_el in root.iter():
        if _local_tag(arc_el.tag) != "arc":
            continue
        src = arc_el.attrib.get("source")
        tgt = arc_el.attrib.get("target")
        if src is None or tgt is None:
            raise ValueError("Arc without source or target in PNML")

        is_src_place = src in place_index
        is_tgt_place = tgt in place_index
        is_src_trans = src in trans_index
        is_tgt_trans = tgt in trans_index

        if is_src_place and is_tgt_trans:
            # place -> transition: input arc
            p_idx = place_index[src]
            t_idx = trans_index[tgt]
            transitions[t_idx].pre.add(p_idx)
        elif is_src_trans and is_tgt_place:
            # transition -> place: output arc
            t_idx = trans_index[src]
            p_idx = place_index[tgt]
            transitions[t_idx].post.add(p_idx)
        else:
            # ignore other arc shapes for this assignment
            continue

    net = PetriNet(
        places=places,
        transitions=transitions,
        initial=tuple(initial),
        place_index=place_index,
        trans_index=trans_index,
    )
    return net


# ---------- Explicit reachability (Task 2) ----------

def explicit_reachability(net: PetriNet) -> Tuple[Set[int], Dict[int, List[Tuple[int, int]]]]:
    """
    Compute reachability graph using plain BFS.

    Returns:
        visited: set of reachable markings (bit-encoded)
        edges: dict m -> list of (transition_index, successor_marking_bits)
    """
    start = net.initial_bits()
    visited: Set[int] = {start}
    edges: Dict[int, List[Tuple[int, int]]] = {}

    from collections import deque
    queue: deque[int] = deque([start])
    while queue:
        m = queue.popleft()
        outgoing: List[Tuple[int, int]] = []
        for ti, t in enumerate(net.transitions):
            if net.is_enabled_bits(m, t):
                m2 = net.fire_bits(m, t)
                outgoing.append((ti, m2))
                if m2 not in visited:
                    visited.add(m2)
                    queue.append(m2)
        edges[m] = outgoing
    return visited, edges


# ---------- BDD-based symbolic reachability (Task 3) ----------

@dataclass
class BDDReachabilityResult:
    bdd: "BDD"
    reachable: "BDD.Function"  # type: ignore
    x_vars: List[str]          # state vars x0..x{n-1}
    xp_vars: List[str]         # next-state vars xp0..xp{n-1}


def build_symbolic_transition_relation(
    net: PetriNet,
    bdd: "BDD",
    x_vars: List[str],
    xp_vars: List[str],
) -> "BDD.Function":  # type: ignore
    """
    Build the transition relation T(x, x') as a BDD.

    For each transition t, and for each place i, we add local constraints:

        if i in pre \\ post:     x_i & ~x'_i
        if i in post \\ pre:     ~x_i & x'_i
        if i in pre ∩ post:      x_i &  x'_i
        if i not in pre ∪ post:  (x_i &  x'_i) | (~x_i & ~x'_i)

    The overall relation is the disjunction over all transitions.
    """
    n = net.num_places
    assert len(x_vars) == len(xp_vars) == n

    clauses = []
    for t in net.transitions:
        # build conjunction for this transition
        conj = bdd.true
        for i in range(n):
            x = bdd.var(x_vars[i])
            xp = bdd.var(xp_vars[i])
            in_pre = i in t.pre
            in_post = i in t.post

            if in_pre and not in_post:
                local = x & ~xp
            elif in_post and not in_pre:
                local = ~x & xp
            elif in_pre and in_post:
                local = x & xp
            else:
                # unchanged
                local = (x & xp) | (~x & ~xp)
            conj &= local
        clauses.append(conj)

    # Disjunction of all transition relations
    T = bdd.false
    for c in clauses:
        T |= c
    return T


def symbolic_reachability(net: PetriNet) -> BDDReachabilityResult:
    """
    Compute set of reachable markings using BDDs.

    Forward fixpoint:

      R_0(x)    = encoding of initial marking
      R_{k+1}   = R_k ∨ post(R_k)
      post(R)(x') = ∃x. R(x) ∧ T(x, x')

    where T(x,x') is the transition relation.
    """
    if BDD is None:
        raise RuntimeError("dd.autoref.BDD is not available. Install package 'dd' first.")

    n = net.num_places
    bdd = BDD()
    # variable names: x0..x{n-1} for current, xp0..xp{n-1} for next
    x_vars = [f"x{i}" for i in range(n)]
    xp_vars = [f"xp{i}" for i in range(n)]
    bdd.declare(*(x_vars + xp_vars))

    def marking_to_bdd(bits: int, prefix: str) -> "BDD.Function":  # type: ignore
        assignment = {}
        for i in range(n):
            vname = f"{prefix}{i}"
            assignment[vname] = bool((bits >> i) & 1)
        return bdd.cube(assignment)    # conjunction of literals

    # initial set R_0
    m0_bits = net.initial_bits()
    R = marking_to_bdd(m0_bits, "x")

    # transition relation
    T = build_symbolic_transition_relation(net, bdd, x_vars, xp_vars)

    x_set = set(x_vars)

    while True:
        # relational product: ∃x. R(x) ∧ T(x, x')
        RT = R & T
        post = bdd.exist(x_set, RT)  # BDD over xp-vars only
        # rename xp -> x
        rename = {xp_vars[i]: x_vars[i] for i in range(n)}
        post_x = bdd.let(rename, post)
        new = post_x & ~R
        if new == bdd.false:
            break
        R |= new

    return BDDReachabilityResult(bdd=bdd, reachable=R, x_vars=x_vars, xp_vars=xp_vars)


def bdd_marking_membership(res: BDDReachabilityResult, marking_bits: int) -> bool:
    """Check if a marking (bit-encoded) is in the reachable set BDD."""
    bdd = res.bdd
    assignment = {}
    for i, vname in enumerate(res.x_vars):
        assignment[vname] = bool((marking_bits >> i) & 1)
    cube = bdd.cube(assignment)
    return (cube & res.reachable) != bdd.false


def count_reachable_markings(res: BDDReachabilityResult) -> int:
    """Return the number of satisfying assignments of the reachable-set BDD."""
    n = len(res.x_vars)
    return int(res.bdd.count(res.reachable, nvars=n))


def format_marking_with_places(net: PetriNet, marking: Tuple[int, ...]) -> str:
    """
    Return a human-readable description of a marking:
      - 0/1 vector
      - list of places that contain tokens
    """
    assert len(marking) == net.num_places
    active_indices = [i for i, v in enumerate(marking) if v]
    if active_indices:
        place_list = ", ".join(f"{net.places[i].name}" for i in active_indices)
    else:
        place_list = "∅ (no tokens)"
    return f"{marking}  |  tokens at: {place_list}"


# ---------- ILP + BDD: deadlock detection (Task 4) ----------

def find_deadlock_ilp_bdd(net: PetriNet, bdd_res: BDDReachabilityResult) -> Optional[Tuple[int, ...]]:
    """
    Find a reachable deadlock marking using ILP + BDD filtering.

    ILP model (binary variables m_p ∈ {0,1} for each place p):

        For each transition t with preset P_t:
            sum_{p ∈ P_t} m_p <= |P_t| - 1   (no transition fully enabled)

    We then use the BDD to keep only reachable solutions.
    """
    if pulp is None:
        raise RuntimeError("PuLP is not available. Install package 'pulp' first.")

    prob = pulp.LpProblem("deadlock_search", pulp.LpMinimize)
    # binary variable for each place
    m_vars = [
        pulp.LpVariable(f"m_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(net.num_places)
    ]
    # trivial objective (any will do)
    prob += pulp.lpSum(m_vars)

    # deadlock constraints
    for t in net.transitions:
        if not t.pre:
            # transitions without preset would always be enabled;
            continue
        prob += pulp.lpSum(m_vars[i] for i in t.pre) <= len(t.pre) - 1

    # iterative solving with BDD filter
    while True:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            return None  # no ILP-feasible deadlock

        # extract marking
        bits = 0
        for i, var in enumerate(m_vars):
            val = pulp.value(var)
            if val is None:
                val = 0
            if val >= 0.5:
                bits |= (1 << i)

        if bdd_marking_membership(bdd_res, bits):
            # Found reachable deadlock
            return net.bits_to_marking(bits)

        # Otherwise, block this marking and continue
        terms = []
        for i, var in enumerate(m_vars):
            if ((bits >> i) & 1) == 1:
                terms.append(1 - var)
            else:
                terms.append(var)
        prob += pulp.lpSum(terms) >= 1


# ---------- ILP + BDD: optimization over reachable markings (Task 5) ----------

def optimize_over_reachable(
    net: PetriNet,
    bdd_res: BDDReachabilityResult,
    weights: List[float],
    sense: str = "max",
) -> Optional[Tuple[Tuple[int, ...], float]]:
    """
    Maximize or minimize a linear expression c^T m over reachable markings.

    Args:
        net: Petri net.
        bdd_res: reachable-set BDD info from symbolic_reachability.
        weights: list of coefficients c_p (same length as number of places).
        sense: 'max' or 'min'.
    """
    if pulp is None:
        raise RuntimeError("PuLP is not available. Install package 'pulp' first.")

    if len(weights) != net.num_places:
        raise ValueError("weights length must equal number of places")

    if sense not in {"max", "min"}:
        raise ValueError("sense must be 'max' or 'min'")

    prob = pulp.LpProblem(
        "reachable_optimization",
        pulp.LpMaximize if sense == "max" else pulp.LpMinimize,
    )
    m_vars = [
        pulp.LpVariable(f"m_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(net.num_places)
    ]

    # objective: c^T m
    prob += pulp.lpSum(weights[i] * m_vars[i] for i in range(net.num_places))

    best_marking: Optional[Tuple[int, ...]] = None
    best_val: Optional[float] = None

    while True:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            break

        # extract marking
        bits = 0
        for i, var in enumerate(m_vars):
            val = pulp.value(var)
            if val is None:
                val = 0
            if val >= 0.5:
                bits |= (1 << i)
        marking = net.bits_to_marking(bits)

        if bdd_marking_membership(bdd_res, bits):
            # feasible reachable solution
            obj_val = float(pulp.value(prob.objective))
            best_marking = marking
            best_val = obj_val
            break

        # otherwise, block this unreachable marking
        terms = []
        for i, var in enumerate(m_vars):
            if ((bits >> i) & 1) == 1:
                terms.append(1 - var)
            else:
                terms.append(var)
        prob += pulp.lpSum(terms) >= 1

    if best_marking is None:
        return None
    return best_marking, best_val


# ---------- CLI: run ALL tasks and write single report ----------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Petri net analyzer: runs Tasks 1–5 and writes a report.",
    )
    parser.add_argument("pnml", help="Input PNML file")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        help="Weights for optimization over reachable markings (Task 5). "
             "If omitted, defaults to 1 for every place.",
    )
    parser.add_argument(
        "--sense",
        choices=["max", "min"],
        default="max",
        help="Optimization sense for Task 5 (default: max)",
    )
    args = parser.parse_args()

    # Prepare report file
    base_name = os.path.splitext(os.path.basename(args.pnml))[0]
    report_path = base_name + ".txt"

    output_lines: List[str] = []

    def out(line: str = "") -> None:
        """Print to terminal AND append to the report buffer."""
        print(line)
        output_lines.append(line)

    # ---------------- Task 1 – Net summary ----------------
    out("=" * 70)
    out("PETRI NET ANALYSIS REPORT")
    out("=" * 70)
    out(f"PNML file : {args.pnml}")
    out("")

    out("Task 1 – Net summary")
    out("-" * 70)

    try:
        net = parse_pnml(args.pnml)
    except Exception as e:
        out(f"[ERROR] Failed to parse PNML file: {e}")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        out(f"\n[INFO] Report written to {report_path}")
        return

    out(f"  Number of places      : {net.num_places}")
    out(f"  Number of transitions : {net.num_transitions}")
    out(f"  Initial marking M0    : {net.initial}")
    out("")
    out("  Places (index : id / name)")
    for p in net.places:
        out(f"    {p.index:2d}: {p.pid} / {p.name}")
    out("")

    # ---------------- Task 2 – Explicit reachability ----------------
    out("Task 2 – Explicit reachability (BFS)")
    out("-" * 70)
    t0 = time.time()
    visited, edges = explicit_reachability(net)
    t1 = time.time()
    out(f"  |R| (reachable markings) : {len(visited)}")
    out(f"  BFS exploration time      : {t1 - t0:.6f} seconds")
    out("")
    out("  Sample reachable markings (up to 10):")
    sorted_markings = sorted(visited)
    max_show = min(10, len(sorted_markings))
    for idx in range(max_show):
        bits = sorted_markings[idx]
        m = net.bits_to_marking(bits)
        out(f"    M{idx}: {format_marking_with_places(net, m)}")
    if len(sorted_markings) > max_show:
        out(f"    ... ({len(sorted_markings) - max_show} more markings not shown)")
    out("")

    # ---------------- Task 3 – BDD reachability ----------------
    out("Task 3 – BDD-based symbolic reachability")
    out("-" * 70)
    bdd_res: Optional[BDDReachabilityResult] = None
    if BDD is None:
        out("[ERROR] Package 'dd' is not available. Install it with 'pip install dd'.")
        out("        Tasks 3–5 (BDD, ILP+BDD) cannot be executed.")
    else:
        try:
            t0 = time.time()
            bdd_res = symbolic_reachability(net)
            t1 = time.time()
            num = count_reachable_markings(bdd_res)
            out(f"  |R| (reachable markings) : {num}")
            out(f"  BDD fixpoint time        : {t1 - t0:.6f} seconds")
            out("")
            out("  Sanity check:")
            m0_bits = net.initial_bits()
            in_R0 = bdd_marking_membership(bdd_res, m0_bits)
            out(f"    - Initial marking M0 is in reachable set: {in_R0}")
        except RuntimeError as e:
            out(f"[ERROR] {e}")
            bdd_res = None
    out("")

    # Only continue with Tasks 4–5 if BDD was successful
    if bdd_res is None:
        out("Skipping Task 4 and Task 5 because BDD reachability is unavailable.")
    else:
        # ---------------- Task 4 – Deadlock detection ----------------
        out("Task 4 – Deadlock detection (ILP + BDD)")
        out("-" * 70)
        if pulp is None:
            out("[ERROR] Package 'pulp' is not available. Install it with 'pip install pulp'.")
            out("        Deadlock detection cannot be executed.")
        else:
            try:
                dl = find_deadlock_ilp_bdd(net, bdd_res)
                if dl is None:
                    out("  Result : No reachable deadlock marking found.")
                else:
                    out("  Result : Reachable deadlock marking found")
                    out("           (no transition is enabled at this marking).")
                    out("")
                    out("  m_deadlock =")
                    out(f"    {format_marking_with_places(net, dl)}")
            except RuntimeError as e:
                out(f"[ERROR] {e}")
        out("")

        # ---------------- Task 5 – Optimization over reachable markings ----------------
        out("Task 5 – Optimization over reachable markings (ILP + BDD)")
        out("-" * 70)
        if pulp is None:
            out("[ERROR] Package 'pulp' is not available. Install it with 'pip install pulp'.")
            out("        Optimization over reachable markings cannot be executed.")
        else:
            # Determine weights
            if args.weights is None:
                weights = [1.0] * net.num_places
                out("  Weights not provided on command line.")
                out("  Using default weights: c_p = 1 for every place p.")
            else:
                if len(args.weights) != net.num_places:
                    out(
                        f"[ERROR] Expected {net.num_places} weights, "
                        f"but got {len(args.weights)}."
                    )
                    weights = None
                else:
                    weights = args.weights
                    out("  Weights (c_p for each place index p):")
                    out("    " + ", ".join(f"{w:.3g}" for w in weights))
            out(f"  Optimization sense        : {args.sense}")
            out("")

            if weights is not None:
                try:
                    res = optimize_over_reachable(net, bdd_res, weights, sense=args.sense)
                    if res is None:
                        out("  Result : No reachable marking satisfies the ILP model.")
                    else:
                        best_marking, obj_val = res
                        out("  Result : Optimal reachable marking found.")
                        out(f"           Objective value = {obj_val}")
                        out("")
                        out("  m_opt =")
                        out(f"    {format_marking_with_places(net, best_marking)}")
                except RuntimeError as e:
                    out(f"[ERROR] {e}")
        out("")

    # Write report file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    out(f"[INFO] Report written to {report_path}")


if __name__ == "__main__":
    main()
0