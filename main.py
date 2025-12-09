"""
Assignment Solution: Petri Net Reachability & Optimization
Fix: Restored missing attributes in BDDManager to fix AttributeError.
"""

from __future__ import annotations
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any
import xml.etree.ElementTree as ET

# Try to import pulp for Task 5
try:
    import pulp
except ImportError:
    pulp = None

# =============================================================================
# PART 1: CUSTOM BDD IMPLEMENTATION
# =============================================================================

class BDDNode:
    def __init__(self, var: int, low: int, high: int):
        self.var = var       
        self.low = low       
        self.high = high     

    def __repr__(self):
        return f"Node(var={self.var}, low={self.low}, high={self.high})"

class BDDManager:
    def __init__(self):
        # 0 is False, 1 is True. Their var index is effectively infinity.
        self.nodes: Dict[int, BDDNode] = {
            0: BDDNode(float('inf'), None, None), 
            1: BDDNode(float('inf'), None, None)
        }
        # --- FIX: Restored these attributes ---
        self.false_node = 0
        self.true_node = 1
        
        self.unique_table: Dict[Tuple[int, int, int], int] = {}
        self.computed_table: Dict[Tuple[Any, ...], int] = {}
        self.next_id = 2
        self.var_map: Dict[str, int] = {}
        self.level_to_name: Dict[int, str] = {}
        self.num_vars = 0

    def declare(self, *names):
        for name in names:
            if name not in self.var_map:
                idx = self.num_vars
                self.var_map[name] = idx
                self.level_to_name[idx] = name
                self.num_vars += 1

    def var(self, name: str) -> int:
        if name not in self.var_map: raise ValueError(f"Var {name} unknown")
        return self.get_node(self.var_map[name], 0, 1)

    def get_node(self, var: int, low: int, high: int) -> int:
        if low == high: return low
        key = (var, low, high)
        if key in self.unique_table: return self.unique_table[key]
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = BDDNode(var, low, high)
        self.unique_table[key] = nid
        return nid

    def ite(self, i: int, t: int, e: int) -> int:
        if i == 1: return t
        if i == 0: return e
        if t == e: return t
        if t == 1 and e == 0: return i
        
        key = ('ite', i, t, e)
        if key in self.computed_table: return self.computed_table[key]
        
        v_i = self.nodes[i].var
        v_t = self.nodes[t].var
        v_e = self.nodes[e].var
        top_var = min(v_i, v_t, v_e)
        
        i_h = self.nodes[i].high if v_i == top_var else i
        i_l = self.nodes[i].low  if v_i == top_var else i
        t_h = self.nodes[t].high if v_t == top_var else t
        t_l = self.nodes[t].low  if v_t == top_var else t
        e_h = self.nodes[e].high if v_e == top_var else e
        e_l = self.nodes[e].low  if v_e == top_var else e
        
        r_h = self.ite(i_h, t_h, e_h)
        r_l = self.ite(i_l, t_l, e_l)
        
        res = self.get_node(top_var, r_l, r_h)
        self.computed_table[key] = res
        return res

    def apply_and(self, u: int, v: int) -> int: return self.ite(u, v, 0)
    def apply_or(self, u: int, v: int)  -> int: return self.ite(u, 1, v)
    def apply_not(self, u: int)         -> int: return self.ite(u, 0, 1)

    def exist(self, vars_to_quantify: Set[str], u: int) -> int:
        if u <= 1: return u
        key = ('exist', frozenset(vars_to_quantify), u)
        if key in self.computed_table: return self.computed_table[key]

        node = self.nodes[u]
        var_name = self.level_to_name[node.var]
        
        high_res = self.exist(vars_to_quantify, node.high)
        low_res  = self.exist(vars_to_quantify, node.low)
        
        if var_name in vars_to_quantify:
            res = self.apply_or(high_res, low_res)
        else:
            res = self.get_node(node.var, low_res, high_res)
        self.computed_table[key] = res
        return res

    def let(self, rename_map: Dict[str, str], u: int) -> int:
        if u <= 1: return u
        key = ('let', tuple(sorted(rename_map.items())), u)
        if key in self.computed_table: return self.computed_table[key]
            
        node = self.nodes[u]
        old_name = self.level_to_name[node.var]
        
        low_res = self.let(rename_map, node.low)
        high_res = self.let(rename_map, node.high)
        
        new_name = rename_map.get(old_name, old_name)
        new_level = self.var_map[new_name]
        
        var_node = self.get_node(new_level, 0, 1)
        res = self.ite(var_node, high_res, low_res)
        self.computed_table[key] = res
        return res

    def evaluate(self, u: int, assignment: Dict[str, int]) -> bool:
        curr = u
        while curr != 0 and curr != 1:
            node = self.nodes[curr]
            name = self.level_to_name[node.var]
            val = assignment.get(name, 0)
            curr = node.high if val == 1 else node.low
        return curr == 1


# =============================================================================
# PART 2: PETRI NET DEFINITIONS
# =============================================================================

@dataclass
class Place:
    pid: str
    name: str
    index: int

@dataclass
class Transition:
    tid: str
    name: str
    pre: Set[int] = field(default_factory=set)
    post: Set[int] = field(default_factory=set)

@dataclass
class PetriNet:
    places: List[Place]
    transitions: List[Transition]
    initial: Tuple[int, ...]
    place_index: Dict[str, int]
    trans_index: Dict[str, int]

    @property
    def num_places(self) -> int: return len(self.places)

    def initial_bits(self) -> int:
        bits = 0
        for i, v in enumerate(self.initial):
            if v: bits |= (1 << i)
        return bits

    def bits_to_marking(self, bits: int) -> Tuple[int, ...]:
        return tuple(1 if (bits >> i) & 1 else 0 for i in range(self.num_places))

    def is_enabled_bits(self, marking_bits: int, t: Transition) -> bool:
        for p in t.pre:
            if ((marking_bits >> p) & 1) == 0: return False
        for p in t.post:
            if p not in t.pre and ((marking_bits >> p) & 1) == 1: return False
        return True

    def fire_bits(self, marking_bits: int, t: Transition) -> int:
        new_bits = marking_bits
        for p in t.pre:
            if p not in t.post: new_bits &= ~(1 << p)
        for p in t.post:
            if p not in t.pre: new_bits |= (1 << p)
        return new_bits

def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def parse_pnml(filename: str) -> PetriNet:
    tree = ET.parse(filename)
    root = tree.getroot()
    places, transitions = [], []
    place_index, trans_index = {}, {}

    for child in root.iter():
        if _local_tag(child.tag) == 'place':
            pid = child.attrib.get('id')
            name = pid
            idx = len(places)
            places.append(Place(pid, name, idx))
            place_index[pid] = idx

    for child in root.iter():
        if _local_tag(child.tag) == 'transition':
            tid = child.attrib.get('id')
            idx = len(transitions)
            transitions.append(Transition(tid, tid))
            trans_index[tid] = idx

    initial = [0] * len(places)
    for child in root.iter():
        if _local_tag(child.tag) == 'place':
            pid = child.attrib.get('id')
            if pid not in place_index: continue
            idx = place_index[pid]
            for sub in child:
                if _local_tag(sub.tag) == 'initialMarking':
                    for t in sub:
                        if _local_tag(t.tag) == 'text' and t.text:
                            try:
                                if int(t.text) > 0: initial[idx] = 1
                            except: pass

    for child in root.iter():
        if _local_tag(child.tag) == 'arc':
            src, tgt = child.attrib.get('source'), child.attrib.get('target')
            if src in place_index and tgt in trans_index:
                transitions[trans_index[tgt]].pre.add(place_index[src])
            elif src in trans_index and tgt in place_index:
                transitions[trans_index[src]].post.add(place_index[tgt])

    return PetriNet(places, transitions, tuple(initial), place_index, trans_index)


# =============================================================================
# PART 3: REACHABILITY ALGORITHMS
# =============================================================================

def explicit_reachability(net: PetriNet) -> Set[int]:
    start = net.initial_bits()
    visited = {start}
    import collections
    queue = collections.deque([start])
    
    while queue:
        m = queue.popleft()
        for t in net.transitions:
            if net.is_enabled_bits(m, t):
                m_next = net.fire_bits(m, t)
                if m_next not in visited:
                    visited.add(m_next)
                    queue.append(m_next)
    return visited

@dataclass
class BDDResult:
    manager: BDDManager
    reachable_node: int
    x_vars: List[str]
    xp_vars: List[str]

def symbolic_reachability(net: PetriNet) -> BDDResult:
    bdd = BDDManager()
    n = net.num_places
    vars_ordered = []
    for i in range(n):
        vars_ordered.append(f"x{i}")
        vars_ordered.append(f"xp{i}")
    bdd.declare(*vars_ordered)
    x_vars = [f"x{i}" for i in range(n)]
    xp_vars = [f"xp{i}" for i in range(n)]
    
    R = bdd.true_node
    for i in range(n):
        var_node = bdd.var(x_vars[i])
        val = net.initial[i]
        lit = var_node if val == 1 else bdd.apply_not(var_node)
        R = bdd.apply_and(R, lit)
        
    T = bdd.false_node
    for t in net.transitions:
        trans_bdd = bdd.true_node
        for i in range(n):
            x = bdd.var(x_vars[i])
            xp = bdd.var(xp_vars[i])
            is_pre, is_post = i in t.pre, i in t.post
            if is_pre and not is_post:
                local = bdd.apply_and(x, bdd.apply_not(xp))
            elif is_post and not is_pre:
                local = bdd.apply_and(bdd.apply_not(x), xp)
            elif is_pre and is_post:
                local = bdd.apply_and(x, xp)
            else:
                case1 = bdd.apply_and(x, xp)
                case2 = bdd.apply_and(bdd.apply_not(x), bdd.apply_not(xp))
                local = bdd.apply_or(case1, case2)
            trans_bdd = bdd.apply_and(trans_bdd, local)
        T = bdd.apply_or(T, trans_bdd)

    x_set = set(x_vars)
    rename_map = {xp_vars[i]: x_vars[i] for i in range(n)}
    while True:
        and_res = bdd.apply_and(R, T)
        exists_res = bdd.exist(x_set, and_res)
        next_states = bdd.let(rename_map, exists_res)
        not_R = bdd.apply_not(R)
        new_states = bdd.apply_and(next_states, not_R)
        if new_states == bdd.false_node: break
        R = bdd.apply_or(R, new_states)
        
    return BDDResult(bdd, R, x_vars, xp_vars)


# =============================================================================
# PART 4: OPTIMIZATION STRATEGIES
# =============================================================================

def solve_task5_explicit(net: PetriNet, visited_states: Set[int]):
    """
    Method: Linear Scan of Visited States (Explicit)
    """
    best_val = -1
    best_marking = None
    
    for m_bits in visited_states:
        current_marking = []
        token_count = 0
        for i in range(net.num_places):
            if (m_bits >> i) & 1:
                token_count += 1
                current_marking.append(1)
            else:
                current_marking.append(0)
        
        if token_count > best_val:
            best_val = token_count
            best_marking = current_marking
            
    return best_marking, best_val

def solve_task5_symbolic(net: PetriNet, bdd_res: BDDResult):
    """
    Method: ILP + BDD Filtering WITH STATE EQUATION
    """
    if pulp is None: return None, "PuLP missing"
        
    prob = pulp.LpProblem("MaxTokens", pulp.LpMaximize)
    
    m_vars = [pulp.LpVariable(f"m_{i}", cat='Binary') for i in range(net.num_places)]
    y_vars = [pulp.LpVariable(f"y_{i}", lowBound=0, cat='Integer') for i in range(len(net.transitions))]
    
    # Objective
    prob += pulp.lpSum(m_vars)
    
    # State Equation Constraints: M = M0 + C*Y
    for p_idx, place in enumerate(net.places):
        expr = net.initial[p_idx] # M0
        for t_idx, trans in enumerate(net.transitions):
            if p_idx in trans.post:
                expr += y_vars[t_idx]
            if p_idx in trans.pre:
                expr -= y_vars[t_idx]
        prob += (m_vars[p_idx] == expr)

    best_marking = None
    best_val = -1
    
    # Limit iterations to avoid infinite run if BDD check keeps failing
    max_iter = 50 
    iter_count = 0
    
    while iter_count < max_iter:
        iter_count += 1
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal: break
            
        candidate_list = []
        assignment = {}
        for i in range(net.num_places):
            val = int(pulp.value(m_vars[i]))
            if val == 1: assignment[f"x{i}"] = 1
            else: assignment[f"x{i}"] = 0
            candidate_list.append(val)
            
        is_reachable = bdd_res.manager.evaluate(bdd_res.reachable_node, assignment)
        
        if is_reachable:
            best_marking = candidate_list
            best_val = sum(candidate_list)
            break 
        else:
            # Cut: forbid this exact candidate
            current_ones = [m_vars[i] for i in range(net.num_places) if candidate_list[i] == 1]
            current_zeros = [m_vars[i] for i in range(net.num_places) if candidate_list[i] == 0]
            prob += (pulp.lpSum(current_ones) - pulp.lpSum(current_zeros)) <= (len(current_ones) - 1)

    return best_marking, best_val


# =============================================================================
# PART 5: MAIN & FORMATTING
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python testing.py <file.pnml>")
        return

    pnml_file = sys.argv[1]
    if not os.path.exists(pnml_file):
        print(f"Error: File {pnml_file} not found.")
        return

    # --- TASK 1: Parsing ---
    print("\n" + "="*80)
    print(f"{'TASK 1: NETWORK SUMMARY':^80}")
    print("="*80)
    try:
        net = parse_pnml(pnml_file)
        print(f"File         : {pnml_file}")
        print(f"Places       : {net.num_places}")
        print(f"Transitions  : {len(net.transitions)}")
        print(f"Initial State: {net.initial}")
    except Exception as e:
        print(f"Parsing failed: {e}")
        return

    # --- EXECUTION: Pre-requisites ---
    
    # 1. Run BFS
    t0 = time.time()
    visited_bfs = explicit_reachability(net)
    time_bfs_construct = time.time() - t0
    count_bfs = len(visited_bfs)

    # 2. Run BDD
    t0 = time.time()
    bdd_res = symbolic_reachability(net)
    time_bdd_construct = time.time() - t0

    # --- TASK 5: OPTIMIZATION COMPARISON ---
    
    # Method A: Before Optimization (Linear Scan)
    t0 = time.time()
    opt_marking_bfs, opt_val_bfs = solve_task5_explicit(net, visited_bfs)
    time_opt_before = time.time() - t0

    # Method B: After Optimization (ILP + BDD)
    t0 = time.time()
    opt_marking_bdd, opt_val_bdd = solve_task5_symbolic(net, bdd_res)
    time_opt_after = time.time() - t0

    # --- REPORT GENERATION ---
    print("\n" + "="*80)
    print(f"{'TASK 2 vs 3: REACHABILITY CONSTRUCTION':^80}")
    print("="*80)
    print(f"{'Metric':<35} | {'BFS (Explicit)':<20} | {'BDD (Symbolic)':<20}")
    print("-" * 80)
    print(f"{'Construction Time (s)':<35} | {time_bfs_construct:<20.5f} | {time_bdd_construct:<20.5f}")
    print(f"{'State Count':<35} | {count_bfs:<20} | {count_bfs:<20}")
    print("-" * 80)

    print("\n" + "="*80)
    print(f"{'TASK 5: OPTIMIZATION COMPARISON (Reachable Markings)':^80}")
    print("="*80)
    print(f"Objective: Maximize total tokens")
    print("-" * 80)
    print(f"{'Metric':<35} | {'Before Optimization':<20} | {'After Optimization':<20}")
    print(f"{'':<35} | {'(Explicit Scan)':<20} | {'(BDD + ILP)':<20}")
    print("-" * 80)
    print(f"{'Search Time (s)':<35} | {time_opt_before:<20.5f} | {time_opt_after:<20.5f}")
    print(f"{'Optimal Value':<35} | {opt_val_bfs:<20} | {opt_val_bdd:<20}")
    print("-" * 80)
    
    if opt_marking_bdd:
        print(f"\nFinal Result (from BDD+ILP):")
        print(f" > Optimal Marking Vector: {opt_marking_bdd}")
        active_places = [net.places[i].name for i, x in enumerate(opt_marking_bdd) if x == 1]
        print(f" > Active Places: {', '.join(active_places)}")
    else:
        print("No feasible marking found.")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()