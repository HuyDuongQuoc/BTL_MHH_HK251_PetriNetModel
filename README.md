# CO2011 – Mathematical Modeling Assignment

This project implements the tasks for the Petri Net analysis assignment, featuring a **custom-built BDD engine** to optimize state-space exploration.

### **Features Implemented**
1. **PNML Parsing**: Reads 1-safe Place/Transition nets from `.pnml` files.
2. **Explicit Reachability**: Standard Breadth-First Search (BFS) for state enumeration.
3. **Symbolic Reachability**: **Custom BDD implementation** (no external BDD library) for optimized state management.
4. **Optimization**: Finds the reachable marking that maximizes tokens using a hybrid **ILP + BDD** approach (State Equation + Symbolic Verification).
5. **Performance Reporting**: Automatically compares "Before Optimization" (BFS) vs "After Optimization" (BDD+ILP) metrics.

---

## 1. Requirements

- **Language**: Python 3.8+
- **External Libraries**:
  - [`pulp`](https://pypi.org/project/PuLP/) – Used for the Integer Linear Programming (ILP) solver in Task 5.
  - *(Note: The BDD engine is implemented from scratch, so `dd` or `cudd` are NOT required.)*

### **Installation**

Install the required ILP solver library:

```bash
pip install pulp
# or
python -m pip install pulp

#command promt
python main.py <test.pnml>