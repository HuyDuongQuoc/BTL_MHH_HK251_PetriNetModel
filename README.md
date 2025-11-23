# CO2011 – Mathematical Modeling Assignment  

This project implements all tasks of the assignment:

1. PNML parsing for 1-safe Place/Transition nets  
2. Explicit (BFS) reachability  
3. BDD-based symbolic reachability  
4. ILP + BDD deadlock detection  
5. Optimization over reachable markings using ILP + BDD filtering  

---

## 1. Requirements

- **Language**: Python 3.8+ (tested with Python 3.x)
- **External libraries**:
  - [`dd`](https://pypi.org/project/dd/) – Binary Decision Diagrams (BDD)
  - [`pulp`](https://pypi.org/project/PuLP/) – Integer Linear Programming

Install dependencies (recommended):

```bash
pip install dd pulp
# or
python -m pip install dd pulp

##How to run
From the folder containing `main.py` and your PNML model:

```bash
python main.py <model.pnml> [--weights w0 w1 ... w_{n-1}] [--sense max|min]
