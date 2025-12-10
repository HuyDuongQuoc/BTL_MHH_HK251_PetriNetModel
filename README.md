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
```

## 2. How to Choose weights vector

- If you do not specify --weights, the program uses weight equal 1 for all places by default.
- For running with a custom weight vector, remember to maximize reaching the global success marking (i.e.,place Done) and optionally reward having puzzles solved. 
- Suggest wieght vector for escape_room_2players.pnml: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
- Note that this suggested weight vector is chosen because the Petri net is fairly complex, which can otherwise cause the execution to freeze if we use a random or default weight vector.


## 3. How to run

From the folder containing `main.py` and your PNML model, run the command with format as below:

python main.py <test.pnml>

