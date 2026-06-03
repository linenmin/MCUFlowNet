# Search Insights

---

### I-004 (retired): Severe Search Saturation Around High-Accuracy Deep Decoder Subnets

Retired. The previous assumption about severe saturation (80% duplicate rate) was inaccurate for the current state. Recent generation metrics (Epochs 12-14) show the duplicate rate has dropped significantly to ~30%, indicating the search algorithm successfully escaped local stagnation and resumed healthy exploration along the Pareto front.

---

### I-005 (active): Extreme Gene Convergence in H1 Upsampling Head Amidst High Duplicate Rates

Analysis of the search metrics reveals strong convergence in the H1 upsampling head (index 7). Verification of recent epochs confirms that the search heavily selects H1=0 (3x3 conv), with a frequency of approximately 73.3% versus 26.7% for H1=1 (5x5 conv). This pattern indicates that expanding the kernel size to 5x5 at the first upsampling layer generally yields poor EPE/FPS trade-offs across the entire Pareto front.

---

### I-006 (active): Backbone Depth Acts as the Primary Pareto Trade-off Lever

While upsampling heads exhibit low entropy, the backbone dimensions (EB1, DB0, DB1 at indices 3, 4, and 5) maintain high variance even in the late stages of the search. Verification shows near-uniform distributions for these blocks (e.g., DB0 is split 36%, 30%, 34% across depth choices 0, 1, 2). Subnets pushing the extreme low-EPE boundary (e.g., `0,1,2,2,2,2,1,1,1,1,0`) consistently max out the backbone depth blocks to Deep3 (value 2). Hardware profiling of this extreme subnet reveals that the DB0 blocks consume a significant portion of inference cycles (around 5% per layer, accumulating to ~20% for the whole block) with relatively low utilization (~30%). Conversely, high-FPS subnets utilize shallower blocks to bypass these costly operations. This structural dichotomy confirms that the NSGA-II optimizer relies primarily on backbone depth variations to traverse the EPE-FPS trade-off space.