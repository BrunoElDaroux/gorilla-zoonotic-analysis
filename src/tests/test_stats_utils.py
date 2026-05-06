# tests/test_stats_utils.py

def test_cramers_v_returns_float():
    ...
def test_bootstrap_ci_bounds():
    ...
    
# Before
def run_mann_whitney_test(group1, group2, label):

# After
def run_mann_whitney_test(group1: np.ndarray, group2: np.ndarray, label: str) -> dict:
