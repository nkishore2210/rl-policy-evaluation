# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
           V[s]+=prob*(reward+gamma *prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V
```

## OUTPUT:

### POLICY 1:

<img width="572" height="171" alt="image" src="https://github.com/user-attachments/assets/07aef7cf-cdd1-45a4-8e49-f02abe56b592" />

<img width="572" height="145" alt="image" src="https://github.com/user-attachments/assets/0cdbaa73-5836-4059-82b4-fcff846c5f13" />

### POLICY 2:

<img width="564" height="169" alt="image" src="https://github.com/user-attachments/assets/0df4fec9-3600-498b-871d-eff5e398f695" />

<img width="595" height="150" alt="image" src="https://github.com/user-attachments/assets/8bda730b-1782-4260-ae38-6324df108c1e" />

<img width="486" height="43" alt="image" src="https://github.com/user-attachments/assets/600b8552-9c10-4924-ad4c-8a1f12f21ed4" />

## RESULT:
Thus, The Python program to evaluate the given policy is successfully executed.

