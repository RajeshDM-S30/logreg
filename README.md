# Logistic Regression Gradient & Convexity — FAANG-Level Hands-On

**Goal:** Derive and implement logistic regression *correctly* (loss, gradient, Hessian intuition) and explain why the loss is convex.

**Outcome:** Students can implement vectorized logistic regression, gradient-check it, and communicate convexity + optimization reasoning at FAANG interview depth.

---

# How to Start

1. **Fork** this repository.  
2. Open `logreg_student_lab.ipynb` in **Google Colab**.  
3. Complete all **TODO** sections.  
4. **Restart runtime → Run All** cells.  
5. Push changes and submit a **Pull Request**.  

⚠️ **Do NOT edit notebooks directly on GitHub.**

---

## Lab Rules (FAANG Style)

- ✅ Derivations must match the implementation
- ✅ Always track shapes: `X (n,d)`, `w (d,)`, `y (n,)`
- ✅ Numerical stability is required (overflow-safe sigmoid, stable log-loss)
- ✅ Verify gradients with finite differences
- ❌ No hardcoded outputs

---

# Out of Scope

- Using scikit-learn to train the model
- Multi-class softmax (covered later)

---

# Notebook Rules

- Do **NOT** rename the notebook  
- Do **NOT** delete TODOs  
- Do **NOT** hardcode outputs  
- Notebook must run **top-to-bottom**  

---

# Dataset

- Synthetic binary classification (linearly separable + non-separable variants)

## Why?

- Mirrors interview settings
- Forces focus on math + stability

---

## Section 1 — Logistic Loss & Sigmoid (Stability)

### Task 1.1: Implement a stable sigmoid

**Checkpoint Questions:**

- Why does naive `exp(-z)` overflow?
- What does sigmoid saturating imply for gradients?

---

### Task 1.2: Implement stable binary cross-entropy loss

**Interview Angle:**

- How do you compute `log(1 + exp(z))` stably?

---

## Section 2 — Gradient Derivation → Vectorized Gradient

### Task 2.1: Derive gradient of logistic regression

**Checkpoint Questions:**

- Why does the gradient simplify to `X^T (p - y) / n`?
- What is the interpretation of `(p - y)`?

---

### Task 2.2: Implement `(loss, grad)` and gradient-check

**FAANG Gotcha:**

- Most bugs are missing `1/n` constants or shape/broadcast mistakes.

---

## Section 3 — Convexity & Hessian Intuition

### Task 3.1: Implement Hessian-vector product (HVP)

**Checkpoint Questions:**

- Why is the Hessian PSD?
- What does PSD imply about convexity?

---

### Task 3.2: Empirical convexity check

Pick random direction `v` and verify:

- `f(w + tv)` is convex in `t` (finite differences)

---

## Section 4 — Optimization Thinking

### Task 4.1: One step of gradient descent vs Newton step (bonus)

**Interview Angle:**

- Why is Newton fast but expensive?
- What breaks if data is perfectly separable?

---

## Submission Expectations

Students must submit:

- Completed notebook with all TODOs
- Gradient check passing
- Short written answers to checkpoint questions

---

## FAANG Interview Evaluation Rubric

| Skill                          | Evaluated |
|--------------------------------|-----------|
| Correct derivation             | ✅        |
| Numerical stability            | ✅        |
| Gradient check discipline      | ✅        |
| Convexity intuition            | ✅        |
| Code clarity                   | ✅        |

---

## Stretch Problems (Optional)

- Add L2 regularization and update gradient
- Show separable-data failure mode (weights blow up)
- Compare GD vs Newton on convergence speed
