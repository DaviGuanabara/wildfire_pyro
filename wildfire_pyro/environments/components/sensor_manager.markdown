
## 🔁 SensorManager Flow Overview

The `SensorManager` module is designed to provide a **clean interface for time-aware and location-aware sampling** over a large dataset of sensor readings.

### 📦 Main Responsibilities

1. **Randomly select a sensor** — and set its internal state (`step`, `_select_random_sensor`)
2. **Retrieve the current reading** — from that sensor at a random timestamp (`get_current_sensor_data`)
3. **Select neighbors** — for the sensor, using time and distance windows (`get_neighbors`, `_compute_neighbors`)
4. **Calculate feature deltas** — between the sensor and its neighbors (`get_neighbors_deltas`, `_compute_deltas`)
5. **Support bootstrapped sampling** — repeatable subsets of neighbors for robust supervised training (`get_bootstrap_neighbors[_deltas]`)

### 🔁 Typical Flow

```
reset(seed) → step() → get_current_sensor_data()
                           ↓
                  get_neighbors_deltas()
                           ↓
                     _compute_deltas()
```

Each cycle above represents one training step. Internally, caching mechanisms ensure performance by avoiding redundant computation if state hasn't changed.

---

### 📘 Didactic Explanation: Indexed Reservoir Sampling

Think of each `sensor_id` as a bucket containing rows (readings). Instead of shuffling buckets or doing groupby repeatedly, we **pre-list** the content of each bucket (the row indices). This is:

- Like organizing marbles into labeled bags (`sensor_index`)
- When sampling, we **draw bags randomly**, and **pick one marble from each bag**
- Picking is done using a seeded RNG, making results deterministic

This technique eliminates pandas overhead and gives us control over performance.

It's called *reservoir-inspired* because:

- We don’t keep all samples in memory
- We draw uniformly without replacement (unless needed)
- We avoid recomputing group structures

And it integrates smoothly with our cache-optimized pipeline.

---

## 🚀 Optimized Random Neighbor Selection Using Indexed Reservoir Strategy

### 🧠 Context

During the training loop, the `_select_random_neighbors` method was identified as a major bottleneck via `line_profiler`. The original implementation relied on `groupby(...).iloc[...]` to sample a single row per sensor, ensuring one observation per neighbor. While correct and deterministic, this strategy became prohibitively expensive as the dataset scaled.

---

### 💡 Solution

We implemented an **optimized indexed reservoir-inspired strategy** that avoids repeated grouping and instead leverages **precomputed row indices** for each `sensor_id`.

---

Perfeito! Abaixo está uma explicação **muito mais detalhada e didática** do funcionamento do método que você implementou — com base no conceito de *Indexed Reservoir Sampling*. A ideia é não só explicar o **"como"**, mas também o **"porquê"** de ele ser tão eficiente para o seu caso.

---

## 🧠 Indexed Reservoir Sampling — In-Depth Explanation

### 🧱 The Problem

Suppose we have a large dataset containing many sensors, and for each sensor, multiple measurements over time. For training, we need to:

* Randomly select `k` **distinct sensors** (i.e., unique sensor IDs).
* From each selected sensor, **randomly pick one reading** (i.e., one row of the DataFrame).

Critically, this must be:

* ✅ **Fast** (done millions of times during training),
* ✅ **Reproducible** (same results if seeded),
* ✅ **Evenly distributed** (no bias toward sensors with more or fewer readings).

Traditional `groupby(...).sample(...)` operations in pandas meet these needs, but are **prohibitively slow** because:

* `groupby` is expensive on large datasets,
* Every call regenerates group objects from scratch,
* The overhead scales with the number of rows, not the number of groups.

---

### 💡 The Indexed Reservoir Insight

We flip the problem:

> Instead of building groups dynamically every time we want to sample from them, why not **precompute the indices** of each group (sensor) once, and reuse them forever?

So we do this:

```python
self.sensor_index = {
    sid: df_sdf.index.tolist()
    for sid, df_sdf in self.data.groupby("sensor_id", sort=False)
}
```

Now for each `sensor_id`, we know **exactly which row indices** belong to it. This gives us O(1) access to any group’s rows.

---

### ⚙️ The Sampling Process Step-by-Step

Let’s say you want to sample `k` random neighbors from the current candidate pool. Here’s what happens:

#### 1. 🎯 Select k random sensors

```python
selected_sids = self.rng.choice(
    sids, size=num_neighbors, replace=(num_neighbors > len(sids))
)
```

This picks `k` random `sensor_id`s from the available ones — uniformly and reproducibly.

#### 2. 🎲 Pick one observation per sensor

```python
for sid in selected_sids:
    idxs = self.sensor_index.get(sid)  # list of row indices
    i = self.rng.integers(len(idxs))  # random index
    rows.append(self.data.iloc[idxs[i]])
```

This ensures that:

* We don’t rebuild DataFrames for each sensor.
* Sampling is done via fast index access (`iloc`) instead of filtering or grouping.
* The randomness is deterministic, thanks to the seed.

#### ⛏️ Why It's Like Reservoir Sampling

In the classic [Reservoir Sampling](https://en.wikipedia.org/wiki/Reservoir_sampling), you maintain a "reservoir" of size `k` and update it as elements stream in.

Here, we're not streaming one sensor at a time, but we use the same **core trick**:

* Randomly and fairly choose **one item per group**, regardless of group size.
* Use a **pre-built reservoir** (i.e., index lookup dictionary) to eliminate recomputation.

That’s why this method is often called “indexed reservoir” — it brings the spirit of reservoir sampling to **group-wise sampling from static datasets**.

---

### 📉 Performance Comparison

| Strategy                   | Rebuilds `groupby`? | Speed  | Memory  | Scalability |
| -------------------------- | ------------------- | ------ | ------- | ----------- |
| `groupby(...).sample(...)` | Yes, every call     | ❌ Slow | ⚠️ High | ❌ Poor      |
| Indexed + `iloc` sampling  | No (precomputed)    | ✅ Fast | ✅ Low   | ✅ Excellent |

---

### ✅ Why This Is a Win

* **Time complexity drops** from O(n) per sample to O(1) + O(k).
* Memory usage remains stable — we store only one list of indices per sensor.
* Sampling remains unbiased and consistent.
* Deterministic behavior with seeding.

---

### 🧪 Optional: Visual Analogy

Imagine each sensor is a **bag full of marbles**, and we want to grab one marble from `k` bags at random:

* Old method: walk into each bag, count how many marbles, pick one, repeat — every time!
* New method: label each marble’s position, keep a notebook with positions, then just flip to the right page and pluck!

---

### 📘 Further Reading

* Vitter, J. S. (1985). *Random sampling with a reservoir.*
* StackOverflow: ["Efficient one-row-per-group sampling in pandas"](https://stackoverflow.com/questions/29576430)

---

Se quiser, posso integrar essa explicação diretamente no `sensor_manager.markdown`, como um apêndice ou seção dedicada. Deseja que eu faça isso agora?

