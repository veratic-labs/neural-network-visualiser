# Neural Network Training Visualiser

Real-time visualiser for the existing PyTorch MNIST MLP in `training/`
(`784 → 128 → 64 → 10`, batch size 1). See `plan.md` for the full design and
milestones.

**Current state: Milestone 1 — static frontend with clearly marked mock data.**
There is no backend, no WebSocket and no training yet.

## Run the frontend locally

From the repository root:

```bash
cd frontend
python3 -m http.server 8000
```

Then open <http://localhost:8000> (WSL2 forwards `localhost` to Windows).

The page must be served over HTTP, not opened as a `file://` URL, because
`main.js` is loaded as an ES module.

## Files

| File | Purpose |
| --- | --- |
| `frontend/index.html` | Page layout: network view, sample / probability / metric panels, controls, progress bar |
| `frontend/style.css` | Styling and responsive layout |
| `frontend/network.js` | Canvas renderer for neurons and connections (full and partial modes) |
| `frontend/main.js` | Panel rendering and UI wiring (no WebSocket yet) |
| `frontend/mock.js` | Placeholder weights, sample image, probabilities, loss and progress |

`training/` is the existing PyTorch implementation and is not modified by the
frontend.
