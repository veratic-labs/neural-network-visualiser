# AGENTS.md

## Project Context

This project is a real-time Neural Network Training Visualiser built with PyTorch, FastAPI, HTML, CSS, and JavaScript.

Read `plan.md` for the full project architecture, requirements, milestones, and acceptance criteria.

## Development Workflow

1. Work on only one milestone at a time.
2. Never proceed to the next milestone without explicit user approval.
3. Do not implement features outside the currently authorised milestone.
4. After completing a milestone, stop and report the results.
5. Wait for the user to manually test and approve the implementation.
6. If the user reports problems, fix only those problems before requesting another review.

## Coding Rules

- Keep the existing PyTorch training logic intact unless changes are required by the current milestone.
- Follow the architecture defined in `plan.md`.
- Keep frontend rendering separate from backend training logic.
- Avoid unnecessary dependencies, abstractions, and refactoring.
- Prefer simple, readable, maintainable code.
- Do not silently change project requirements.

## Existing Training Code

The `training/` directory contains the existing PyTorch training implementation, data-loading code, and MNIST dataset.

Rules:
- Inspect the existing files before implementing any training-related functionality.
- Reuse the existing neural network and data-loading logic.
- Do not create a separate training implementation.
- Do not modify the existing training code unless necessary for the current milestone.
- Do not move, rename, or delete files inside `training/` without explicit approval.
- Do not download or duplicate the dataset unnecessarily.
- Keep the training code separate from the visualisation frontend.

## Testing and Verification

After each milestone:

1. Run relevant automated tests.
2. Report which tests passed or failed.
3. Explain which files were created or modified.
4. Provide exact instructions for manually testing the milestone.
5. Clearly identify any incomplete functionality.

Do not claim that a feature works unless it has been verified.

## Git Rules

- Do not create commits without user approval.
- Do not push changes to remote repositories without user approval.
- Keep changes limited to the authorised milestone.

## Approval Gate

A milestone is complete only when the user explicitly approves it.

Never interpret successful automated tests as permission to continue to the next milestone.