# Walkthrough — GhostSig REST API + Analyst Dashboard MVP (Day 1 Production Readiness)

## Changes Made

### 1. Startup Script
- Created [start.ps1](start.ps1) in the project root to automate the Day 1 startup process:
  - Verifies if the Docker Desktop daemon is running (with a 30s timeout retry loop).
  - Starts the infrastructure containers (`docker compose up -d`).
  - Polls `pg_isready` inside the postgres container to wait for the database to be fully ready.
  - Activates the virtual environment and launches the FastAPI server using `uvicorn` in a separate PowerShell window.
  - Launches the Streamlit analyst dashboard in another separate PowerShell window.
  - Prints the local endpoints for access.

### 2. Streamlit UI Fixes
- Addressed Streamlit deprecation warnings in [app.py](dashboard/app.py):
  - Replaced all instances of `use_container_width=True` with `width='stretch'` to comply with the latest Streamlit layout API.

### 3. API Parameter Validation & Error Handling
- Modified [main.py](api/main.py) to validate input UUID paths:
  - Updated path parameter type from `str` to `UUID` in the campaigns detail, evidence, and PDF routes.
  - FastAPI now natively handles malformed UUID inputs and returns a proper `422 Unprocessable Entity` response, preventing psycopg from throwing unhandled `psycopg.errors.InvalidTextRepresentation` exceptions (which previously returned `500 Internal Server Error`).

### 4. Tests
- Added a validation test `test_campaign_invalid_uuid` to [test_api.py](tests/test_api.py) to verify the malformed UUID parameter validation (ensuring 422 HTTP responses).

## Validation Results

### Automated Tests
Successfully executed the entire test suite of **44 tests** (including the new API validation test):
```
================ 44 passed, 230 warnings in 101.02s (0:01:41) =================
```
All tests passed with zero failures.

### Live API Verification
Created and executed a live endpoint validation script (`verify_endpoints.py`) to verify that all 10 endpoints handle requests against the live database properly, returning clean `200`, `404`, and `422` responses with zero `500` server errors:
```
[Health] GET /health -> HTTP 200
[List Campaigns] GET /campaigns -> HTTP 200
[Get Campaign (Real)] GET /campaigns/47a5920c-4234-4658-8c9b-1f5ec8682130 -> HTTP 200
[Get Campaign (Malformed UUID)] GET /campaigns/invalid-uuid-1234 -> HTTP 422
[Get Campaign (Nonexistent)] GET /campaigns/00000000-0000-0000-0000-000000000000 -> HTTP 404
[Get Evidence (Real)] GET /campaigns/47a5920c-4234-4658-8c9b-1f5ec8682130/evidence -> HTTP 200
[Get Evidence (Malformed UUID)] GET /campaigns/invalid-uuid-1234/evidence -> HTTP 422
[Get Evidence (Nonexistent)] GET /campaigns/00000000-0000-0000-0000-000000000000/evidence -> HTTP 404
[Get PDF (Real)] GET /campaigns/47a5920c-4234-4658-8c9b-1f5ec8682130/pdf -> HTTP 200
[Get PDF (Malformed UUID)] GET /campaigns/invalid-uuid-1234/pdf -> HTTP 422
[Get PDF (Nonexistent)] GET /campaigns/00000000-0000-0000-0000-000000000000/pdf -> HTTP 404
[Get Account Fingerprint (Real)] GET /accounts/twitter:embed_test_user_0/fingerprint -> HTTP 200
[Get Account Fingerprint (Nonexistent)] GET /accounts/nonexistent:user/fingerprint -> HTTP 404
[List Fingerprints] GET /fingerprints -> HTTP 200
[Screen Accounts] POST /screen -> HTTP 200
[List Operators] GET /operators -> HTTP 200
[Get Operator Campaigns] GET /operators/op_xyz123/campaigns -> HTTP 200
```
All endpoints verified successfully.

